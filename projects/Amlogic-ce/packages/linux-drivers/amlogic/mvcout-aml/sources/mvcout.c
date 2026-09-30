// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * mvcout - vframe provider for frames written by user space (Amlogic S5)
 *
 * Software decoded video (3D MVC: both views composed into one frame packed
 * picture) goes to the video layer like a hardware decoder's output: user
 * space writes linear 8-bit frames into CMA buffers mapped from here and
 * queues them; the "mvcout" provider hands them to the vfm receiver chain
 * (e.g. "mvcout amvideo", or "mvcout amlvideo amvideo" under Kodi).
 *
 * Buffers are codec_mm allocations and carried as vf->mem_handle, so the
 * video keeper can hold the frame on screen after a stop; a kept buffer is
 * not handed out again until the keeper lets it go.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/miscdevice.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <linux/workqueue.h>
#include <linux/amlogic/media/vfm/vframe.h>
#include <linux/amlogic/media/vfm/vframe_provider.h>
#include <linux/amlogic/media/vfm/vframe_receiver.h>
#include <linux/amlogic/media/codec_mm/codec_mm.h>
#include <linux/amlogic/media/canvas/canvas.h>

#include "mvcout_uapi.h"

#define DRV_NAME	"mvcout"

static int debug;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "log provider events and frame flow");

#define mvc_dbg(fmt, ...) \
	do { if (debug) pr_info(DRV_NAME ": " fmt, ##__VA_ARGS__); } while (0)

/* BT.709, limited range, colour description present */
#define SIGNAL_BT709 ((1 << 29) | (5 << 26) | (1 << 24) | (1 << 16) | (1 << 8) | 1)

enum buf_state {
	BUF_FREE,	/* may be dequeued */
	BUF_USER,	/* dequeued, user space writes it */
	BUF_QUEUED,	/* waits in the output queue */
	BUF_SHOWN,	/* taken by the receiver */
	BUF_HELD,	/* was on screen at a stop, the keeper may hold it */
};

struct mvcout_buf {
	struct codec_mm_s *mm;
	ulong phys;
	enum buf_state state;
	unsigned long held_since;	/* jiffies */
	struct vframe_s vf;
};

struct mvcout_dev {
	struct miscdevice misc;
	struct mutex mlock;		/* ioctl, alloc/free */
	spinlock_t lock;		/* buffer states, queue */
	wait_queue_head_t wq;
	struct file *owner;
	struct vframe_provider_s prov;
	bool started;

	struct mvcout_buf bufs[MVCOUT_MAX_BUFS];
	int nbufs;
	u32 w, h, fmt, stride, size, map_size;

	int outq[MVCOUT_MAX_BUFS];
	int outq_head, outq_cnt;

	struct delayed_work free_work;
	struct mvcout_stats st;
};

static struct mvcout_dev *gdev;

/* ------------------------------------------------------------------ */
/* buffers                                                            */
/* ------------------------------------------------------------------ */

static bool buf_kept(struct mvcout_buf *b)
{
	return b->mm && atomic_read(&b->mm->use_cnt) > 1;
}

/* a held buffer comes back once the keeper let it go (d->lock held) */
static bool buf_free_locked(struct mvcout_buf *b)
{
	/* not kept: the display may still scan it out for a frame or two */
	if (b->state == BUF_HELD && !buf_kept(b) &&
	    time_after(jiffies, b->held_since + msecs_to_jiffies(100)))
		b->state = BUF_FREE;
	return b->state == BUF_FREE;
}

static void free_bufs(struct mvcout_dev *d)
{
	int i;

	/* a keeper reference keeps its frame alive past this */
	for (i = 0; i < d->nbufs; i++) {
		if (d->bufs[i].mm)
			codec_mm_release(d->bufs[i].mm, DRV_NAME);
		d->bufs[i].mm = NULL;
		d->bufs[i].phys = 0;
		d->bufs[i].state = BUF_FREE;
	}
	d->nbufs = 0;
}

static void free_work_fn(struct work_struct *work)
{
	struct mvcout_dev *d = container_of(to_delayed_work(work),
					    struct mvcout_dev, free_work);

	mutex_lock(&d->mlock);
	if (!d->owner && !d->started && d->nbufs) {
		free_bufs(d);
		pr_info(DRV_NAME ": buffers released\n");
	}
	mutex_unlock(&d->mlock);
}

static void build_vf(struct mvcout_dev *d, int idx, u32 w, u32 h)
{
	struct mvcout_buf *b = &d->bufs[idx];
	struct vframe_s *vf = &b->vf;
	struct canvas_config_s *c = vf->canvas0_config;

	memset(vf, 0, sizeof(*vf));
	INIT_LIST_HEAD(&vf->list);
	vf->index = idx;
	vf->width = w;
	vf->height = h;
	vf->type = VIDTYPE_PROGRESSIVE | VIDTYPE_VIU_FIELD;
	vf->flag = VFRAME_FLAG_VIDEO_LINEAR;
	vf->bitdepth = BITDEPTH_Y8 | BITDEPTH_U8 | BITDEPTH_V8;
	vf->signal_type = SIGNAL_BT709;
	vf->canvas0Addr = (u32)-1;
	vf->canvas1Addr = (u32)-1;
	if (d->fmt == MVCOUT_FMT_YUV422) {
		vf->type |= VIDTYPE_VIU_422 | VIDTYPE_VIU_SINGLE_PLANE;
		vf->plane_num = 1;
		c[0].phy_addr = b->phys;
		c[0].width = d->stride;
		c[0].height = h;
	} else {
		vf->type |= VIDTYPE_VIU_NV21;
		vf->plane_num = 2;
		c[0].phy_addr = b->phys;
		c[0].width = d->stride;
		c[0].height = h;
		c[1].phy_addr = b->phys + d->stride * d->h;
		c[1].width = d->stride;
		c[1].height = h / 2;
	}
	c[0].block_mode = c[1].block_mode = CANVAS_BLKMODE_LINEAR;
	memcpy(vf->canvas1_config, vf->canvas0_config, sizeof(vf->canvas1_config));
	vf->type_original = vf->type;
	vf->type_backup = vf->type;
	vf->mem_handle = b->mm;
}

static long ioc_alloc(struct mvcout_dev *d, void __user *arg)
{
	struct mvcout_alloc a;
	unsigned long flags;
	int i;

	if (copy_from_user(&a, arg, sizeof(a)))
		return -EFAULT;
	if (d->started)
		return -EBUSY;
	if (a.width < 64 || a.width > 4096 || a.height < 64 || a.height > 4400 ||
	    a.count < 2 || a.count > MVCOUT_MAX_BUFS ||
	    a.format > MVCOUT_FMT_YUV422 ||
	    (a.format == MVCOUT_FMT_NV21 && (a.width | a.height) & 1))
		return -EINVAL;

	cancel_delayed_work_sync(&d->free_work);
	free_bufs(d);

	d->w = a.width;
	d->h = a.height;
	d->fmt = a.format;
	if (a.format == MVCOUT_FMT_YUV422) {
		d->stride = ALIGN(a.width * 2, 64);
		d->size = d->stride * a.height;
	} else {
		d->stride = ALIGN(a.width, 64);
		d->size = d->stride * a.height * 3 / 2;
	}
	d->map_size = PAGE_ALIGN(d->size);

	for (i = 0; i < a.count; i++) {
		struct mvcout_buf *b = &d->bufs[i];

		b->mm = codec_mm_alloc(DRV_NAME, d->map_size, 0, CODEC_MM_FLAGS_DMA);
		if (!b->mm) {
			pr_err(DRV_NAME ": allocation of buffer %d (%u KiB) failed\n",
			       i, d->map_size >> 10);
			d->nbufs = i;
			free_bufs(d);
			return -ENOMEM;
		}
		b->phys = b->mm->phy_addr;
		b->state = BUF_FREE;
	}
	spin_lock_irqsave(&d->lock, flags);
	d->nbufs = a.count;
	d->outq_head = d->outq_cnt = 0;
	spin_unlock_irqrestore(&d->lock, flags);

	pr_info(DRV_NAME ": %d buffers %ux%u %s, %u KiB each\n", a.count, a.width,
		a.height, a.format == MVCOUT_FMT_YUV422 ? "yuv422" : "nv21",
		d->map_size >> 10);
	a.stride = d->stride;
	a.size = d->size;
	a.map_size = d->map_size;
	return copy_to_user(arg, &a, sizeof(a)) ? -EFAULT : 0;
}

/* ------------------------------------------------------------------ */
/* provider                                                           */
/* ------------------------------------------------------------------ */

static struct mvcout_buf *buf_of(struct mvcout_dev *d, struct vframe_s *vf)
{
	int i;

	for (i = 0; i < d->nbufs; i++)
		if (vf == &d->bufs[i].vf)
			return &d->bufs[i];
	return NULL;
}

static struct vframe_s *mvcout_peek(void *op_arg)
{
	struct mvcout_dev *d = op_arg;
	struct vframe_s *vf = NULL;
	unsigned long flags;

	spin_lock_irqsave(&d->lock, flags);
	if (d->outq_cnt)
		vf = &d->bufs[d->outq[d->outq_head]].vf;
	spin_unlock_irqrestore(&d->lock, flags);
	return vf;
}

static struct vframe_s *mvcout_get(void *op_arg)
{
	struct mvcout_dev *d = op_arg;
	struct vframe_s *vf = NULL;
	unsigned long flags;

	spin_lock_irqsave(&d->lock, flags);
	if (d->outq_cnt) {
		struct mvcout_buf *b = &d->bufs[d->outq[d->outq_head]];

		d->outq_head = (d->outq_head + 1) % MVCOUT_MAX_BUFS;
		d->outq_cnt--;
		b->state = BUF_SHOWN;
		vf = &b->vf;
		d->st.taken++;
	}
	spin_unlock_irqrestore(&d->lock, flags);
	return vf;
}

static void mvcout_put(struct vframe_s *vf, void *op_arg)
{
	struct mvcout_dev *d = op_arg;
	struct mvcout_buf *b;
	unsigned long flags;

	if (!vf)
		return;
	spin_lock_irqsave(&d->lock, flags);
	b = buf_of(d, vf);
	if (b && (b->state == BUF_SHOWN || b->state == BUF_HELD)) {
		/* returned by the display: only the keeper may still hold it */
		b->state = buf_kept(b) ? BUF_HELD : BUF_FREE;
		b->held_since = jiffies - HZ;
		d->st.returned++;
	}
	spin_unlock_irqrestore(&d->lock, flags);
	wake_up_interruptible(&d->wq);
}

static int mvcout_event(int type, void *data, void *op_arg)
{
	mvc_dbg("receiver event 0x%x\n", type);
	return 0;
}

static int mvcout_states(struct vframe_states *states, void *op_arg)
{
	struct mvcout_dev *d = op_arg;
	unsigned long flags;
	int i, nfree = 0;

	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->nbufs; i++)
		if (d->bufs[i].state == BUF_FREE)
			nfree++;
	states->vf_pool_size = d->nbufs;
	states->buf_free_num = nfree;
	states->buf_recycle_num = 0;
	states->buf_avail_num = d->outq_cnt;
	spin_unlock_irqrestore(&d->lock, flags);
	return 0;
}

static const struct vframe_operations_s mvcout_ops = {
	.peek = mvcout_peek,
	.get = mvcout_get,
	.put = mvcout_put,
	.event_cb = mvcout_event,
	.vf_states = mvcout_states,
};

/* drop queued frames; frames on screen come back through put (d->lock held) */
static void flush_locked(struct mvcout_dev *d)
{
	while (d->outq_cnt) {
		d->bufs[d->outq[d->outq_head]].state = BUF_FREE;
		d->outq_head = (d->outq_head + 1) % MVCOUT_MAX_BUFS;
		d->outq_cnt--;
	}
}

static long ioc_start(struct mvcout_dev *d)
{
	unsigned long flags;

	if (!d->nbufs)
		return -EINVAL;
	if (d->started)
		return 0;
	spin_lock_irqsave(&d->lock, flags);
	flush_locked(d);
	spin_unlock_irqrestore(&d->lock, flags);
	vf_provider_init(&d->prov, MVCOUT_PROVIDER, &mvcout_ops, d);
	vf_reg_provider(&d->prov);
	vf_notify_receiver(MVCOUT_PROVIDER, VFRAME_EVENT_PROVIDER_START, NULL);
	d->started = true;
	mvc_dbg("started\n");
	return 0;
}

static void do_stop(struct mvcout_dev *d)
{
	unsigned long flags;
	int i;

	if (!d->started)
		return;
	spin_lock_irqsave(&d->lock, flags);
	flush_locked(d);
	spin_unlock_irqrestore(&d->lock, flags);
	/* the receiver puts what it holds or hands it to the keeper */
	vf_unreg_provider(&d->prov);
	d->started = false;
	spin_lock_irqsave(&d->lock, flags);
	d->st.held = 0;
	for (i = 0; i < d->nbufs; i++) {
		struct mvcout_buf *b = &d->bufs[i];

		if (b->state == BUF_SHOWN) {
			b->state = BUF_HELD;
			b->held_since = jiffies;
		}
		if (b->state == BUF_HELD && !buf_free_locked(b))
			d->st.held++;
	}
	spin_unlock_irqrestore(&d->lock, flags);
	wake_up_interruptible(&d->wq);
	pr_info(DRV_NAME ": stopped (queued %llu, taken %llu, returned %llu, held %u)\n",
		d->st.queued, d->st.taken, d->st.returned, d->st.held);
}

/*
 * Seek: drop queued frames and take back what the receivers hold. amlvideo
 * forgets its queue on RESET without returning frames, amvideo keeps the
 * frame on screen through the keeper (light unreg).
 */
static void do_flush(struct mvcout_dev *d)
{
	unsigned long flags;
	int i;

	spin_lock_irqsave(&d->lock, flags);
	flush_locked(d);
	for (i = 0; i < d->nbufs; i++) {
		struct mvcout_buf *b = &d->bufs[i];

		if (b->state == BUF_SHOWN) {
			b->state = BUF_HELD;
			b->held_since = jiffies;
		}
	}
	spin_unlock_irqrestore(&d->lock, flags);
	if (d->started)
		vf_notify_receiver(MVCOUT_PROVIDER, VFRAME_EVENT_PROVIDER_RESET, NULL);
	wake_up_interruptible(&d->wq);
	mvc_dbg("flushed\n");
}

/* ------------------------------------------------------------------ */
/* user space                                                         */
/* ------------------------------------------------------------------ */

static int take_free(struct mvcout_dev *d)
{
	unsigned long flags;
	int i, idx = -1;

	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->nbufs; i++) {
		if (buf_free_locked(&d->bufs[i])) {
			d->bufs[i].state = BUF_USER;
			idx = i;
			break;
		}
	}
	spin_unlock_irqrestore(&d->lock, flags);
	return idx;
}

static bool take_free_ready(struct mvcout_dev *d)
{
	unsigned long flags;
	bool ready = false;
	int i;

	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->nbufs && !ready; i++)
		ready = buf_free_locked(&d->bufs[i]);
	spin_unlock_irqrestore(&d->lock, flags);
	return ready;
}

static long ioc_dequeue(struct mvcout_dev *d, void __user *arg)
{
	struct mvcout_frame f;
	unsigned long deadline;
	int idx;
	long ret;

	if (copy_from_user(&f, arg, sizeof(f)))
		return -EFAULT;
	if (!d->nbufs)
		return -EINVAL;
	deadline = jiffies + msecs_to_jiffies(f.timeout_ms);
	/* a held buffer frees itself without an event: recheck every 10 ms */
	while ((idx = take_free(d)) < 0) {
		if (time_after_eq(jiffies, deadline))
			return -EAGAIN;
		ret = wait_event_interruptible_timeout(d->wq, take_free_ready(d),
				min_t(unsigned long, deadline - jiffies,
				      msecs_to_jiffies(10)));
		if (ret < 0)
			return ret;
	}
	f.index = idx;
	return copy_to_user(arg, &f, sizeof(f)) ? -EFAULT : 0;
}

static long ioc_queue(struct mvcout_dev *d, void __user *arg)
{
	struct mvcout_frame f;
	struct mvcout_buf *b;
	unsigned long flags;

	if (copy_from_user(&f, arg, sizeof(f)))
		return -EFAULT;
	if (f.index < 0 || f.index >= d->nbufs)
		return -EINVAL;
	b = &d->bufs[f.index];
	if (b->state != BUF_USER)
		return -EINVAL;

	if (!f.width || f.width > d->w)
		f.width = d->w;
	if (!f.height || f.height > d->h)
		f.height = d->h;
	if (d->fmt == MVCOUT_FMT_NV21)
		f.height &= ~1;
	build_vf(d, f.index, f.width, f.height);
	b->vf.duration = f.duration;
	b->vf.pts_us64 = f.pts_us;
	b->vf.pts = (u32)div_u64(f.pts_us * 9, 100);

	spin_lock_irqsave(&d->lock, flags);
	if (!d->started) {
		b->state = BUF_FREE;
		spin_unlock_irqrestore(&d->lock, flags);
		return -EPIPE;
	}
	b->state = BUF_QUEUED;
	d->outq[(d->outq_head + d->outq_cnt) % MVCOUT_MAX_BUFS] = f.index;
	d->outq_cnt++;
	d->st.queued++;
	spin_unlock_irqrestore(&d->lock, flags);

	vf_notify_receiver(MVCOUT_PROVIDER, VFRAME_EVENT_PROVIDER_VFRAME_READY, NULL);
	return 0;
}

static long mvcout_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct mvcout_dev *d = gdev;
	void __user *p = (void __user *)arg;
	long ret = 0;

	mutex_lock(&d->mlock);
	if (d->owner != file) {
		mutex_unlock(&d->mlock);
		return -EBUSY;
	}
	switch (cmd) {
	case MVCOUT_IOC_ALLOC:
		ret = ioc_alloc(d, p);
		break;
	case MVCOUT_IOC_START:
		ret = ioc_start(d);
		break;
	case MVCOUT_IOC_STOP:
		do_stop(d);
		break;
	case MVCOUT_IOC_DEQUEUE:
		/* waits without the mutex: queue/flush may not block on it */
		mutex_unlock(&d->mlock);
		return ioc_dequeue(d, p);
	case MVCOUT_IOC_QUEUE:
		ret = ioc_queue(d, p);
		break;
	case MVCOUT_IOC_FLUSH:
		do_flush(d);
		break;
	case MVCOUT_IOC_STATS:
		ret = copy_to_user(p, &d->st, sizeof(d->st)) ? -EFAULT : 0;
		break;
	default:
		ret = -ENOTTY;
	}
	mutex_unlock(&d->mlock);
	return ret;
}

static int mvcout_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct mvcout_dev *d = gdev;
	unsigned long len = vma->vm_end - vma->vm_start;
	unsigned long idx;
	int ret = -EINVAL;

	mutex_lock(&d->mlock);
	if (d->owner != file || !d->map_size)
		goto out;
	idx = vma->vm_pgoff / (d->map_size >> PAGE_SHIFT);
	if (vma->vm_pgoff % (d->map_size >> PAGE_SHIFT) || idx >= d->nbufs ||
	    len > d->map_size)
		goto out;
	/* CPU writes whole lines once: write combining, no cache maintenance */
	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
	vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;
	ret = remap_pfn_range(vma, vma->vm_start, d->bufs[idx].phys >> PAGE_SHIFT,
			      len, vma->vm_page_prot);
out:
	mutex_unlock(&d->mlock);
	return ret;
}

static int mvcout_open(struct inode *inode, struct file *file)
{
	struct mvcout_dev *d = gdev;
	int ret = 0;

	mutex_lock(&d->mlock);
	if (d->owner)
		ret = -EBUSY;
	else
		d->owner = file;
	memset(&d->st, 0, sizeof(d->st));
	mutex_unlock(&d->mlock);
	return ret;
}

/* runs once every mapping is gone too */
static int mvcout_release(struct inode *inode, struct file *file)
{
	struct mvcout_dev *d = gdev;

	mutex_lock(&d->mlock);
	if (d->owner == file) {
		do_stop(d);
		d->owner = NULL;
		/* the display may still scan out the last buffer for a while */
		schedule_delayed_work(&d->free_work, msecs_to_jiffies(3000));
	}
	mutex_unlock(&d->mlock);
	return 0;
}

static const struct file_operations mvcout_fops = {
	.owner = THIS_MODULE,
	.open = mvcout_open,
	.release = mvcout_release,
	.unlocked_ioctl = mvcout_ioctl,
	.compat_ioctl = mvcout_ioctl,
	.mmap = mvcout_mmap,
};

static int __init mvcout_init(void)
{
	struct mvcout_dev *d = kzalloc(sizeof(*d), GFP_KERNEL);
	int ret;

	if (!d)
		return -ENOMEM;
	mutex_init(&d->mlock);
	spin_lock_init(&d->lock);
	init_waitqueue_head(&d->wq);
	INIT_DELAYED_WORK(&d->free_work, free_work_fn);
	d->misc.minor = MISC_DYNAMIC_MINOR;
	d->misc.name = DRV_NAME;
	d->misc.fops = &mvcout_fops;
	gdev = d;
	ret = misc_register(&d->misc);
	if (ret) {
		gdev = NULL;
		kfree(d);
		return ret;
	}
	pr_info(DRV_NAME ": ready\n");
	return 0;
}

static void __exit mvcout_exit(void)
{
	struct mvcout_dev *d = gdev;

	misc_deregister(&d->misc);
	cancel_delayed_work_sync(&d->free_work);
	mutex_lock(&d->mlock);
	do_stop(d);
	free_bufs(d);
	mutex_unlock(&d->mlock);
	kfree(d);
	gdev = NULL;
}

module_init(mvcout_init);
module_exit(mvcout_exit);

MODULE_DESCRIPTION("vframe provider for user space frames (3D MVC output) on Amlogic S5");
MODULE_LICENSE("GPL");
