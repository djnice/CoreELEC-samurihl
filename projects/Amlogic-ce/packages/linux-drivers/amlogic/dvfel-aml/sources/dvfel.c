// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * dvfel - Dolby Vision FEL composition node for Amlogic S5 (S928X)
 *
 * vfm node inserted between the HEVC decoder and the display path:
 *
 *     dvbldec (DV) / vdec.h265.00 -> dvfel -> amlvideo -> deinterlace -> amvideo
 *
 * For every 10-bit compressed (AFBC) decoder frame:
 *   (A) VICP decompresses it into a linear 10-bit 4:4:4 buffer,
 *   (G) a userspace compositor (/dev/dvfel, GPU) composes the enhancement
 *       layer into a second buffer; without a compositor the frame is kept,
 *   (B) VICP compresses the result back into an AFBC buffer owned by dvfel,
 * and a copy of the decoder vframe pointing at that buffer is passed on.
 * The copy keeps vf->index, so the Dolby Vision driver still fetches the
 * (already P8.1-converted) RPU from the decoder by index. The original
 * vframe is held until the display returns the copy.
 *
 * Two kernel threads form a pipeline so VICP and GPU work overlap:
 *   stage A: take decoder frames, (A), post GPU jobs      -> pending queue
 *   stage B: in display order wait for the job, (B), hand to display
 *
 * Linear layout: see dvfel_uapi.h. Unsupported frames (8-bit, not
 * compressed, too large, interlaced, mode 0) are passed through unchanged.
 * Supported frames wait for free buffers instead, so composed and
 * non-composed frames are never mixed because of buffer shortage.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/spinlock.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/debugfs.h>
#include <linux/uaccess.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>
#include <linux/amlogic/media/vfm/vframe.h>
#include <linux/amlogic/media/vfm/vframe_provider.h>
#include <linux/amlogic/media/vfm/vframe_receiver.h>
#include <linux/amlogic/media/codec_mm/codec_mm.h>
#include <linux/amlogic/media/vicp/vicp.h>

#include "dvfel_uapi.h"

#define DRV_NAME	"dvfel"
#define MAX_SLOTS	12
#define QUEUE_LEN	16
#define MAX_W		3840
#define MAX_H		2160

#define LIN_STRIDE(w)		((w) * 4)
#define LIN_SIZE(w, h)		DVFEL_BUF_SIZE(w, h)
/* AFBC output, 10-bit 4:2:0 worst case plus header/table */
#define FBC_BODY_SIZE(w, h)	PAGE_ALIGN((w) * (h) * 15 / 8 + 1024 * 1658 * 2)
#define FBC_HEAD_SIZE(w, h)	PAGE_ALIGN(ALIGN(w, 64) * ALIGN(h, 64) / 32)
#define FBC_TABLE_SIZE(w, h)	PAGE_ALIGN(FBC_BODY_SIZE(w, h) / PAGE_SIZE * 4)

static int mode = 2;
module_param(mode, int, 0644);
MODULE_PARM_DESC(mode, "0: passthrough, 1: VICP round trip, 2: GPU compositor when connected");

static int slots = 8;
module_param(slots, int, 0444);
MODULE_PARM_DESC(slots, "number of output AFBC buffers (2..12)");

static unsigned long long capture_pts;
module_param(capture_pts, ullong, 0644);
MODULE_PARM_DESC(capture_pts, "debug: capture the frame with this pts_us64 (0: off)");

static int gpu_timeout_ms = 100;
module_param(gpu_timeout_ms, int, 0644);
MODULE_PARM_DESC(gpu_timeout_ms, "max wait for the compositor before showing the base layer");

static int debug;
module_param(debug, int, 0644);

#define dvfel_dbg(fmt, ...) \
	do { if (debug) pr_info(DRV_NAME ": " fmt, ##__VA_ARGS__); } while (0)

enum slot_state { SLOT_FREE, SLOT_BUSY, SLOT_OUT };

struct dvfel_slot {
	enum slot_state state;
	u32 gen;
	struct vframe_s *orig;
	struct vframe_s out_vf;
	ulong body_phys, head_phys, table_phys, table_handle;
};

struct dvfel_pass {
	struct vframe_s *vf;
	u32 gen;
};

enum job_state {
	JOB_FREE,
	JOB_RESERVED,	/* stage A is decompressing into in[] */
	JOB_POSTED,	/* waiting for the compositor */
	JOB_TAKEN,	/* compositor is working on it */
	JOB_DONE,	/* compositor finished */
	JOB_FINISH,	/* stage B is compressing from in[]/out[] */
};

struct dvfel_kjob {
	enum job_state state;
	bool orphan;		/* stage B gave up; free when the compositor reports */
	bool capture;		/* debug capture at stage B */
	u32 id;
	u32 gen;
	u32 flags;
	int status;
	u64 pts_us;
	ktime_t posted;
	struct dvfel_slot *slot;
	struct vframe_s *orig;
};

struct dvfel_cbuf {
	struct dma_buf *db;
	struct dma_buf_attachment *att;
	struct sg_table *sgt;
	ulong phys;
};

/* pending queue entry, in display order */
struct dvfel_pend {
	struct vframe_s *vf;	/* ready frame (processed copy or passthrough) */
	int job;		/* >= 0: wait for this job instead */
	u32 gen;
};

/* VICP descriptors per thread (too big for the stack) */
struct vicp_ctx {
	struct vicp_data_config_s cfg;
	struct dma_data_config_s dma;
	struct vframe_s vin;
};

struct dvfel_dev {
	spinlock_t lock;
	struct vframe_receiver_s recv;
	struct vframe_provider_s prov;
	bool prov_reg;
	u32 gen;
	int nslots;

	struct vframe_s *outq[QUEUE_LEN];
	int outq_head, outq_cnt;
	struct dvfel_pend pendq[QUEUE_LEN];
	int pend_head, pend_cnt;
	struct dvfel_slot slots[MAX_SLOTS];
	struct dvfel_pass pass[QUEUE_LEN];

	struct task_struct *thread_a, *thread_b;
	wait_queue_head_t wq_a, wq_b;
	bool kick_a, kick_b;
	bool new_stream;

	/* compositor (userspace) */
	struct miscdevice misc;
	struct mutex client_lock;
	struct file *client;
	bool client_ready;
	u32 client_w, client_h;
	int client_nbufs;
	struct dvfel_cbuf cin[DVFEL_MAX_BUFS], cout[DVFEL_MAX_BUFS];
	struct dvfel_kjob jobs[DVFEL_MAX_BUFS];
	u32 job_seq;
	wait_queue_head_t wq_client;

	/* dvfel owned buffers, allocated on the first processable frame */
	struct mutex buf_lock;
	bool bufs_ok;
	ulong lin_phys;
	u32 buf_w, buf_h;
	struct delayed_work free_work;

	struct vicp_ctx ctx_a, ctx_b;

	/* stats */
	u64 frames_in, frames_proc, frames_pass, vicp_err, slot_waits;
	u64 gpu_jobs, gpu_composed, gpu_fallback, gpu_timeouts;
	s64 us_a_sum, us_b_sum, us_a_max, us_b_max, us_gpu_sum, us_gpu_max;
	u64 us_cnt, us_b_cnt, us_gpu_cnt;

	/* debug capture of the next frame (A output and A of our B output) */
	struct dentry *dbg;
	atomic_t capture_req;
	void *cap_buf;
	size_t cap_size;
	int cap_frames;
	u64 cap_pts_us;
	u32 cap_w, cap_h;
};

static struct dvfel_dev *gdev;

/* ------------------------------------------------------------------ */
/* dvfel owned buffers                                                */
/* ------------------------------------------------------------------ */

static void dvfel_free_bufs(struct dvfel_dev *d)
{
	int i;

	for (i = 0; i < d->nslots; i++) {
		struct dvfel_slot *s = &d->slots[i];

		if (s->table_handle)
			codec_mm_dma_free_coherent(s->table_handle);
		if (s->head_phys)
			codec_mm_free_for_dma(DRV_NAME, s->head_phys);
		if (s->body_phys)
			codec_mm_free_for_dma(DRV_NAME, s->body_phys);
		s->table_handle = s->table_phys = 0;
		s->head_phys = s->body_phys = 0;
	}
	if (d->lin_phys)
		codec_mm_free_for_dma(DRV_NAME, d->lin_phys);
	d->lin_phys = 0;
	d->bufs_ok = false;
}

static ulong alloc_dma(u32 size)
{
	return codec_mm_alloc_for_dma(DRV_NAME, PAGE_ALIGN(size) / PAGE_SIZE, 0,
				      CODEC_MM_FLAGS_DMA);
}

static int dvfel_alloc_bufs(struct dvfel_dev *d, u32 w, u32 h)
{
	int i;

	d->lin_phys = alloc_dma(LIN_SIZE(w, h));
	if (!d->lin_phys)
		goto fail;

	for (i = 0; i < d->nslots; i++) {
		struct dvfel_slot *s = &d->slots[i];
		u32 body = FBC_BODY_SIZE(w, h), j, *tbl;
		ulong phys;

		s->body_phys = alloc_dma(body);
		s->head_phys = alloc_dma(FBC_HEAD_SIZE(w, h));
		tbl = codec_mm_dma_alloc_coherent(&s->table_handle, &phys,
						  FBC_TABLE_SIZE(w, h), DRV_NAME);
		if (!s->body_phys || !s->head_phys || !tbl)
			goto fail;
		/* the AFBCE MMU table holds 20-bit page numbers */
		if (s->body_phys + body > 0x100000000ULL)
			goto fail;
		s->table_phys = phys;
		for (j = 0; j < body; j += PAGE_SIZE)
			*tbl++ = ((s->body_phys + j) >> PAGE_SHIFT) & 0xfffff;
	}
	d->buf_w = w;
	d->buf_h = h;
	d->bufs_ok = true;
	pr_info(DRV_NAME ": buffers allocated for %ux%u (lin %u KiB, fbc %u KiB x%d)\n",
		w, h, LIN_SIZE(w, h) >> 10,
		(FBC_BODY_SIZE(w, h) + FBC_HEAD_SIZE(w, h)) >> 10, d->nslots);
	return 0;
fail:
	pr_err(DRV_NAME ": buffer allocation failed\n");
	dvfel_free_bufs(d);
	return -ENOMEM;
}

static bool dvfel_idle_locked(struct dvfel_dev *d)
{
	int i;

	if (d->prov_reg)
		return false;
	for (i = 0; i < d->nslots; i++)
		if (d->slots[i].state == SLOT_BUSY)
			return false;
	return true;
}

static void dvfel_free_work(struct work_struct *work)
{
	struct dvfel_dev *d = container_of(to_delayed_work(work),
					   struct dvfel_dev, free_work);
	unsigned long flags;
	bool idle;

	spin_lock_irqsave(&d->lock, flags);
	idle = dvfel_idle_locked(d);
	spin_unlock_irqrestore(&d->lock, flags);

	mutex_lock(&d->buf_lock);
	if (idle && d->bufs_ok) {
		dvfel_free_bufs(d);
		pr_info(DRV_NAME ": buffers released\n");
	}
	mutex_unlock(&d->buf_lock);
}

/* ------------------------------------------------------------------ */
/* queues and bookkeeping (d->lock held)                              */
/* ------------------------------------------------------------------ */

static void outq_push(struct dvfel_dev *d, struct vframe_s *vf)
{
	d->outq[(d->outq_head + d->outq_cnt) % QUEUE_LEN] = vf;
	d->outq_cnt++;
}

static struct vframe_s *outq_pop(struct dvfel_dev *d)
{
	struct vframe_s *vf;

	if (!d->outq_cnt)
		return NULL;
	vf = d->outq[d->outq_head];
	d->outq_head = (d->outq_head + 1) % QUEUE_LEN;
	d->outq_cnt--;
	return vf;
}

static void pend_push(struct dvfel_dev *d, struct vframe_s *vf, int job, u32 gen)
{
	struct dvfel_pend *p = &d->pendq[(d->pend_head + d->pend_cnt) % QUEUE_LEN];

	p->vf = vf;
	p->job = job;
	p->gen = gen;
	d->pend_cnt++;
}

static void pend_pop(struct dvfel_dev *d)
{
	d->pend_head = (d->pend_head + 1) % QUEUE_LEN;
	d->pend_cnt--;
}

static struct dvfel_slot *slot_of(struct dvfel_dev *d, struct vframe_s *vf)
{
	int i;

	for (i = 0; i < d->nslots; i++)
		if (&d->slots[i].out_vf == vf)
			return &d->slots[i];
	return NULL;
}

static struct dvfel_slot *slot_free(struct dvfel_dev *d)
{
	int i;

	for (i = 0; i < d->nslots; i++)
		if (d->slots[i].state == SLOT_FREE)
			return &d->slots[i];
	return NULL;
}

static struct dvfel_pass *pass_free(struct dvfel_dev *d)
{
	int i;

	for (i = 0; i < QUEUE_LEN; i++)
		if (!d->pass[i].vf)
			return &d->pass[i];
	return NULL;
}

static int job_free(struct dvfel_dev *d)
{
	int i;

	for (i = 0; i < d->client_nbufs; i++)
		if (d->jobs[i].state == JOB_FREE)
			return i;
	return -1;
}

/* release a job whose buffers stage B no longer needs */
static void job_release(struct dvfel_dev *d, struct dvfel_kjob *j)
{
	if (j->state == JOB_TAKEN) {
		/* the compositor still writes out[]: free on JOB_DONE/close */
		j->orphan = true;
	} else {
		j->state = JOB_FREE;
		j->orphan = false;
	}
	j->slot = NULL;
	j->orig = NULL;
}

/* drop everything we hold; frames are not returned upstream */
static void dvfel_reset_locked(struct dvfel_dev *d)
{
	int i;

	d->gen++;
	d->new_stream = true;
	d->outq_head = d->outq_cnt = 0;
	/* pending entries of the old generation are dropped by stage B */
	for (i = 0; i < d->nslots; i++)
		if (d->slots[i].state == SLOT_OUT)
			d->slots[i].state = SLOT_FREE;
	memset(d->pass, 0, sizeof(d->pass));
}

/* ------------------------------------------------------------------ */
/* VICP                                                               */
/* ------------------------------------------------------------------ */

static bool frame_supported(struct vframe_s *vf)
{
	return (vf->type & VIDTYPE_COMPRESS) &&
	       (vf->bitdepth & BITDEPTH_YMASK) == BITDEPTH_Y10 &&
	       !(vf->type & VIDTYPE_TYPEMASK) &&	/* progressive only */
	       vf->compWidth && vf->compHeight &&
	       vf->compWidth <= MAX_W && vf->compHeight <= MAX_H &&
	       !(vf->compWidth & 1) && !(vf->compHeight & 1);
}

/*
 * The write MIF derives its line stride from the background width assuming
 * 30 bits per pixel, but writes 32 bits per pixel in 10-bit 4:4:4 mode.
 * Pick the background width whose computed stride is exactly 4 * w.
 */
static u32 wmif_bg_width(u32 w)
{
	u32 bw;

	for (bw = w; bw <= w * 2; bw++)
		if (ALIGN(DIV_ROUND_UP(bw * 30, 128) * 16, 64) == LIN_STRIDE(w))
			return bw;
	return 0;
}

/* (A) AFBC frame -> linear 10-bit 4:4:4 at @dst */
static int vicp_decompress(struct vicp_ctx *c, struct vframe_s *vf,
			   ulong dst, u32 w, u32 h)
{
	u32 bw = wmif_bg_width(w);

	if (!bw)
		return -EINVAL;
	/*
	 * VICP enables its HDR->SDR block for any non-SDR source format and
	 * clears fgs_valid on its input: hand it a private copy with the
	 * source format neutralized.
	 */
	c->vin = *vf;
	c->vin.src_fmt.sei_magic_code = 0;
	c->vin.src_fmt.fmt = VFRAME_SIGNAL_FMT_INVALID;
	c->vin.fgs_valid = false;

	memset(&c->cfg, 0, sizeof(c->cfg));
	c->cfg.input_data.is_vframe = true;
	c->cfg.input_data.data_vf = &c->vin;
	c->cfg.output_data.phy_addr[0] = dst;
	c->cfg.output_data.width = bw;
	c->cfg.output_data.height = h;
	c->cfg.output_data.endian = 1;
	c->cfg.output_data.mif_out_en = 1;
	c->cfg.output_data.mif_color_fmt = VICP_COLOR_FORMAT_YUV444;
	c->cfg.output_data.mif_color_dep = 10;
	c->cfg.output_data.fbc_out_en = 0;
	c->cfg.output_data.fbc_color_fmt = VICP_COLOR_FORMAT_YUV420;
	c->cfg.output_data.fbc_color_dep = 10;
	c->cfg.output_data.out_sig_fmt = VFRAME_SIGNAL_FMT_SDR;
	c->cfg.data_option.rotation_mode = VICP_ROTATION_0;
	c->cfg.data_option.output_axis.width = w;
	c->cfg.data_option.output_axis.height = h;
	return vicp_process(&c->cfg);
}

/* (B) linear 10-bit 4:4:4 at @src -> AFBC 10-bit 4:2:0 owned by the slot */
static int vicp_compress(struct vicp_ctx *c, ulong src, struct dvfel_slot *s,
			 u32 w, u32 h)
{
	memset(&c->dma, 0, sizeof(c->dma));
	c->dma.buf_addr = src;
	c->dma.buf_stride_w = LIN_STRIDE(w);	/* bytes */
	c->dma.buf_stride_h = h;
	c->dma.data_width = w;
	c->dma.data_height = h;
	c->dma.plane_count = 1;
	c->dma.color_format = VICP_COLOR_FORMAT_YUV444;
	c->dma.color_depth = 10;
	c->dma.endian = 1;

	memset(&c->cfg, 0, sizeof(c->cfg));
	c->cfg.input_data.is_vframe = false;
	c->cfg.input_data.data_dma = &c->dma;
	c->cfg.output_data.phy_addr[0] = s->body_phys;
	c->cfg.output_data.phy_addr[1] = s->head_phys;
	c->cfg.output_data.phy_addr[2] = s->table_phys;
	c->cfg.output_data.width = w;
	c->cfg.output_data.height = h;
	c->cfg.output_data.endian = 1;
	c->cfg.output_data.mif_out_en = 0;
	c->cfg.output_data.mif_color_fmt = VICP_COLOR_FORMAT_YUV444;
	c->cfg.output_data.mif_color_dep = 10;
	c->cfg.output_data.fbc_out_en = 1;
	c->cfg.output_data.fbc_color_fmt = VICP_COLOR_FORMAT_YUV420;
	c->cfg.output_data.fbc_color_dep = 10;
	c->cfg.output_data.fbc_init_ctrl = 1;
	c->cfg.output_data.fbc_pip_mode = 1;
	c->cfg.output_data.out_sig_fmt = VFRAME_SIGNAL_FMT_SDR;
	c->cfg.data_option.rotation_mode = VICP_ROTATION_0;
	c->cfg.data_option.output_axis.width = w;
	c->cfg.data_option.output_axis.height = h;
	return vicp_process(&c->cfg);
}

/* vicp_process() reports 0 even on a 200 ms hardware timeout */
static bool vicp_ok(struct dvfel_dev *d, int ret, s64 us, const char *what)
{
	if (!ret && us < 190000)
		return true;
	d->vicp_err++;
	pr_err(DRV_NAME ": VICP %s failed (ret %d, %lld us)\n", what, ret, us);
	return false;
}

static void build_out_vf(struct dvfel_slot *s, struct vframe_s *orig, u32 w, u32 h)
{
	struct vframe_s *o = &s->out_vf;

	/* keep index, timing, signal and source info of the decoder frame */
	*o = *orig;
	INIT_LIST_HEAD(&o->list);
	o->type = VIDTYPE_PROGRESSIVE | VIDTYPE_VIU_FIELD | VIDTYPE_COMPRESS |
		  VIDTYPE_SCATTER | VIDTYPE_NO_DW;
	o->type_backup = o->type;
	o->type_original = o->type;
	o->bitdepth = BITDEPTH_Y10 | BITDEPTH_U10 | BITDEPTH_V10;
	o->flag |= VFRAME_FLAG_COMPOSER_DONE;
	o->compHeadAddr = s->head_phys;
	o->compBodyAddr = s->body_phys;
	o->compWidth = w;
	o->compHeight = h;
	o->width = w;
	o->height = h;
	o->canvas0Addr = (u32)-1;
	o->canvas1Addr = (u32)-1;
	o->plane_num = 0;
	memset(o->canvas0_config, 0, sizeof(o->canvas0_config));
	memset(o->canvas1_config, 0, sizeof(o->canvas1_config));
	o->mem_handle = NULL;
	o->mem_handle_1 = NULL;
	o->mem_head_handle = NULL;
	o->mem_dw_handle = NULL;
	o->vf_ext = NULL;
	o->uvm_vf = NULL;
	o->early_process_fun = NULL;
	o->process_fun = NULL;
	o->private_data = NULL;
	o->fence = NULL;
	o->fgs_valid = false;
}

static void stat_b(struct dvfel_dev *d, s64 ub)
{
	d->us_b_sum += ub;
	d->us_b_max = max(d->us_b_max, ub);
	d->us_b_cnt++;
}

/* ------------------------------------------------------------------ */
/* debug capture                                                      */
/* ------------------------------------------------------------------ */

static int capture_copy(struct dvfel_dev *d, ulong phys, int idx, u32 w, u32 h)
{
	size_t frame = (size_t)LIN_STRIDE(w) * h;
	void *va = codec_mm_phys_to_virt(phys);

	if (!va)
		return -EFAULT;
	if (!d->cap_buf || d->cap_size < frame * 2) {
		vfree(d->cap_buf);
		d->cap_buf = vmalloc(frame * 2);
		d->cap_size = d->cap_buf ? frame * 2 : 0;
		if (!d->cap_buf)
			return -ENOMEM;
	}
	codec_mm_dma_flush(va, frame, DMA_FROM_DEVICE);
	memcpy(d->cap_buf + frame * idx, va, frame);
	return 0;
}

/*
 * frame 0: what was sent to (B) (decoder frame or compositor output),
 * frame 1: (A) of our (B) output, i.e. what the display shows
 */
static void capture_frame(struct dvfel_dev *d, struct vframe_s *vf, ulong src,
			  struct dvfel_slot *s, u32 w, u32 h)
{
	d->cap_frames = 0;
	if (capture_copy(d, src, 0, w, h))
		return;
	d->cap_frames = 1;
	d->cap_pts_us = vf->pts_us64;
	d->cap_w = w;
	d->cap_h = h;
	if (d->bufs_ok &&
	    !vicp_decompress(&d->ctx_b, &s->out_vf, d->lin_phys, w, h) &&
	    !capture_copy(d, d->lin_phys, 1, w, h))
		d->cap_frames = 2;
	pr_info(DRV_NAME ": captured pts_us %llu (%ux%u), %d frame(s)\n",
		vf->pts_us64, w, h, d->cap_frames);
}

/* ------------------------------------------------------------------ */
/* stage A: decoder frames -> (A) -> job or direct round trip         */
/* ------------------------------------------------------------------ */

static bool ensure_bufs(struct dvfel_dev *d, u32 w, u32 h)
{
	bool ok;

	mutex_lock(&d->buf_lock);
	if (d->bufs_ok && (d->buf_w != w || d->buf_h != h)) {
		unsigned long flags;
		bool idle;

		spin_lock_irqsave(&d->lock, flags);
		idle = !d->outq_cnt && !d->pend_cnt;
		spin_unlock_irqrestore(&d->lock, flags);
		if (idle)
			dvfel_free_bufs(d);
	}
	ok = d->bufs_ok ? (d->buf_w == w && d->buf_h == h) : !dvfel_alloc_bufs(d, w, h);
	mutex_unlock(&d->buf_lock);
	return ok;
}

/* phase 1 path: (A) and (B) through the internal linear buffer */
static bool process_direct(struct dvfel_dev *d, struct dvfel_slot *s,
			   struct vframe_s *vf, bool capture)
{
	u32 w = vf->compWidth, h = vf->compHeight;
	ktime_t t0, t1, t2;
	bool ok;

	mutex_lock(&d->buf_lock);
	t0 = ktime_get();
	ok = vicp_ok(d, vicp_decompress(&d->ctx_a, vf, d->lin_phys, w, h),
		     ktime_us_delta(ktime_get(), t0), "A");
	t1 = ktime_get();
	if (ok)
		ok = vicp_ok(d, vicp_compress(&d->ctx_a, d->lin_phys, s, w, h),
			     ktime_us_delta(ktime_get(), t1), "B");
	t2 = ktime_get();
	if (ok) {
		s64 ua = ktime_us_delta(t1, t0);

		d->us_a_sum += ua;
		d->us_a_max = max(d->us_a_max, ua);
		d->us_cnt++;
		stat_b(d, ktime_us_delta(t2, t1));
		build_out_vf(s, vf, w, h);
		if (capture)
			capture_frame(d, vf, d->lin_phys, s, w, h);
	}
	mutex_unlock(&d->buf_lock);
	return ok;
}

static int dvfel_thread_a(void *data)
{
	struct dvfel_dev *d = data;

	while (!kthread_should_stop()) {
		wait_event_interruptible(d->wq_a, d->kick_a || kthread_should_stop());
		d->kick_a = false;

		for (;;) {
			struct dvfel_slot *s = NULL;
			struct dvfel_pass *p;
			struct vframe_s *vf;
			unsigned long flags;
			bool gpu = false, ok = false, capture = false;
			int j = -1;
			u32 gen, w, h;

			spin_lock_irqsave(&d->lock, flags);
			if (!d->prov_reg || d->pend_cnt >= QUEUE_LEN || !pass_free(d)) {
				spin_unlock_irqrestore(&d->lock, flags);
				break;
			}
			if (mode) {
				/* backpressure: never fall back to passthrough */
				s = slot_free(d);
				if (mode == 2 && d->client_ready) {
					j = job_free(d);
					gpu = true;
				}
				if (!s || (gpu && j < 0)) {
					d->slot_waits++;
					spin_unlock_irqrestore(&d->lock, flags);
					break;
				}
				s->state = SLOT_BUSY;
				if (gpu)
					d->jobs[j].state = JOB_RESERVED;
			}
			gen = d->gen;
			spin_unlock_irqrestore(&d->lock, flags);

			vf = vf_peek(DRV_NAME) ? vf_get(DRV_NAME) : NULL;
			if (!vf) {
				spin_lock_irqsave(&d->lock, flags);
				if (s)
					s->state = SLOT_FREE;
				if (j >= 0)
					d->jobs[j].state = JOB_FREE;
				spin_unlock_irqrestore(&d->lock, flags);
				break;
			}
			d->frames_in++;
			w = vf->compWidth;
			h = vf->compHeight;

			if (s && frame_supported(vf) && ensure_bufs(d, w, h)) {
				capture = atomic_xchg(&d->capture_req, 0) ||
					  (capture_pts && vf->pts_us64 == capture_pts);
				if (gpu && d->client_w == w && d->client_h == h) {
					ktime_t t0 = ktime_get();
					s64 ua;

					ok = vicp_ok(d, vicp_decompress(&d->ctx_a, vf,
									d->cin[j].phys, w, h),
						     ktime_us_delta(ktime_get(), t0), "A");
					ua = ktime_us_delta(ktime_get(), t0);
					if (ok) {
						d->us_a_sum += ua;
						d->us_a_max = max(d->us_a_max, ua);
						d->us_cnt++;
					}
				} else {
					if (j >= 0) {
						spin_lock_irqsave(&d->lock, flags);
						d->jobs[j].state = JOB_FREE;
						spin_unlock_irqrestore(&d->lock, flags);
						j = -1;
					}
					gpu = false;
					ok = process_direct(d, s, vf, capture);
				}
			} else if (j >= 0) {
				spin_lock_irqsave(&d->lock, flags);
				d->jobs[j].state = JOB_FREE;
				spin_unlock_irqrestore(&d->lock, flags);
				j = -1;
				gpu = false;
			}

			spin_lock_irqsave(&d->lock, flags);
			if (gen != d->gen) {
				/* reset/unreg while processing: drop the frame */
				if (s)
					s->state = SLOT_FREE;
				if (j >= 0)
					d->jobs[j].state = JOB_FREE;
				spin_unlock_irqrestore(&d->lock, flags);
				continue;
			}
			if (ok && gpu) {
				struct dvfel_kjob *jb = &d->jobs[j];

				jb->id = ++d->job_seq;
				jb->gen = gen;
				jb->pts_us = vf->pts_us64;
				jb->flags = d->new_stream ? DVFEL_JOB_NEW_STREAM : 0;
				jb->slot = s;
				jb->orig = vf;
				jb->capture = capture;
				jb->posted = ktime_get();
				jb->state = JOB_POSTED;
				d->new_stream = false;
				d->gpu_jobs++;
				pend_push(d, NULL, j, gen);
				wake_up_interruptible(&d->wq_client);
			} else if (ok) {
				s->orig = vf;
				s->gen = gen;
				s->state = SLOT_OUT;
				pend_push(d, &s->out_vf, -1, gen);
				d->frames_proc++;
			} else {
				if (s)
					s->state = SLOT_FREE;
				if (j >= 0)
					d->jobs[j].state = JOB_FREE;
				p = pass_free(d);
				p->vf = vf;
				p->gen = gen;
				pend_push(d, vf, -1, gen);
				d->frames_pass++;
			}
			spin_unlock_irqrestore(&d->lock, flags);
			d->kick_b = true;
			wake_up_interruptible(&d->wq_b);
		}
	}
	return 0;
}

/* ------------------------------------------------------------------ */
/* stage B: pending queue (display order) -> (B) -> display           */
/* ------------------------------------------------------------------ */

enum pend_ready { PEND_NONE, PEND_READY, PEND_WAIT };

static enum pend_ready pend_head_ready(struct dvfel_dev *d, ktime_t *deadline)
{
	struct dvfel_pend *p;
	struct dvfel_kjob *j;

	if (!d->pend_cnt)
		return PEND_NONE;
	p = &d->pendq[d->pend_head];
	if (p->job < 0 || p->gen != d->gen)
		return PEND_READY;
	j = &d->jobs[p->job];
	if (j->state == JOB_DONE || !d->client_ready)
		return PEND_READY;
	*deadline = ktime_add_ms(j->posted, gpu_timeout_ms);
	return ktime_after(ktime_get(), *deadline) ? PEND_READY : PEND_WAIT;
}

static void finish_job(struct dvfel_dev *d, struct dvfel_pend *pe)
{
	struct dvfel_kjob *j = &d->jobs[pe->job];
	struct dvfel_slot *s = j->slot;
	struct vframe_s *orig = j->orig;
	u32 w = orig->compWidth, h = orig->compHeight;
	bool composed, capture, ok;
	unsigned long flags;
	ulong src;
	ktime_t t0;
	s64 lat;

	spin_lock_irqsave(&d->lock, flags);
	composed = j->state == JOB_DONE && j->status == DVFEL_DONE_COMPOSED;
	if (j->state == JOB_DONE) {
		lat = ktime_us_delta(ktime_get(), j->posted);
		d->us_gpu_sum += lat;
		d->us_gpu_max = max(d->us_gpu_max, lat);
		d->us_gpu_cnt++;
		j->state = JOB_FINISH;
	} else {
		/* timed out or compositor gone: show the base layer */
		d->gpu_timeouts++;
		if (j->state == JOB_POSTED)
			j->state = JOB_FINISH;
		/* JOB_TAKEN stays: out[] may still be written, in[] is read only */
	}
	if (composed)
		d->gpu_composed++;
	else
		d->gpu_fallback++;
	src = composed ? d->cout[pe->job].phys : d->cin[pe->job].phys;
	spin_unlock_irqrestore(&d->lock, flags);

	capture = j->capture;
	j->capture = false;
	mutex_lock(&d->buf_lock);
	t0 = ktime_get();
	ok = vicp_ok(d, vicp_compress(&d->ctx_b, src, s, w, h),
		     ktime_us_delta(ktime_get(), t0), "B");
	if (ok) {
		stat_b(d, ktime_us_delta(ktime_get(), t0));
		build_out_vf(s, orig, w, h);
		if (capture)
			capture_frame(d, orig, src, s, w, h);
	}
	mutex_unlock(&d->buf_lock);

	spin_lock_irqsave(&d->lock, flags);
	if (pe->gen != d->gen) {
		s->state = SLOT_FREE;
	} else if (ok) {
		s->orig = orig;
		s->gen = pe->gen;
		s->state = SLOT_OUT;
		outq_push(d, &s->out_vf);
		d->frames_proc++;
	} else {
		/* cannot show it: give the decoder frame back */
		s->state = SLOT_FREE;
		spin_unlock_irqrestore(&d->lock, flags);
		vf_put(orig, DRV_NAME);
		spin_lock_irqsave(&d->lock, flags);
	}
	if (j->state == JOB_FINISH)
		j->state = JOB_DONE;	/* so job_release frees it */
	job_release(d, j);
	spin_unlock_irqrestore(&d->lock, flags);
}

static int dvfel_thread_b(void *data)
{
	struct dvfel_dev *d = data;

	while (!kthread_should_stop()) {
		struct dvfel_pend pe;
		unsigned long flags;
		enum pend_ready r;
		ktime_t deadline = 0;

		spin_lock_irqsave(&d->lock, flags);
		r = pend_head_ready(d, &deadline);
		spin_unlock_irqrestore(&d->lock, flags);

		if (r == PEND_NONE) {
			wait_event_interruptible(d->wq_b, d->kick_b || kthread_should_stop());
			d->kick_b = false;
			continue;
		}
		if (r == PEND_WAIT) {
			s64 left = ktime_us_delta(deadline, ktime_get());

			wait_event_interruptible_timeout(d->wq_b, d->kick_b || kthread_should_stop(),
							 usecs_to_jiffies(max_t(s64, left, 1000)));
			d->kick_b = false;
			continue;
		}

		spin_lock_irqsave(&d->lock, flags);
		pe = d->pendq[d->pend_head];
		pend_pop(d);
		if (pe.gen != d->gen) {
			/*
			 * stale entry from before a reset: the reset already
			 * freed SLOT_OUT slots of direct entries (which may be
			 * reused by now); a job still owns its BUSY slot
			 */
			if (pe.job >= 0) {
				struct dvfel_kjob *j = &d->jobs[pe.job];

				if (j->slot)
					j->slot->state = SLOT_FREE;
				if (j->state == JOB_POSTED)
					j->state = JOB_DONE;
				job_release(d, j);
			}
			spin_unlock_irqrestore(&d->lock, flags);
			d->kick_a = true;
			wake_up_interruptible(&d->wq_a);
			continue;
		}
		if (pe.job < 0) {
			outq_push(d, pe.vf);
			spin_unlock_irqrestore(&d->lock, flags);
		} else {
			spin_unlock_irqrestore(&d->lock, flags);
			finish_job(d, &pe);
		}
		vf_notify_receiver(DRV_NAME, VFRAME_EVENT_PROVIDER_VFRAME_READY, NULL);
		d->kick_a = true;
		wake_up_interruptible(&d->wq_a);
	}
	return 0;
}

static void dvfel_kick(struct dvfel_dev *d)
{
	d->kick_a = true;
	wake_up_interruptible(&d->wq_a);
}

/* ------------------------------------------------------------------ */
/* provider side (downstream calls us, often from vsync irq)          */
/* ------------------------------------------------------------------ */

static struct vframe_s *dvfel_peek(void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	struct vframe_s *vf = NULL;
	unsigned long flags;

	spin_lock_irqsave(&d->lock, flags);
	if (d->outq_cnt)
		vf = d->outq[d->outq_head];
	spin_unlock_irqrestore(&d->lock, flags);
	return vf;
}

static struct vframe_s *dvfel_get(void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	struct vframe_s *vf;
	unsigned long flags;

	spin_lock_irqsave(&d->lock, flags);
	vf = outq_pop(d);
	spin_unlock_irqrestore(&d->lock, flags);
	if (vf)
		dvfel_kick(d);
	return vf;
}

static void dvfel_put(struct vframe_s *vf, void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	struct vframe_s *upstream = NULL;
	struct dvfel_slot *s;
	unsigned long flags;
	int i;

	if (!vf)
		return;
	spin_lock_irqsave(&d->lock, flags);
	s = slot_of(d, vf);
	if (s) {
		if (s->state == SLOT_OUT && s->gen == d->gen)
			upstream = s->orig;
		if (s->state == SLOT_OUT)
			s->state = SLOT_FREE;
		s->orig = NULL;
	} else {
		for (i = 0; i < QUEUE_LEN; i++) {
			if (d->pass[i].vf == vf) {
				if (d->pass[i].gen == d->gen)
					upstream = vf;
				d->pass[i].vf = NULL;
				break;
			}
		}
	}
	spin_unlock_irqrestore(&d->lock, flags);

	if (upstream && d->prov_reg)
		vf_put(upstream, DRV_NAME);
	dvfel_kick(d);
}

/* events sent upstream by the display side: map our copy back */
static int dvfel_prov_event(int type, void *data, void *op_arg)
{
	struct dvfel_dev *d = op_arg;

	if (data && (type & (VFRAME_EVENT_RECEIVER_GET_AUX_DATA |
			     VFRAME_EVENT_RECEIVER_DISP_MODE |
			     VFRAME_EVENT_RECEIVER_REQ_STATE))) {
		/* all three request structs start with 'struct vframe_s *vf' */
		struct vframe_s **pvf = data, *mine = *pvf;
		struct dvfel_slot *s;
		unsigned long flags;

		spin_lock_irqsave(&d->lock, flags);
		s = mine ? slot_of(d, mine) : NULL;
		if (s && s->orig)
			*pvf = s->orig;
		spin_unlock_irqrestore(&d->lock, flags);
		vf_notify_provider(DRV_NAME, type, data);
		*pvf = mine;
		return 0;
	}
	vf_notify_provider(DRV_NAME, type, data);
	return 0;
}

static int dvfel_vf_states(struct vframe_states *states, void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	unsigned long flags;
	int i, nfree = 0;

	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->nslots; i++)
		if (d->slots[i].state == SLOT_FREE)
			nfree++;
	states->vf_pool_size = d->nslots;
	states->buf_free_num = nfree;
	states->buf_recycle_num = 0;
	states->buf_avail_num = d->outq_cnt;
	spin_unlock_irqrestore(&d->lock, flags);
	return 0;
}

static const struct vframe_operations_s dvfel_vf_ops = {
	.peek = dvfel_peek,
	.get = dvfel_get,
	.put = dvfel_put,
	.event_cb = dvfel_prov_event,
	.vf_states = dvfel_vf_states,
};

/* ------------------------------------------------------------------ */
/* receiver side (upstream decoder events)                            */
/* ------------------------------------------------------------------ */

static int dvfel_recv_event(int type, void *data, void *op_arg)
{
	struct dvfel_dev *d = op_arg;
	unsigned long flags;

	switch (type) {
	case VFRAME_EVENT_PROVIDER_REG:
		dvfel_dbg("upstream REG %s\n", data ? (char *)data : "");
		cancel_delayed_work_sync(&d->free_work);
		spin_lock_irqsave(&d->lock, flags);
		dvfel_reset_locked(d);
		d->prov_reg = true;
		spin_unlock_irqrestore(&d->lock, flags);
		vf_reg_provider(&d->prov);
		dvfel_kick(d);
		break;
	case VFRAME_EVENT_PROVIDER_UNREG:
		dvfel_dbg("upstream UNREG\n");
		spin_lock_irqsave(&d->lock, flags);
		d->prov_reg = false;
		dvfel_reset_locked(d);
		spin_unlock_irqrestore(&d->lock, flags);
		vf_unreg_provider(&d->prov);
		d->kick_b = true;
		wake_up_interruptible(&d->wq_b);
		/* the display may still scan out our last buffer for a while */
		schedule_delayed_work(&d->free_work, msecs_to_jiffies(3000));
		break;
	case VFRAME_EVENT_PROVIDER_LIGHT_UNREG:
		spin_lock_irqsave(&d->lock, flags);
		dvfel_reset_locked(d);
		spin_unlock_irqrestore(&d->lock, flags);
		vf_light_unreg_provider(&d->prov);
		d->kick_b = true;
		wake_up_interruptible(&d->wq_b);
		break;
	case VFRAME_EVENT_PROVIDER_RESET:
		spin_lock_irqsave(&d->lock, flags);
		dvfel_reset_locked(d);
		spin_unlock_irqrestore(&d->lock, flags);
		vf_notify_receiver(DRV_NAME, type, data);
		d->kick_b = true;
		wake_up_interruptible(&d->wq_b);
		break;
	case VFRAME_EVENT_PROVIDER_VFRAME_READY:
		dvfel_kick(d);
		break;
	case VFRAME_EVENT_PROVIDER_QUREY_STATE:
		return RECEIVER_ACTIVE;
	default:
		/* START, FR_HINT, FR_END_HINT, ... */
		return vf_notify_receiver(DRV_NAME, type, data);
	}
	return 0;
}

static const struct vframe_receiver_op_s dvfel_recv_ops = {
	.event_cb = dvfel_recv_event,
};

/* ------------------------------------------------------------------ */
/* /dev/dvfel (compositor)                                            */
/* ------------------------------------------------------------------ */

static void cbuf_put(struct dvfel_cbuf *b)
{
	if (b->sgt)
		dma_buf_unmap_attachment(b->att, b->sgt, DMA_BIDIRECTIONAL);
	if (b->att)
		dma_buf_detach(b->db, b->att);
	if (b->db)
		dma_buf_put(b->db);
	memset(b, 0, sizeof(*b));
}

/* import a physically contiguous dma-buf of at least @size bytes */
static int cbuf_get(struct dvfel_dev *d, struct dvfel_cbuf *b, int fd, size_t size)
{
	struct scatterlist *sg;
	phys_addr_t next;
	size_t len = 0;
	int i;

	b->db = dma_buf_get(fd);
	if (IS_ERR(b->db)) {
		b->db = NULL;
		return -EBADF;
	}
	b->att = dma_buf_attach(b->db, d->misc.this_device);
	if (IS_ERR(b->att)) {
		b->att = NULL;
		goto fail;
	}
	b->sgt = dma_buf_map_attachment(b->att, DMA_BIDIRECTIONAL);
	if (IS_ERR(b->sgt)) {
		b->sgt = NULL;
		goto fail;
	}
	b->phys = page_to_phys(sg_page(b->sgt->sgl)) + b->sgt->sgl->offset;
	next = b->phys;
	for_each_sgtable_sg(b->sgt, sg, i) {
		if (page_to_phys(sg_page(sg)) + sg->offset != next)
			goto fail;	/* not contiguous */
		next += sg->length;
		len += sg->length;
	}
	if (len < size)
		goto fail;
	return 0;
fail:
	cbuf_put(b);
	return -EINVAL;
}

static bool client_jobs_busy(struct dvfel_dev *d)
{
	unsigned long flags;
	bool busy = false;
	int i;

	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->client_nbufs; i++)
		if (d->jobs[i].state != JOB_FREE)
			busy = true;
	spin_unlock_irqrestore(&d->lock, flags);
	return busy;
}

/* d->client_lock held */
static void client_unreg(struct dvfel_dev *d)
{
	unsigned long flags;
	int i;

	spin_lock_irqsave(&d->lock, flags);
	d->client_ready = false;
	for (i = 0; i < d->client_nbufs; i++)
		if (d->jobs[i].state == JOB_TAKEN) {
			/* nobody will report it any more */
			d->jobs[i].state = d->jobs[i].orphan ? JOB_FREE : JOB_DONE;
			d->jobs[i].status = DVFEL_DONE_PASSTHROUGH;
			d->jobs[i].orphan = false;
		}
	spin_unlock_irqrestore(&d->lock, flags);
	d->kick_b = true;
	wake_up_interruptible(&d->wq_b);
	wake_up_interruptible(&d->wq_client);

	/* stage B still reads in[] of pending jobs: wait until they are gone */
	for (i = 0; i < 100 && client_jobs_busy(d); i++)
		msleep(10);
	if (client_jobs_busy(d))
		pr_warn(DRV_NAME ": compositor buffers still busy at unregister\n");

	for (i = 0; i < d->client_nbufs; i++) {
		cbuf_put(&d->cin[i]);
		cbuf_put(&d->cout[i]);
		d->jobs[i].state = JOB_FREE;
	}
	d->client_nbufs = 0;
}

static long ioc_reg_bufs(struct dvfel_dev *d, struct file *f, void __user *arg)
{
	struct dvfel_reg_bufs r;
	size_t size;
	int i, ret;

	if (copy_from_user(&r, arg, sizeof(r)))
		return -EFAULT;
	if (!r.count || r.count > DVFEL_MAX_BUFS || !r.width || !r.height ||
	    r.width > MAX_W || r.height > MAX_H || !wmif_bg_width(r.width))
		return -EINVAL;
	size = LIN_SIZE(r.width, r.height);

	mutex_lock(&d->client_lock);
	if (d->client && d->client != f) {
		mutex_unlock(&d->client_lock);
		return -EBUSY;
	}
	if (d->client_nbufs)
		client_unreg(d);
	for (i = 0; i < r.count; i++) {
		ret = cbuf_get(d, &d->cin[i], r.in_fd[i], size);
		if (!ret)
			ret = cbuf_get(d, &d->cout[i], r.out_fd[i], size);
		if (ret) {
			d->client_nbufs = i + 1;
			client_unreg(d);
			mutex_unlock(&d->client_lock);
			pr_err(DRV_NAME ": buffer %d rejected (need %zu contiguous bytes)\n",
			       i, size);
			return ret;
		}
	}
	d->client = f;
	d->client_w = r.width;
	d->client_h = r.height;
	d->client_nbufs = r.count;
	memset(d->jobs, 0, sizeof(d->jobs));
	d->client_ready = true;
	d->new_stream = true;
	mutex_unlock(&d->client_lock);
	pr_info(DRV_NAME ": compositor registered %u buffer pairs %ux%u\n",
		r.count, r.width, r.height);
	return 0;
}

static bool job_posted(struct dvfel_dev *d, int *idx)
{
	unsigned long flags;
	u32 best = 0;
	int i;

	*idx = -1;
	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->client_nbufs; i++) {
		struct dvfel_kjob *j = &d->jobs[i];

		if (j->state == JOB_POSTED && (*idx < 0 || (s32)(j->id - best) < 0)) {
			*idx = i;
			best = j->id;
		}
	}
	spin_unlock_irqrestore(&d->lock, flags);
	return *idx >= 0 || !d->client_ready;
}

static long ioc_wait_job(struct dvfel_dev *d, struct file *f, void __user *arg)
{
	struct dvfel_job u;
	unsigned long flags;
	long ret;
	int i;

	if (copy_from_user(&u, arg, sizeof(u)))
		return -EFAULT;
	if (d->client != f || !d->client_ready)
		return -ENODEV;
	ret = wait_event_interruptible_timeout(d->wq_client, job_posted(d, &i),
					       msecs_to_jiffies(u.timeout_ms));
	if (ret < 0)
		return ret;
	spin_lock_irqsave(&d->lock, flags);
	if (i < 0 || d->jobs[i].state != JOB_POSTED || !d->client_ready) {
		spin_unlock_irqrestore(&d->lock, flags);
		return d->client_ready ? -ETIMEDOUT : -ENODEV;
	}
	d->jobs[i].state = JOB_TAKEN;
	u.id = d->jobs[i].id;
	u.buf = i;
	u.width = d->client_w;
	u.height = d->client_h;
	u.flags = d->jobs[i].flags;
	u.pts_us = d->jobs[i].pts_us;
	spin_unlock_irqrestore(&d->lock, flags);
	return copy_to_user(arg, &u, sizeof(u)) ? -EFAULT : 0;
}

static long ioc_job_done(struct dvfel_dev *d, struct file *f, void __user *arg)
{
	struct dvfel_job_done u;
	unsigned long flags;
	int i, ret = -ENOENT;

	if (copy_from_user(&u, arg, sizeof(u)))
		return -EFAULT;
	if (d->client != f)
		return -ENODEV;
	spin_lock_irqsave(&d->lock, flags);
	for (i = 0; i < d->client_nbufs; i++) {
		struct dvfel_kjob *j = &d->jobs[i];

		if (j->id != u.id || j->state != JOB_TAKEN)
			continue;
		if (j->orphan) {
			j->state = JOB_FREE;
			j->orphan = false;
		} else {
			j->state = JOB_DONE;
			j->status = u.status;
		}
		ret = 0;
		break;
	}
	spin_unlock_irqrestore(&d->lock, flags);
	d->kick_b = true;
	wake_up_interruptible(&d->wq_b);
	dvfel_kick(d);
	return ret;
}

static long dvfel_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct dvfel_dev *d = gdev;
	void __user *p = (void __user *)arg;

	switch (cmd) {
	case DVFEL_IOC_REG_BUFS:
		return ioc_reg_bufs(d, f, p);
	case DVFEL_IOC_WAIT_JOB:
		return ioc_wait_job(d, f, p);
	case DVFEL_IOC_JOB_DONE:
		return ioc_job_done(d, f, p);
	case DVFEL_IOC_UNREG_BUFS:
		mutex_lock(&d->client_lock);
		if (d->client == f) {
			client_unreg(d);
			d->client = NULL;
		}
		mutex_unlock(&d->client_lock);
		return 0;
	}
	return -ENOTTY;
}

static int dvfel_release(struct inode *inode, struct file *f)
{
	struct dvfel_dev *d = gdev;

	mutex_lock(&d->client_lock);
	if (d->client == f) {
		client_unreg(d);
		d->client = NULL;
		pr_info(DRV_NAME ": compositor disconnected\n");
	}
	mutex_unlock(&d->client_lock);
	return 0;
}

static const struct file_operations dvfel_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = dvfel_ioctl,
	.compat_ioctl = dvfel_ioctl,
	.release = dvfel_release,
};

/* ------------------------------------------------------------------ */
/* debugfs                                                            */
/* ------------------------------------------------------------------ */

static int stats_show(struct seq_file *m, void *v)
{
	struct dvfel_dev *d = m->private;
	s64 na = d->us_cnt ? d->us_cnt : 1;
	s64 nb = d->us_b_cnt ? d->us_b_cnt : 1;
	s64 ng = d->us_gpu_cnt ? d->us_gpu_cnt : 1;

	seq_printf(m, "mode %d, slots %d, provider %s, buffers %s (%ux%u), compositor %s\n",
		   mode, d->nslots, d->prov_reg ? "registered" : "idle",
		   d->bufs_ok ? "allocated" : "free", d->buf_w, d->buf_h,
		   d->client_ready ? "connected" : "none");
	seq_printf(m, "frames in %llu, processed %llu, passthrough %llu, vicp errors %llu, slot waits %llu\n",
		   d->frames_in, d->frames_proc, d->frames_pass, d->vicp_err,
		   d->slot_waits);
	seq_printf(m, "gpu jobs %llu, composed %llu, fallback %llu, timeouts %llu\n",
		   d->gpu_jobs, d->gpu_composed, d->gpu_fallback, d->gpu_timeouts);
	seq_printf(m, "VICP A (decompress)  avg %lld us max %lld us\n",
		   d->us_a_sum / na, d->us_a_max);
	seq_printf(m, "VICP B (compress)    avg %lld us max %lld us\n",
		   d->us_b_sum / nb, d->us_b_max);
	seq_printf(m, "GPU job (post->done) avg %lld us max %lld us\n",
		   d->us_gpu_sum / ng, d->us_gpu_max);
	seq_printf(m, "queued %d, pending %d\n", d->outq_cnt, d->pend_cnt);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(stats);

static ssize_t stats_reset_write(struct file *f, const char __user *buf,
				 size_t len, loff_t *ppos)
{
	struct dvfel_dev *d = f->private_data;

	d->frames_in = d->frames_proc = d->frames_pass = d->vicp_err = 0;
	d->slot_waits = 0;
	d->gpu_jobs = d->gpu_composed = d->gpu_fallback = d->gpu_timeouts = 0;
	d->us_a_sum = d->us_b_sum = d->us_a_max = d->us_b_max = 0;
	d->us_gpu_sum = d->us_gpu_max = 0;
	d->us_cnt = d->us_b_cnt = d->us_gpu_cnt = 0;
	return len;
}

static const struct file_operations stats_reset_fops = {
	.open = simple_open,
	.write = stats_reset_write,
};

/* write anything: capture the next processed frame */
static ssize_t capture_write(struct file *f, const char __user *buf,
			     size_t len, loff_t *ppos)
{
	struct dvfel_dev *d = f->private_data;

	atomic_set(&d->capture_req, 1);
	return len;
}

/*
 * read: 64-byte text header "DVFEL444P10 w h stride pts_us nframes",
 * then nframes linear 10-bit 4:4:4 frames
 */
static ssize_t capture_read(struct file *f, char __user *buf,
			    size_t len, loff_t *ppos)
{
	struct dvfel_dev *d = f->private_data;
	size_t frame = (size_t)LIN_STRIDE(d->cap_w) * d->cap_h;
	char hdr[64];
	loff_t pos;
	ssize_t r;
	int hl;

	if (!d->cap_frames)
		return 0;
	if (*ppos < 64) {
		hl = scnprintf(hdr, sizeof(hdr), "DVFEL444P10 %u %u %u %llu %d",
			       d->cap_w, d->cap_h, LIN_STRIDE(d->cap_w),
			       d->cap_pts_us, d->cap_frames);
		memset(hdr + hl, ' ', sizeof(hdr) - hl);
		hdr[63] = '\n';
		return simple_read_from_buffer(buf, len, ppos, hdr, 64);
	}
	pos = *ppos - 64;
	r = simple_read_from_buffer(buf, len, &pos, d->cap_buf, frame * d->cap_frames);
	if (r > 0)
		*ppos += r;
	return r;
}

static const struct file_operations capture_fops = {
	.open = simple_open,
	.read = capture_read,
	.write = capture_write,
	.llseek = default_llseek,
};

/* ------------------------------------------------------------------ */

static int __init dvfel_init(void)
{
	struct dvfel_dev *d;
	int ret;

	d = kzalloc(sizeof(*d), GFP_KERNEL);
	if (!d)
		return -ENOMEM;
	d->nslots = clamp(slots, 2, MAX_SLOTS);
	spin_lock_init(&d->lock);
	mutex_init(&d->buf_lock);
	mutex_init(&d->client_lock);
	init_waitqueue_head(&d->wq_a);
	init_waitqueue_head(&d->wq_b);
	init_waitqueue_head(&d->wq_client);
	INIT_DELAYED_WORK(&d->free_work, dvfel_free_work);
	atomic_set(&d->capture_req, 0);
	gdev = d;

	d->misc.minor = MISC_DYNAMIC_MINOR;
	d->misc.name = DRV_NAME;
	d->misc.fops = &dvfel_fops;
	d->misc.mode = 0600;
	ret = misc_register(&d->misc);
	if (ret)
		goto err_free;
	dma_coerce_mask_and_coherent(d->misc.this_device, DMA_BIT_MASK(32));

	vf_receiver_init(&d->recv, DRV_NAME, &dvfel_recv_ops, d);
	vf_provider_init(&d->prov, DRV_NAME, &dvfel_vf_ops, d);

	d->thread_a = kthread_run(dvfel_thread_a, d, DRV_NAME "_a");
	if (IS_ERR(d->thread_a)) {
		ret = PTR_ERR(d->thread_a);
		goto err_misc;
	}
	d->thread_b = kthread_run(dvfel_thread_b, d, DRV_NAME "_b");
	if (IS_ERR(d->thread_b)) {
		ret = PTR_ERR(d->thread_b);
		kthread_stop(d->thread_a);
		goto err_misc;
	}
	vf_reg_receiver(&d->recv);

	d->dbg = debugfs_create_dir(DRV_NAME, NULL);
	debugfs_create_file("stats", 0444, d->dbg, d, &stats_fops);
	debugfs_create_file("stats_reset", 0200, d->dbg, d, &stats_reset_fops);
	debugfs_create_file("capture", 0600, d->dbg, d, &capture_fops);

	pr_info(DRV_NAME ": loaded, mode %d, %d slots\n", mode, d->nslots);
	return 0;

err_misc:
	misc_deregister(&d->misc);
err_free:
	kfree(d);
	gdev = NULL;
	return ret;
}

static void __exit dvfel_exit(void)
{
	struct dvfel_dev *d = gdev;

	debugfs_remove_recursive(d->dbg);
	vf_unreg_receiver(&d->recv);
	if (d->prov_reg)
		vf_unreg_provider(&d->prov);
	misc_deregister(&d->misc);
	mutex_lock(&d->client_lock);
	if (d->client_nbufs)
		client_unreg(d);
	mutex_unlock(&d->client_lock);
	kthread_stop(d->thread_b);
	kthread_stop(d->thread_a);
	cancel_delayed_work_sync(&d->free_work);
	mutex_lock(&d->buf_lock);
	dvfel_free_bufs(d);
	mutex_unlock(&d->buf_lock);
	vfree(d->cap_buf);
	kfree(d);
}

module_init(dvfel_init);
module_exit(dvfel_exit);

MODULE_DESCRIPTION("Dolby Vision FEL composition vfm node for Amlogic S5");
MODULE_LICENSE("GPL");
MODULE_IMPORT_NS(DMA_BUF);
