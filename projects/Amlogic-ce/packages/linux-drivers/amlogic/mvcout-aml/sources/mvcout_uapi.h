/* SPDX-License-Identifier: GPL-2.0-or-later WITH Linux-syscall-note */
/*
 * mvcout - show frames written by user space on the video layer
 *
 * Usage: MVCOUT_IOC_ALLOC, mmap each buffer (offset = index * map_size),
 * MVCOUT_IOC_START (registers the "mvcout" vframe provider; the vfm map
 * that connects it to a receiver is set up by user space), then per frame
 * MVCOUT_IOC_DEQUEUE -> write -> MVCOUT_IOC_QUEUE. The receiver shows a
 * queued frame as soon as it takes it (freerun), so user space paces.
 */
#ifndef MVCOUT_UAPI_H
#define MVCOUT_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define MVCOUT_DEV		"/dev/mvcout"
#define MVCOUT_PROVIDER		"mvcout"
#define MVCOUT_MAX_BUFS		16

#define MVCOUT_FMT_NV21		0	/* Y plane, then interleaved CrCb, 8 bit */
#define MVCOUT_FMT_YUV422	1	/* packed 4:2:2, 2 bytes per pixel, 8 bit */

struct mvcout_alloc {
	__u32 width;		/* in */
	__u32 height;		/* in */
	__u32 format;		/* in: MVCOUT_FMT_* */
	__u32 count;		/* in: 2..MVCOUT_MAX_BUFS */
	__u32 stride;		/* out: bytes per line (Y plane for NV21) */
	__u32 size;		/* out: bytes used per buffer */
	__u32 map_size;		/* out: mmap offset step between buffers */
	__u32 reserved;
};

struct mvcout_frame {
	__s32 index;		/* DEQUEUE out, QUEUE in */
	__u32 timeout_ms;	/* DEQUEUE in: 0 = do not wait */
	__u64 pts_us;		/* QUEUE in */
	__u32 duration;		/* QUEUE in: 1/96000 s units (4004 = 23.976 Hz) */
	__u32 flags;		/* reserved, 0 */
	__u32 width;		/* QUEUE in: picture size within the buffer, */
	__u32 height;		/* 0 = the allocated size; same stride */
};

struct mvcout_stats {
	__u64 queued;
	__u64 taken;
	__u64 returned;
	__u32 held;		/* buffers kept by the display after a stop */
	__u32 reserved;
};

#define MVCOUT_IOC_MAGIC	'M'
#define MVCOUT_IOC_ALLOC	_IOWR(MVCOUT_IOC_MAGIC, 1, struct mvcout_alloc)
#define MVCOUT_IOC_START	_IO(MVCOUT_IOC_MAGIC, 2)
#define MVCOUT_IOC_STOP		_IO(MVCOUT_IOC_MAGIC, 3)
#define MVCOUT_IOC_DEQUEUE	_IOWR(MVCOUT_IOC_MAGIC, 4, struct mvcout_frame)
#define MVCOUT_IOC_QUEUE	_IOW(MVCOUT_IOC_MAGIC, 5, struct mvcout_frame)
#define MVCOUT_IOC_FLUSH	_IO(MVCOUT_IOC_MAGIC, 6)
#define MVCOUT_IOC_STATS	_IOR(MVCOUT_IOC_MAGIC, 7, struct mvcout_stats)

#endif
