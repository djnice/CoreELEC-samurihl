/* SPDX-License-Identifier: GPL-2.0-or-later WITH Linux-syscall-note */
/*
 * dvfel userspace interface (/dev/dvfel)
 *
 * A compositor process (Kodi) registers pairs of linear frame buffers
 * allocated from /dev/dma_heap/heap-codecmm. For every decoded frame dvfel
 * decompresses the base layer into in[buf] and posts a job; the compositor
 * writes the composed frame into out[buf] and reports completion. dvfel then
 * compresses out[buf] for display. Jobs are posted in display order.
 *
 * Buffer layout (both directions): width x height pixels, one little-endian
 * 32-bit word per pixel, stride = width * 4 bytes:
 *   bits  0- 9  Cr (10-bit)
 *   bits 10-19  Cb (10-bit)
 *   bits 20-29  Y  (10-bit)
 *   bits 30-31  0
 * In in[buf], chroma at even x and even y is the original 4:2:0 sample; odd
 * positions are interpolated. dvfel converts out[buf] back to 4:2:0 by
 * keeping the chroma at even x/y.
 *
 * Hardware decoded enhancement layer (Dolby Vision dual layer decoding, the
 * EL decoder feeding the "dvfelel" receiver): with el_width set, dvfel
 * decompresses the EL picture of each frame into el[buf] in the same layout
 * (el_width x el_height) and flags the job with DVFEL_JOB_EL.
 */
#ifndef _UAPI_DVFEL_H
#define _UAPI_DVFEL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define DVFEL_MAX_BUFS		4
/* bytes needed per buffer (includes a write-MIF margin) */
#define DVFEL_BUF_SIZE(w, h)	((((w) * 4 * (h) + 256 * 1024) + 4095) & ~4095)

struct dvfel_reg_bufs {
	__u32 width;
	__u32 height;
	__u32 count;			/* 1..DVFEL_MAX_BUFS */
	__s32 in_fd[DVFEL_MAX_BUFS];	/* dma-buf fds */
	__s32 out_fd[DVFEL_MAX_BUFS];
	__u32 el_width;			/* 0: no hardware decoded EL */
	__u32 el_height;
	__s32 el_fd[DVFEL_MAX_BUFS];
};

#define DVFEL_JOB_NEW_STREAM	(1 << 0)	/* first job after start/seek */
#define DVFEL_JOB_EL		(1 << 1)	/* el[buf] holds the EL picture */

struct dvfel_job {
	__u32 timeout_ms;		/* in: max wait */
	__u32 id;			/* out */
	__u32 buf;			/* out: buffer pair index */
	__u32 width;			/* out */
	__u32 height;			/* out */
	__u32 flags;			/* out: DVFEL_JOB_* */
	__u64 pts_us;			/* out: vframe pts_us64 (Kodi packet pts) */
	/*
	 * out: the access units (decode order, 16 bits, wrapping) of the BL
	 * frame and of its EL picture. They differ when the EL is coded in
	 * another picture order than the BL (paired by display order): the
	 * RPU is then the one of the EL picture's access unit.
	 */
	__u32 bl_au;
	__u32 el_au;
};

#define DVFEL_DONE_COMPOSED	0	/* out[buf] holds the frame */
#define DVFEL_DONE_PASSTHROUGH	1	/* show in[buf] unchanged */

struct dvfel_job_done {
	__u32 id;
	__s32 status;			/* DVFEL_DONE_* */
};

#define DVFEL_IOC_MAGIC		'F'
#define DVFEL_IOC_REG_BUFS	_IOW(DVFEL_IOC_MAGIC, 1, struct dvfel_reg_bufs)
#define DVFEL_IOC_WAIT_JOB	_IOWR(DVFEL_IOC_MAGIC, 2, struct dvfel_job)
#define DVFEL_IOC_JOB_DONE	_IOW(DVFEL_IOC_MAGIC, 3, struct dvfel_job_done)
#define DVFEL_IOC_UNREG_BUFS	_IO(DVFEL_IOC_MAGIC, 4)

#endif
