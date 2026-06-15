/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * LD_PRELOAD shim that fakes per-island machine identity so that UCX classifies
 * cross-island endpoints (which physically live on the same host) as
 * inter-node. UCX picks cuda_ipc for intra-node device lanes and only becomes
 * eligible for rc_gda when it believes the peer is on a different machine.
 *
 * UCX's intra-node detection combines several sources, so we override all the
 * ones we know about:
 *   - gethostname() / uname()
 *   - gethostid() (libc-derived 32-bit id)
 *   - reads of /etc/machine-id, /var/lib/dbus/machine-id,
 *     /proc/sys/kernel/random/boot_id (via open/openat/fopen redirection)
 *
 * For each of the above, we return a value that is deterministic for a given
 * UCX_FAKE_HOSTNAME value but distinct across different values. Two processes
 * in the same NVL island share UCX_FAKE_HOSTNAME ("ep-island-0") and so see
 * identical machine identity (UCX picks cuda_ipc, preserving NVL).
 * Processes in different islands see different identity (UCX is free to pick
 * rc_gda for the device lane, enabling kernel-issued RDMA on the NIC).
 *
 * Build: gcc -O2 -shared -fPIC -ldl island_hostname.c -o island_hostname.so
 * Use:   LD_PRELOAD=/path/island_hostname.so UCX_FAKE_HOSTNAME=ep-island-0 ./prog
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

/* ---- Helpers ---- */

static const char *get_suffix(void) {
    const char *h = getenv("UCX_FAKE_HOSTNAME");
    return (h && *h) ? h : NULL;
}

static unsigned long djb2(const char *s) {
    unsigned long h = 5381;
    for (; *s; ++s) h = ((h << 5) + h) + (unsigned char)*s;
    return h;
}

static void hash4(const char *suffix, unsigned long h[4]) {
    unsigned long base = djb2(suffix);
    h[0] = base;
    h[1] = base * 2654435761UL + 0xdeadbeef;
    h[2] = base * 0x9e3779b9UL + 0xcafebabe;
    h[3] = base * 0x85ebca6bUL + 0x12345678;
}

/* /etc/machine-id format: 32 lowercase hex chars + newline. */
static void make_machine_id(const char *suffix, char *out, size_t outsize) {
    unsigned long h[4]; hash4(suffix, h);
    snprintf(out, outsize, "%08lx%08lx%08lx%08lx\n",
             h[0] & 0xffffffff, h[1] & 0xffffffff,
             h[2] & 0xffffffff, h[3] & 0xffffffff);
}

/* boot_id format: UUID with dashes + newline. */
static void make_boot_id(const char *suffix, char *out, size_t outsize) {
    unsigned long h[4]; hash4(suffix, h);
    snprintf(out, outsize, "%08lx-%04lx-%04lx-%04lx-%012lx\n",
             h[0] & 0xffffffff,
             (h[1] >> 16) & 0xffff, h[1] & 0xffff,
             (h[2] >> 16) & 0xffff,
             h[3] & 0xffffffffffffUL);
}

/* Cached fake file paths (one of each per process). */
static char fake_machine_id_path[128] = {0};
static char fake_boot_id_path[128] = {0};
static int  fake_files_ready = 0;

static int (*real_open)(const char *, int, ...) = NULL;
static int (*real_openat)(int, const char *, int, ...) = NULL;
static FILE *(*real_fopen)(const char *, const char *) = NULL;
static FILE *(*real_fopen64)(const char *, const char *) = NULL;
static int (*real_gethostname)(char *, size_t) = NULL;
static int (*real_uname)(struct utsname *) = NULL;
static long (*real_gethostid)(void) = NULL;

static void resolve_reals(void) {
    if (!real_open)        real_open        = dlsym(RTLD_NEXT, "open");
    if (!real_openat)      real_openat      = dlsym(RTLD_NEXT, "openat");
    if (!real_fopen)       real_fopen       = dlsym(RTLD_NEXT, "fopen");
    if (!real_fopen64)     real_fopen64     = dlsym(RTLD_NEXT, "fopen64");
    if (!real_gethostname) real_gethostname = dlsym(RTLD_NEXT, "gethostname");
    if (!real_uname)       real_uname       = dlsym(RTLD_NEXT, "uname");
    if (!real_gethostid)   real_gethostid   = dlsym(RTLD_NEXT, "gethostid");
}

static void ensure_fake_files(void) {
    if (fake_files_ready) return;
    resolve_reals();
    const char *suffix = get_suffix();
    if (!suffix) { fake_files_ready = 1; return; }

    snprintf(fake_machine_id_path, sizeof(fake_machine_id_path),
             "/tmp/.island_machine_id.%s.%d", suffix, (int)getpid());
    snprintf(fake_boot_id_path, sizeof(fake_boot_id_path),
             "/tmp/.island_boot_id.%s.%d", suffix, (int)getpid());

    int fd = real_open(fake_machine_id_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        char buf[64];
        make_machine_id(suffix, buf, sizeof(buf));
        ssize_t _ = write(fd, buf, strlen(buf));
        (void)_;
        close(fd);
    }
    fd = real_open(fake_boot_id_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        char buf[64];
        make_boot_id(suffix, buf, sizeof(buf));
        ssize_t _ = write(fd, buf, strlen(buf));
        (void)_;
        close(fd);
    }
    fake_files_ready = 1;
}

static const char *maybe_redirect(const char *path) {
    if (!path || !get_suffix()) return NULL;
    ensure_fake_files();
    if (strcmp(path, "/etc/machine-id") == 0 ||
        strcmp(path, "/var/lib/dbus/machine-id") == 0)
        return fake_machine_id_path;
    if (strcmp(path, "/proc/sys/kernel/random/boot_id") == 0)
        return fake_boot_id_path;
    return NULL;
}

/* ---- gethostname / uname / gethostid ---- */

int gethostname(char *name, size_t len) {
    const char *fake = get_suffix();
    if (fake) {
        size_t n = strlen(fake);
        if (n >= len) n = len - 1;
        memcpy(name, fake, n);
        name[n] = '\0';
        return 0;
    }
    resolve_reals();
    return real_gethostname ? real_gethostname(name, len) : -1;
}

int uname(struct utsname *buf) {
    resolve_reals();
    int rc = real_uname ? real_uname(buf) : -1;
    if (rc != 0) return rc;
    const char *fake = get_suffix();
    if (fake) {
        size_t n = strlen(fake);
        if (n >= sizeof(buf->nodename)) n = sizeof(buf->nodename) - 1;
        memcpy(buf->nodename, fake, n);
        buf->nodename[n] = '\0';
    }
    return 0;
}

long gethostid(void) {
    const char *fake = get_suffix();
    if (fake) {
        unsigned long h = djb2(fake);
        return (long)(h & 0x7fffffff);
    }
    resolve_reals();
    return real_gethostid ? real_gethostid() : 0;
}

/* ---- File redirection ---- */

int open(const char *pathname, int flags, ...) {
    resolve_reals();
    const char *r = maybe_redirect(pathname);
    if (r) pathname = r;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap; va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t);
        va_end(ap);
        return real_open(pathname, flags, mode);
    }
    return real_open(pathname, flags);
}

int openat(int dirfd, const char *pathname, int flags, ...) {
    resolve_reals();
    const char *r = maybe_redirect(pathname);
    if (r) { dirfd = AT_FDCWD; pathname = r; }
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap; va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t);
        va_end(ap);
        return real_openat(dirfd, pathname, flags, mode);
    }
    return real_openat(dirfd, pathname, flags);
}

FILE *fopen(const char *path, const char *mode) {
    resolve_reals();
    const char *r = maybe_redirect(path);
    if (r) path = r;
    return real_fopen ? real_fopen(path, mode) : NULL;
}

FILE *fopen64(const char *path, const char *mode) {
    resolve_reals();
    const char *r = maybe_redirect(path);
    if (r) path = r;
    return real_fopen64 ? real_fopen64(path, mode) : fopen(path, mode);
}
