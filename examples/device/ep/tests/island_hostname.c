/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * LD_PRELOAD shim that overrides gethostname()/uname() to return whatever is
 * in the UCX_FAKE_HOSTNAME env var. Used by run_test_ht_2x2.sh to trick UCX
 * into treating cross-island endpoints on a single physical host as inter-node,
 * so it considers rc_gda (kernel-issued RDMA) for the device lane instead of
 * defaulting to cuda_ipc for everything.
 *
 * Within an NVL island (same fake hostname) UCX still picks cuda_ipc/NVL.
 * Across islands (different fake hostnames) UCX is eligible to pick rc_gda.
 *
 * Build: gcc -O2 -shared -fPIC -ldl island_hostname.c -o island_hostname.so
 * Use:   LD_PRELOAD=/path/to/island_hostname.so UCX_FAKE_HOSTNAME=island-0 ./prog
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

static int (*real_gethostname)(char *, size_t) = NULL;
static int (*real_uname)(struct utsname *) = NULL;

static void resolve_real(void) {
    if (!real_gethostname)
        real_gethostname = dlsym(RTLD_NEXT, "gethostname");
    if (!real_uname)
        real_uname = dlsym(RTLD_NEXT, "uname");
}

int gethostname(char *name, size_t len) {
    const char *fake = getenv("UCX_FAKE_HOSTNAME");
    if (fake && *fake) {
        size_t n = strlen(fake);
        if (n >= len) n = len - 1;
        memcpy(name, fake, n);
        name[n] = '\0';
        return 0;
    }
    resolve_real();
    return real_gethostname ? real_gethostname(name, len) : -1;
}

int uname(struct utsname *buf) {
    resolve_real();
    int rc = real_uname ? real_uname(buf) : -1;
    if (rc != 0) return rc;
    const char *fake = getenv("UCX_FAKE_HOSTNAME");
    if (fake && *fake) {
        size_t n = strlen(fake);
        if (n >= sizeof(buf->nodename)) n = sizeof(buf->nodename) - 1;
        memcpy(buf->nodename, fake, n);
        buf->nodename[n] = '\0';
    }
    return 0;
}
