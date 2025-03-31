#ifndef TIME_COMMON_H
#define TIME_COMMON_H

#include <stdio.h>
#include <time.h>

static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}

static inline void print_time(const char* label, double start, double end) {
    printf("%s: %.6f seconds\n", label, end - start);
}

#endif /* TIME_COMMON_H */ 