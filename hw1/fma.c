#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#ifndef LEN
#define LEN 1000
#endif

#ifndef ITER
#define ITER 1000000
#endif

#ifdef USINGD
typedef double float_t;
#else
typedef float float_t;
#endif

#ifndef PAR
#define PAR 16
#endif

float_t a[LEN];
float_t b[LEN];
float_t r[ITER] = {};

// a, b are vectors of length len
inline float_t dot(float_t a[], float_t b[]) {
    int i, j;
    float_t result[PAR] = {};
    for (i = 0; i < LEN/PAR; i++) {
	for (j = 0; j < PAR; j++)
            result[j] += a[i*PAR+j]*b[i*PAR+j];
    }
    float_t res;
    for (j = 0; j < PAR; j++)
        res += result[j];
    return res;
}

typedef struct timespec* tsp;
void performance(tsp start, tsp end, tsp diff)
{
    long int sec, nsec;
    if ((end->tv_nsec - start->tv_nsec)<0) {
        sec = end->tv_sec - start->tv_sec - 1;
        nsec = 1000000000 + end->tv_nsec - start->tv_nsec;
    } else {
        sec = end->tv_sec - start->tv_sec;
        nsec = end->tv_nsec - start->tv_nsec;
    }
    diff->tv_sec = sec;
    diff->tv_nsec = nsec;
}

int main() {
    long len = LEN, iter = ITER;
    long i;
    for (i = 0; i < len; i++) {
        a[i] = (i+1)/(float_t) (7);
        b[i] = (i-1)/(float_t) (7);
    }
    // measure time
    struct timespec start = {0,}, end = {0,}, diff = {0,};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for (i = 0; i < iter; i++) {
        r[i] = dot(a, b);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    performance(&start, &end, &diff);
    printf("%ld", diff.tv_sec*1000000000 + diff.tv_nsec);
    return 0;
}
