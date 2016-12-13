#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/opencl.h>

static void my_pooling(float * input, float * output, int N)
{
  // hard for gpu optimization, opencl optimization?
  int i, j, k, l;
  for(i = 0; i < N; i++)
  {
    for(k = 0; k < 2; k++)
    {
      for(j = 0; j < N; j++)
      {
        float max = output[i*N+j];
        for(l = 0; l < 2; l++)
        {
          float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
          max = (max > pixel) ? max : pixel;
        }
        output[i * N + j] = max;
      }
    }
  }
}

static void pooling(float * input, float * output, int N)
{
  int i, j, k, l;
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      float max = 0;
      for(k = 0; k < 2; k++)
      {
        for(l = 0; l < 2; l++)
        {
          float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
          max = (max > pixel) ? max : pixel;
        }
      }
      output[i*N+j] = max;
    }
  }
}
int timespec_subtract(struct timespec* result, struct timespec *x, struct timespec *y) {
  if (x->tv_nsec < y->tv_nsec) {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  return x->tv_sec < y->tv_sec;
}
int main() {
  float * a = (float*)malloc(sizeof(float)*512*512*4);
  float * b = (float*)malloc(sizeof(float)*512*512);
  struct timespec start, end, spent;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < 100; i++)
    my_pooling(a, b, 512);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < 100; i++)
    pooling(a, b, 512);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);

  return 0;
}
