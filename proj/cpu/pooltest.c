#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <time.h>

const char *getErrorString(cl_int error) {
switch(error) {
    // run-time and JIT compiler errors
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

void CL_CHECK(cl_int err) {
    if(err) {puts(getErrorString(err)); exit(0);}
}

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
  int i;
  float * a = (float*)malloc(sizeof(float)*512*512*4);
  float * b = (float*)malloc(sizeof(float)*512*512);
  a[0] = 8.0;
  struct timespec start, end, spent;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < 100; i++)
    my_pooling(a, b, 512);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  printf("%f\n", b[0]);
  b[0] = 0;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < 100; i++)
    pooling(a, b, 512);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  printf("%f\n", b[0]);
  b[0] = 0;
  clock_gettime(CLOCK_MONOTONIC, &start);
  cl_platform_id pid;
  cl_device_id did;
  cl_context c;
  cl_command_queue q;
  cl_program p;
  cl_mem A;
  cl_mem B;
  cl_kernel k;
  const char* f = "./cpu.cl";
  FILE*fp = fopen(f, "r");
  char * src = malloc(6000);
  float * dest = malloc(sizeof(float)*512*512);
  size_t size = fread(src, 1, 6000, fp);
  cl_int res;
  CL_CHECK(clGetPlatformIDs(1, &pid, NULL));
  CL_CHECK(clGetDeviceIDs(pid, CL_DEVICE_TYPE_CPU, 1, &did, NULL));
  c = clCreateContext(NULL, 1, &did, NULL,NULL,&res);
  CL_CHECK(res);
  q = clCreateCommandQueue(c, did, 0, &res);
  CL_CHECK(res);
  A = clCreateBuffer(c, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, 512*512*4*sizeof(float), a, NULL);
  B = clCreateBuffer(c, CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR, 512*512*4*sizeof(float), b, NULL);
  p = clCreateProgramWithSource(c, 1, (const char **)&src, (const size_t*)&size, NULL);
  clBuildProgram(p, 1, &did, NULL, NULL, NULL);
  size_t len;
  char *buffer;
  clGetProgramBuildInfo(p, did, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  buffer = malloc(len);
  clGetProgramBuildInfo(p, did, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
  printf("%s\n", buffer);
  k = clCreateKernel(p, "pooling_layer", NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), (void*)&A);
  clSetKernelArg(k, 1, sizeof(cl_mem), (void*)&B);
  int N = 512;
  clSetKernelArg(k, 2, sizeof(int), (void*)&N);
  size_t global[1] = {1};
  size_t local[1] = {1};
  for(i = 0; i < 100; i++)
    clEnqueueNDRangeKernel(q, k, 1, NULL, global, local, 0, NULL, NULL);
  clEnqueueReadBuffer(q, B, CL_TRUE, 0, 512*512*sizeof(float), dest, 0, NULL,NULL); 
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  printf("%f\n", dest[0]);
  printf("%f\n", b[0]);

  return 0;
}
