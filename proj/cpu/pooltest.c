#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static void convolution3x3(float * input, float * output, float * filter, int N)
{
  // perfect for opencl
  int i, j, k, l;
  for(i = 0; i < N; i++)
  {
    for(j = 0; j < N; j++)
    {
      float sum = 0;
      for(k = 0; k < 3; k++)
      {
        for(l = 0; l < 3; l++)
        {
          int x = i + k - 1;
          int y = j + l - 1;
          if(x >= 0 && x < N && y >= 0 && y < N)
            sum += input[x * N + y] * filter[k * 3 + l];
        }
      }
      output[i * N + j] += sum;
    }
  }
}
#define ReLU(x) (((x)>0)?(x):0)
static void convolution_layer(float * inputs, float * outputs, float * filters, float * biases, int N, int D1, int D2)
{
  int i, j;

  memset(outputs, 0, sizeof(float) * N * N * D2);

  // opencl the whole proccess?
  for(j = 0; j < D2; j++)
  {
    for(i = 0; i < D1; i++)
    {
      float * input = inputs + N * N * i;
      float * output = outputs + N * N * j;
      float * filter = filters + 3 * 3 * (j * D1 + i);
      convolution3x3(input, output, filter, N);
    }
  }

  for(i = 0; i < D2; i++)
  {
    float * output = outputs + N * N * i;
    float bias = biases[i];
    // optimizable
    for(j = 0; j < N * N; j++)
    {
      output[j] = ReLU(output[j] + bias);
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
  int D1 = 64, D2 = 64;
  int N = 224;
  float * in = (float*)malloc(sizeof(float)*N*N*D1);
  float * out = (float*)malloc(sizeof(float)*N*N*D2);
  float * fil = (float*)malloc(sizeof(float)*3*3*D1*D2);
  float * bias = (float*)malloc(sizeof(float)*D2);
  in[0] = 8.0;
  fil[0] = 23;
  bias[0] = 2;
  struct timespec start, end, spent;
  clock_gettime(CLOCK_MONOTONIC, &start);
    convolution_layer(in, out, fil, bias, N, D1, D2);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  printf("%f\n", out[0]);
  out[0] = 0;
  clock_gettime(CLOCK_MONOTONIC, &start);
  cl_platform_id pid;
  cl_device_id did;
  cl_context c;
  cl_command_queue q;
  cl_program p;
  cl_mem IN, OUT, FIL, BIAS;
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
  IN = clCreateBuffer(c, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, N*N*D1*sizeof(float), in, NULL);
  OUT = clCreateBuffer(c, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, N*N*D2*sizeof(float), out, NULL);
  FIL = clCreateBuffer(c, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, 3*3*D1*D2*sizeof(float), fil, NULL);
  BIAS = clCreateBuffer(c, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, D2*sizeof(float), bias, NULL);
  p = clCreateProgramWithSource(c, 1, (const char **)&src, (const size_t*)&size, NULL);
  clBuildProgram(p, 1, &did, NULL, NULL, NULL);
  size_t len;
  char *buffer;
  clGetProgramBuildInfo(p, did, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  buffer = malloc(len);
  clGetProgramBuildInfo(p, did, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
  printf("%s\n", buffer);
  k = clCreateKernel(p, "convolution_layer", NULL);
  clSetKernelArg(k, 0, sizeof(cl_mem), (void*)&IN);
  clSetKernelArg(k, 1, sizeof(cl_mem), (void*)&OUT);
  clSetKernelArg(k, 2, sizeof(cl_mem), (void*)&FIL);
  clSetKernelArg(k, 3, sizeof(cl_mem), (void*)&BIAS);
  clSetKernelArg(k, 4, sizeof(int), (void*)&N);
  clSetKernelArg(k, 5, sizeof(int), (void*)&D1);
  size_t global[1] = {D2};
  size_t local[1] = {16};
  clEnqueueNDRangeKernel(q, k, 1, NULL, global, local, 0, NULL, NULL);
  clEnqueueReadBuffer(q, OUT, CL_TRUE, 0, N*N*D2*sizeof(float), out, 0, NULL, NULL);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  printf("%f\n", out[0]);

  return 0;
}
