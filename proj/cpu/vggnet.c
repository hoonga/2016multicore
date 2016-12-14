#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>

static cl_kernel pool;
static cl_kernel conv;
static cl_context c;
static cl_command_queue q[4];

static void pooling2x2(float * input, float * output, int N)
{
  int i, j, k, l;
  for(i = 0; i < N; i++)
  {
    for(j = 0; j < N; j++)
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
      output[i * N + j] = max;
    }
  }
}

static void pooling_layer(float * inputs, float * outputs, int N, int D, int device, cl_event * event)
{
  cl_mem INPUTS = clCreateBuffer(c, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(float) * N * N * 4, inputs, NULL);
  cl_mem OUTPUTS = clCreateBuffer(c, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, sizeof(float) * N * N, outputs, NULL);
  clSetKernelArg(pool, 0, sizeof(cl_mem), (void*)&INPUTS);
  clSetKernelArg(pool, 1, sizeof(cl_mem), (void*)&OUTPUTS);
  clSetKernelArg(pool, 2, sizeof(int), (void*)&N);
  size_t global[1] = {D};
  size_t local[1] = {16};
  clEnqueueNDRangeKernel(q[device], pool, NULL, global, local, 0, NULL, event);
/*  int i;
  for(i = 0; i < D; i++)
  {
    float * input = inputs + i * N * N * 4;
    float * output = outputs + i * N * N;
    pooling2x2(input, output, N);
  }
*/
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
static void convolution_layer(float * inputs, float * outputs, float * filters, float * biases, int N, int D1, int D2, int device)
{
  //int i, j;

  memset(outputs, 0, sizeof(float) * N * N * D2);
  cl_mem INPUTS = clCreateBuffer(c, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(float) * N * N * D1, inputs, NULL);
  cl_mem OUTPUTS = clCreateBuffer(c, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, sizeof(float) * N * N * D2, outputs, NULL);
  cl_mem FILTERS = clCreateBuffer(c, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 3 * D1 * D2, filters, NULL);
  cl_mem BIASES = clCreateBuffer(c, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(float)*D2, biases, NULL);
  clSetKernelArg(conv, 0, sizeof(cl_mem), (void*)&INPUTS);
  clSetKernelArg(conv, 1, sizeof(cl_mem), (void*)&OUTPUTS);
  clSetKernelArg(conv, 2, sizeof(cl_mem), (void*)&FILTERS);
  clSetKernelArg(conv, 3, sizeof(cl_mem), (void*)&BIASES);
  clSetKernelArg(conv, 4, sizeof(int), (void*)&N);
  clSetKernelArg(conv, 5, sizeof(int), (void*)&D1);
  size_t global[1] = {D2};
  size_t local[1] = {16};
  clEnqueueNDRangeKernel(q[device], conv, 1, NULL, global, local, 0, NULL, NULL);

  // opencl the whole proccess?
  /*for(j = 0; j < D2; j++)
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
  }*/
}

static void fc_layer(float * input_neuron, float * output_neuron, float * weights, float * biases, int N, int M)
{
  int i, j;
  for(j = 0; j < M; j++)
  {
    float sum = 0;
    for(i = 0; i < N; i++)
    {
      sum += input_neuron[i] * weights[j * N + i];
    }
    sum += biases[j];
    output_neuron[j] = ReLU(sum);
  }
}

static void softmax(float * output)
{
  int i;
  float max = output[0];
  for(i = 1; i < 1000; i++)
  {
    max = (output[i] > max)?output[i]:max;
  }
  float sum = 0;
  for(i = 0; i < 1000; i++)
  {
    sum += exp(output[i] - max);
  }
  for(i = 0; i < 1000; i++)
  {
    output[i] = exp(output[i] - max) / sum;
  }
}

static int find_max(float * fc)
{
  int i;
  int maxid = 0;
  float maxval = 0;
  for(i = 0; i < 1000; i++)
  {
    if(maxval < fc[i])
    {
      maxval = fc[i];
      maxid = i;
    }
  }
  return maxid;
}


static float * get_param(float ** array, int size)
{
  float * subarray = *array;
  *array += size;
  return subarray;
}

void vggnet(float * images, float * network, int * labels, float * confidences, int num_images)
{
  float *c1_1[4], *c1_2[4], *c2_1[4], *c2_2[4], *c3_1[4], *c3_2[4], *c3_3[4], *c4_1[4], *c4_2[4], *c4_3[4], *c5_1[4], *c5_2[4], *c5_3[4]; // Convolution layers
  float *p1[4], *p2[4], *p3[4], *p4[4], *p5[4]; // Pooling layers
  float *fc1[4], *fc2[4], *fc3[4]; // Fully connected layers
  float *f1_1, *f1_2, *f2_1, *f2_2, *f3_1, *f3_2, *f3_3, *f4_1, *f4_2, *f4_3, *f5_1, *f5_2, *f5_3, *w1, *w2, *w3; // Filters and weights
  float *b1_1, *b1_2, *b2_1, *b2_2, *b3_1, *b3_2, *b3_3, *b4_1, *b4_2, *b4_3, *b5_1, *b5_2, *b5_3, *b1, *b2, *b3; // Biases
  int i, j;

  for(i = 0; i < 4; i++)
  {
    c1_1[i] = (float *)malloc(sizeof(float) * 224 * 224 * 64);
    c1_2[i] = (float *)malloc(sizeof(float) * 224 * 224 * 64);

    p1[i] = (float *)malloc(sizeof(float) * 112 * 112 * 64);

    c2_1[i] = (float *)malloc(sizeof(float) * 112 * 112 * 128);
    c2_2[i] = (float *)malloc(sizeof(float) * 112 * 112 * 128);

    p2[i] = (float *)malloc(sizeof(float) * 56 * 56 * 128);

    c3_1[i] = (float *)malloc(sizeof(float) * 56 * 56 * 256);
    c3_2[i] = (float *)malloc(sizeof(float) * 56 * 56 * 256);
    c3_3[i] = (float *)malloc(sizeof(float) * 56 * 56 * 256);

    p3[i] = (float *)malloc(sizeof(float) * 28 * 28 * 256);

    c4_1[i] = (float *)malloc(sizeof(float) * 28 * 28 * 512);
    c4_2[i] = (float *)malloc(sizeof(float) * 28 * 28 * 512);
    c4_3[i] = (float *)malloc(sizeof(float) * 28 * 28 * 512);

    p4[i] = (float *)malloc(sizeof(float) * 14 * 14 * 512);

    c5_1[i] = (float *)malloc(sizeof(float) * 14 * 14 * 512);
    c5_2[i] = (float *)malloc(sizeof(float) * 14 * 14 * 512);
    c5_3[i] = (float *)malloc(sizeof(float) * 14 * 14 * 512);

    p5[i] = (float *)malloc(sizeof(float) * 7 * 7 * 512);

    fc1[i] = (float *)malloc(sizeof(float) * 4096);
    fc2[i] = (float *)malloc(sizeof(float) * 4096);
    fc3[i] = (float *)malloc(sizeof(float) * 1000);
  }

  f1_1 = get_param(&network, 3 * 3 * 3 * 64);
  b1_1 = get_param(&network, 64);
  f1_2 = get_param(&network, 3 * 3 * 64 * 64);
  b1_2 = get_param(&network, 64);

  f2_1 = get_param(&network, 3 * 3 * 64 * 128);
  b2_1 = get_param(&network, 128);
  f2_2 = get_param(&network, 3 * 3 * 128 * 128);
  b2_2 = get_param(&network, 128);

  f3_1 = get_param(&network, 3 * 3 * 128 * 256);
  b3_1 = get_param(&network, 256);
  f3_2 = get_param(&network, 3 * 3 * 256 * 256);
  b3_2 = get_param(&network, 256);
  f3_3 = get_param(&network, 3 * 3 * 256 * 256);
  b3_3 = get_param(&network, 256);

  f4_1 = get_param(&network, 3 * 3 * 256 * 512);
  b4_1 = get_param(&network, 512);
  f4_2 = get_param(&network, 3 * 3 * 512 * 512);
  b4_2 = get_param(&network, 512);
  f4_3 = get_param(&network, 3 * 3 * 512 * 512);
  b4_3 = get_param(&network, 512);

  f5_1 = get_param(&network, 3 * 3 * 512 * 512);
  b5_1 = get_param(&network, 512);
  f5_2 = get_param(&network, 3 * 3 * 512 * 512);
  b5_2 = get_param(&network, 512);
  f5_3 = get_param(&network, 3 * 3 * 512 * 512);
  b5_3 = get_param(&network, 512);

  w1 = get_param(&network, 7 * 7 * 512 * 4096);
  b1 = get_param(&network, 4096);
  w2 = get_param(&network, 4096 * 4096);
  b2 = get_param(&network, 4096);
  w3 = get_param(&network, 4096 * 1000);
  b3 = get_param(&network, 1000);

  cl_platform_id pid;
  cl_device_id did[4];
  cl_program p;
  clGetPlatformIDs(1, &pid, NULL);
  clGetDeviceIDs(pid, CL_DEVICE_TYPE_CPU, 4, did, NULL);
  c = clCreateContext(NULL, 4, did, NULL,NULL,&res);
  for(i = 0; i < 4; i++)
    q = clCreateCommandQueue(c, did[i], 0, &res);
  const char* f = "./cpu.cl";
  FILE*fp = fopen(f, "r");
  char * src = malloc(6000);
  size_t size = fread(src, 1, 6000, fp);
  p = clCreateProgramWithSource(c, 1, (const char **)&src, (const size_t*)&size, NULL);
  clBuildProgram(p, 1, &did[0], NULL, NULL, NULL);
  pool = clCreateKernel(p, "pooling_layer", NULL);
  conv = clCreateKenrel(p, "convolution_layer", NULL);
  cl_event event[4];
  for(j = 0; j < num_images / 4 + 1; j++)
  {
    for(i = 0; i < 4; i++)
    {
      if (j * 4 + i > num_images)
        break;
      float * image = images + (j * 4 + i) * 224 * 224 * 3;

      convolution_layer(image, c1_1[i], f1_1, b1_1, 224, 3, 64, i);
      convolution_layer(c1_1[i], c1_2[i], f1_2, b1_2, 224, 64, 64, i);
      pooling_layer(c1_2[i], p1[i], 112, 64, i, NULL);

      convolution_layer(p1[i], c2_1[i], f2_1, b2_1, 112, 64, 128, i);
      convolution_layer(c2_1[i], c2_2[i], f2_2, b2_2, 112, 128, 128, i);
      pooling_layer(c2_2[i], p2[i], 56, 128, i, NULL);

      convolution_layer(p2[i], c3_1[i], f3_1, b3_1, 56, 128, 256, i);
      convolution_layer(c3_1[i], c3_2[i], f3_2, b3_2, 56, 256, 256, i);
      convolution_layer(c3_2[i], c3_3[i], f3_3, b3_3, 56, 256, 256, i);
      pooling_layer(c3_3[i], p3[i], 28, 256, i, NULL);

      convolution_layer(p3[i], c4_1[i], f4_1, b4_1, 28, 256, 512, i);
      convolution_layer(c4_1[i], c4_2[i], f4_2, b4_2, 28, 512, 512, i);
      convolution_layer(c4_2[i], c4_3[i], f4_3, b4_3, 28, 512, 512, i);
      pooling_layer(c4_3[i], p4[i], 14, 512, i, NULL);

      convolution_layer(p4[i], c5_1[i], f5_1, b5_1, 14, 512, 512, i);
      convolution_layer(c5_1[i], c5_2[i], f5_2, b5_2, 14, 512, 512, i);
      convolution_layer(c5_2[i], c5_3[i], f5_3, b5_3, 14, 512, 512, i);
      pooling_layer(c5_3[i], p5[i], 7, 512, i, &event[i]);
    }
    clWaitForEvents(4, event);
    for(i = 0; i < 4; i++)
    {
      fc_layer(p5[i], fc1[i], w1, b1, 7 * 7 * 512, 4096);
      fc_layer(fc1[i], fc2[i], w2, b2, 4096, 4096);
      fc_layer(fc2[i], fc3[i], w3, b3, 4096, 1000);

      softmax(fc3[i]);

      labels[j * 4 + i] = find_max(fc3[i]);
      confidences[j * 4 + i] = fc3[labels[j * 4 + i]];
    }
  }

  for(i = 0; i < 4; i++)
  {
    free(c1_1[i]);
    free(c1_2[i]);
    free(p1[i]);

    free(c2_1[i]);
    free(c2_2[i]);
    free(p2[i]);

    free(c3_1[i]);
    free(c3_2[i]);
    free(c3_3[i]);
    free(p3[i]);

    free(c4_1[i]);
    free(c4_2[i]);
    free(c4_3[i]);
    free(p4[i]);

    free(c5_1[i]);
    free(c5_2[i]);
    free(c5_3[i]);
    free(p5[i]);

    free(fc1[i]);
    free(fc2[i]);
    free(fc3[i]);
  }
}
