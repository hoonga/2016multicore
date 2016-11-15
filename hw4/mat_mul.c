#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "timers.h"
#include <CL/opencl.h>
#include <string.h>

int print_matrix = 0;
int validation = 0;
#ifdef TRANSMUL
const char * ktranspose =
"__kernel void transpose(__global const float* src, __global float* dest, \n"
"__local float *buf, int NDIM) {\n"
"    size_t X = get_global_id(0);\n"
"    size_t Y = get_global_id(1);\n"
"    buf[get_local_id(1)*9 + get_local_id(0)] = src[Y*NDIM+X];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    X = get_group_id(1) * 8 + get_local_id(0);\n"
"    Y = get_group_id(0) * 8 + get_local_id(1);\n"
"    dest[Y*NDIM + X] = buf[get_local_id(0)*9 + get_local_id(1)];\n"
"}";
const char * kmat_mult =
"__kernel void mat_mult(__global const float* A, __global const float* B, \n"
"__global float* C, int D) {\n"
"    size_t X = get_group_id(0)*64;\n"
"    size_t x = get_local_id(0);\n"
"    float sum = 0;\n"
"    size_t Y = X%D;\n"
"    Y = Y + x;\n"
"    Y = Y*D;\n"
"    for(int k = 0; k < D; k++)\n"
"        sum += A[X/D*D + k] * B[Y + k];\n"
"    C[X + x] = sum;\n"
"}";
#else
const char * kmat_mul =
"__kernel void mat_mul(__global const float* A, __global const float* B, \n"
"__global float* C, int D) {\n"
"    size_t X = get_group_id(0)*64;\n"
"    size_t x = get_local_id(0);\n"
"    int k = 0;\n"
"    float sum = 0;\n"
"    for(;k < D; k++)\n"
"        sum += A[X/D*D + k] * B[k*D+(X&(D-1))+x];\n"
"    C[X+x] = sum;\n"
"}";
#endif

void mat_mul( float * c, float * a, float * b, int NDIM )
{
    cl_device_id device[1];
    cl_platform_id platform[1];
    cl_uint numdev;
    cl_int res;
    size_t size = NDIM*NDIM*sizeof(float);
    int where = 0;
    do {
        // prepare
        res = clGetPlatformIDs(1, platform, &numdev); where++; if( res != CL_SUCCESS ) break; // 1
        res = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_CPU, 1, device, &numdev); where++; if( res != CL_SUCCESS ) break; // 2
        cl_context context = clCreateContext(0, 1, device, NULL, NULL, &res); where++; if( res != CL_SUCCESS ) break; // 3
        cl_command_queue q = clCreateCommandQueue(context, device[0], 0, NULL); where++; if( res != CL_SUCCESS ) break; // 4
        cl_mem A, B, C;
        A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, a, &res); where++; if( res != CL_SUCCESS ) break; // 5
        B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, b, &res); where++; if( res != CL_SUCCESS ) break; // 6
#ifdef TRANSMUL
        cl_mem BT = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &res); where++; if( res != CL_SUCCESS ) break; // 7
#endif
        C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &res); where++; if( res != CL_SUCCESS ) break; // 8
#ifdef TRANSMUL
        size_t len = strlen(ktranspose);
        cl_program ptrans = clCreateProgramWithSource(context, 1, (const char**) &ktranspose, &len, &res); where++; if( res != CL_SUCCESS ) break;// 9
        res = clBuildProgram(ptrans, 1, device, NULL, NULL, NULL); where++; if( res != CL_SUCCESS ) break; // 10
        cl_kernel trans = clCreateKernel(ptrans, "transpose", NULL); where++; if( res != CL_SUCCESS ) break; // 11
        len = strlen(kmat_mult);
        cl_program pmul = clCreateProgramWithSource(context, 1, (const char**) &kmat_mult, &len, &res); where++; if( res != CL_SUCCESS ) break; // 12
        res = clBuildProgram(pmul, 1, device, NULL, NULL, NULL); where++; if( res != CL_SUCCESS ) break; // 13
        cl_kernel mult = clCreateKernel(pmul, "mat_mult", &res); where++; if( res != CL_SUCCESS ) break; // 14
        // trans
        res = clSetKernelArg(trans, 0, sizeof(cl_mem), (void*) &B); where++; if( res != CL_SUCCESS ) break; // 15
        res = clSetKernelArg(trans, 1, sizeof(cl_mem), (void*) &BT); where++; if( res != CL_SUCCESS ) break; // 16
        res = clSetKernelArg(trans, 2, sizeof(9*8*sizeof(float)), NULL); where++; if( res != CL_SUCCESS ) break; // 17
        res = clSetKernelArg(trans, 3, sizeof(int), (void*) &NDIM); where++; if( res != CL_SUCCESS ) break; // 18
        // mult
        res = clSetKernelArg(mult, 0, sizeof(cl_mem), (void*) &A); where++; if( res != CL_SUCCESS ) break; // 19
        res = clSetKernelArg(mult, 1, sizeof(cl_mem), (void*) &BT); where++; if( res != CL_SUCCESS ) break; // 20
        res = clSetKernelArg(mult, 2, sizeof(cl_mem), (void*) &C); where++; if( res != CL_SUCCESS ) break; // 21
        res = clSetKernelArg(mult, 3, sizeof(int), (void*) &NDIM); where++; if( res != CL_SUCCESS ) break; // 22
        // enque
        size_t tg[] = {NDIM, NDIM};
        size_t tl[] = {8, 8};
        res = clEnqueueNDRangeKernel(q, trans, 2, NULL, tg, tl, 0, NULL, NULL); where++; if( res != CL_SUCCESS ) break;
        size_t global[] = {NDIM * NDIM};
        size_t local[] = {8 * 8};
        printf("enque mul\n");
        res = clEnqueueNDRangeKernel(q, mult, 1, NULL, global, local, 0, NULL, NULL); where++; if( res != CL_SUCCESS ) break;
#else
        size_t len = strlen(kmat_mul);
        cl_program program = clCreateProgramWithSource(context, 1, (const char**) &kmat_mul, &len, &res); where++; if( res != CL_SUCCESS ) break;
        res = clBuildProgram(program, 1, device, NULL, NULL, NULL); where++; if( res != CL_SUCCESS ) break;
        cl_kernel mul = clCreateKernel(program, "mat_mul", &res); where++; if( res != CL_SUCCESS ) break;
        clSetKernelArg(mul, 0, sizeof(cl_mem), (void*) &A);
        clSetKernelArg(mul, 1, sizeof(cl_mem), (void*) &B);
        clSetKernelArg(mul, 2, sizeof(cl_mem), (void*) &C);
        clSetKernelArg(mul, 3, sizeof(int), (void*) &(NDIM));
        size_t global[] = {NDIM * NDIM};
        size_t local[] = {8 * 8};
        res = clEnqueueNDRangeKernel(q, mul, 1, NULL, global, local, 0, NULL, NULL); where++; if (res != CL_SUCCESS) break;
#endif
        printf("done");
        clEnqueueReadBuffer(q, C, CL_TRUE, 0, size, c, 0, NULL, NULL);
    } while(0);
    if (res != CL_SUCCESS) printf("failed with : %d at %d \n", res, where);
}

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul( float * c, float * a, float * b, int NDIM )
{
	int i, j, k, x, y;
	float sum, result;
	int validated = 1;

	printf("Validating the result..\n");
	
  srand(time(NULL));
	// C = AB
	for( x = 0; x < 10; x++ )
	{
		for( y = 0; y < 10; y++ )
		{
      i = rand() % NDIM;
      j = rand() % NDIM;
			sum = 0;

			for( k = 0; k < NDIM; k++ )
			{
				sum += a[i * NDIM + k] * b[k * NDIM + j];
			}

      result = c[i * NDIM + j];

			if( result - sum > 0.001 || result - sum < -0.001 )
			{
				printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i * NDIM + j], sum );
				validated = 0;
			}
		}
	}

	printf("Validation : ");
	if( validated )
		printf("SUCCESSFUL.\n");
	else
		printf("FAILED.\n");
}

void print_mat( float * mat, int NDIM )
{
	int i, j;

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			printf("%8.2lf ", mat[i * NDIM + j]);
		}
		printf("\n");
	}
}

void print_help(const char* prog_name)
{
	printf("Usage: %s [NDIM] [-pvh]\n", prog_name );
	printf("\n");
	printf("OPTIONS\n");
	printf("  -p : print matrix data.\n");
	printf("  -v : validate matrix multiplication.\n");
	printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "pvhikjs:")) != -1 )
	{
		switch(opt)
		{
		case 'p':
			// print matrix data.
			print_matrix = 1;
			break;

		case 'v':
			// validation
			validation = 1;
			break;

		case 'h':
		default:
			print_help(argv[0]);
			exit(0);
			break;
		}
	}
}

int main(int argc, char** argv)
{
	int i, j, k = 1;
  int NDIM = 1024;
  float * a, * b, * c;

  NDIM = atoi(argv[1]);
	parse_opt( argc, argv );

  a = (float *)malloc(sizeof(float) * NDIM * NDIM);
  b = (float *)malloc(sizeof(float) * NDIM * NDIM);
  c = (float *)malloc(sizeof(float) * NDIM * NDIM);

  printf("%d x %d x %d\n", NDIM, NDIM, NDIM);

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			a[i * NDIM + j] = k;
			b[i * NDIM + j] = k;
			k++;
		}
	}

	timer_start(1);
	mat_mul( c, a, b, NDIM );
	timer_stop(1);

	printf("Time elapsed : %lf sec\n", timer_read(1));

	if( validation )
		check_mat_mul( c, a, b, NDIM );

	if( print_matrix )
	{
		printf("MATRIX A: \n");
		print_mat(a, NDIM);

		printf("MATRIX B: \n");
		print_mat(b, NDIM);

		printf("MATRIX C: \n");
		print_mat(c, NDIM);
	}

	return 0;
}
