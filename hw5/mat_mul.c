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

#define GPU1 1
#define GPU2 2
#define GPU4 4
#define CPU 8
#define CPUGPU1 9
#define CPUGPU4 12
#ifndef FLAG
#define FLAG GPU1
#endif

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

const char* kernels[] = {
"__kernel void mat_mult(__global const float* A,"
                       "__global const float *B,"
                       "__global float *C, int D) {"
"    size_t X = get_group_id(0)*8;\n"
"    size_t Y = get_group_id(1)*8;\n"
"    size_t x = get_local_id(0);\n"
"    size_t y = get_local_id(1);\n"
"    float sum = 0;\n"
"    for(int k = 0; k < D; k++) {\n"
"        sum += A[(Y+y)*D + k] * B[k*D+(X+x)];\n"
"    }"
"    C[(Y+y)*D+X+x] = sum;\n"
"}"
};

void mat_mul( float * c, float * a, float * b, int NDIM )
{
    // for error check
    cl_int res;
    // size of matrix
    size_t size = NDIM*NDIM*sizeof(float);

    // get platform
    cl_platform_id platform[1];
    CL_CHECK(clGetPlatformIDs(1, platform, NULL));

    // get all devices
#define GPUS (FLAG&7)
#define CPUS (FLAG>>3)
    int numdev = GPUS+CPUS;
    puts("getting devices");
    cl_device_id device[numdev];
#if (GPUS) != 0
    cl_device_id gpus[GPUS];
    puts("getting gpus");
    CL_CHECK(clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 4, gpus, NULL));
    for (int i = 0; i < GPUS; i++) {
        device[i] = gpus[i];
    }
#endif
#if (CPUS) != 0
    puts("getting cpus");
    cl_device_id cpus[CPUS];
    CL_CHECK(clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_CPU, 1, cpus, NULL));
    device[GPUS] = cpus[0];
#endif
    
    // create context
    puts("creating context");
    cl_context context = clCreateContext(0, numdev, device, NULL, NULL, &res); CL_CHECK(res);

    // create queues
    puts("creating queues");
    cl_command_queue q[numdev]; 
    for (int i = 0; i < numdev; i++) {
        printf("q[%d] done\n", i);
        q[i] = clCreateCommandQueue(context, device[i], 0, NULL);
    }

    // buffers
    cl_mem A[numdev];
    cl_mem C[numdev];
    for (int i = 0; i < numdev; i++) {
        A[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (i==numdev-1)?sizeof(float)*NDIM*(NDIM-NDIM/numdev*(numdev-1)):size/numdev, a+i*NDIM*NDIM/numdev, &res);
        C[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (i==numdev-1)?sizeof(float)*NDIM*(NDIM-NDIM/numdev*(numdev-1)):size/numdev, NULL, &res);
    }
    cl_mem B;
    B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, b, &res);

    // make program
    puts("compiling");
    cl_program program = clCreateProgramWithSource(context, 1, kernels, NULL, &res); CL_CHECK(res);
    puts("building");
    CL_CHECK(clBuildProgram(program, numdev, device, NULL, NULL, NULL));

    // make kernels
    cl_kernel mul_naive[numdev];
    puts("mul_naive kernel");
    size_t global[numdev][2];
    size_t local[numdev][2];
    for (int i = 0; i < numdev; i++) {
        mul_naive[i] = clCreateKernel(program, "mat_mult", &res); CL_CHECK(res);
        // naive mult
        CL_CHECK(clSetKernelArg(mul_naive[i], 0, sizeof(cl_mem), (void*) &A[i]));
        CL_CHECK(clSetKernelArg(mul_naive[i], 1, sizeof(cl_mem), (void*) &B));
        CL_CHECK(clSetKernelArg(mul_naive[i], 2, sizeof(cl_mem), (void*) &C[i]));
        CL_CHECK(clSetKernelArg(mul_naive[i], 3, sizeof(int), (void*) &NDIM));
        global[i][0] = NDIM;
        global[i][1] = (i==numdev-1)?(NDIM-NDIM/numdev*(numdev-1)):NDIM/numdev;
        local[i][0] = 8;
        local[i][1] = 8;
        // enque
        puts("enqueued");
        CL_CHECK(clEnqueueNDRangeKernel(q[i], mul_naive[i], 2, NULL, global[i], local[i], 0, NULL, NULL));
    }
    for (int i = 0; i < numdev; i++) {
        CL_CHECK(clEnqueueReadBuffer(q[i], C[i], CL_TRUE, 0, (i==numdev-1)?sizeof(float)*NDIM*(NDIM-NDIM/numdev*(numdev-1)):size/numdev, c+i*NDIM*NDIM/numdev, 0, NULL, NULL));
    }
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
