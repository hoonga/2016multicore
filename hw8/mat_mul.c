#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "timers.h"
#include <immintrin.h>

int print_matrix = 0;
int validation = 0;

void mat_mul( float * c, float * a, float * b, int NDIM )
{
	int i, j, k;
	
	// C = AB
#ifdef INTRINSIC
	for( i = 0; i < NDIM; i++ )
	{
		for( k = 0; k < NDIM; k++ )
		{
			__m256 a_vec = _mm256_broadcast_ss(&a[i*NDIM+k]);
			for( j = 0; j < NDIM; j+=8 )
			{
				__m256 c_vec = _mm256_load_ps(&c[i*NDIM+j]);
				__m256 b_vec = _mm256_load_ps(&b[k*NDIM+j]);
				__m256 tmp = _mm256_mul_ps(a_vec, b_vec);
				__m256 result = _mm256_add_ps(c_vec, tmp);
				_mm256_store_ps(&c[i*NDIM+j], result);
			}
		}
	}
#else
	for( i = 0; i < NDIM; i++ )
	{
		for( k = 0; k < NDIM; k++)
		{
			for( j = 0; j < NDIM; j++ )
			{
				c[i * NDIM + j] += a[i * NDIM + k] * b[k * NDIM + j];
			}
		}
	}
#endif
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

  a = (float *)_mm_malloc(sizeof(float) * NDIM * NDIM, 32);
  b = (float *)_mm_malloc(sizeof(float) * NDIM * NDIM, 32);
  c = (float *)_mm_malloc(sizeof(float) * NDIM * NDIM, 32);

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
