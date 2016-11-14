#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include "timers.h"

#define NDIM    2048
#ifndef THREADS
#define THREADS 16
#endif

float a[NDIM][NDIM];
float b[NDIM][NDIM];
float c[NDIM][NDIM];

int print_matrix = 0;
int validation = 0;

#include<pthread.h>
struct workload
{
	int l;
	int w;
	int u;
	int h;
};

void *mat_mul_par(void* work)
{
	struct workload* W = (struct workload*) work;
	int i, j, k;
	int l = W->l;
	int w = W->w;
	int u = W->u;
	int h = W->h;
	for( i = 0; i < h; i++ )
	{
		for ( j = 0; j < w; j++ )
		{
			for ( k = 0; k < NDIM; k++)
			{
				c[u+i][l+j] += a[i][k] * b[k][j]; // expecting harsh cache miss on b[k][j]
			}
		}
	}
	return NULL;
}

void mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
	pthread_t t[THREADS];
	int i;
	struct workload work[THREADS];

	for( i = 0; i < THREADS; i++ )
	{
		#if THREADS == 2 // simply divide work up down
		work[i] = (struct workload) { 0, NDIM, i*NDIM/2, NDIM/2 };
		#elif THREADS == 4 // divide up down left right
		work[i] = (struct workload) { (i/2)*NDIM/2, NDIM/2, (i%2)*NDIM/2, NDIM/2};
		#elif THREADS == 8 // divide upup up down downdown left right
		work[i] = (struct workload) { (i/4)*NDIM/2, NDIM/2, (i%4)*NDIM/4, NDIM/4};
		#elif THREADS == 16 // divide upup up down downdown leftleft rightright left right
		work[i] = (struct workload) { (i/4)*NDIM/4, NDIM/4, (i%4)*NDIM/4, NDIM/4};
		#else // divide vertically
		work[i] = (struct workload) { 0, NDIM, (i%THREADS)*NDIM/THREADS, NDIM/THREADS};
		#endif
		pthread_create(&t[i], NULL, *mat_mul_par, &work[i]);
	}
	for ( i = 0; i < THREADS; i++ )
	{
		void *status;
		pthread_join(t[i], &status);
	}
	
	/* C = AB
	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			for( k = 0; k < NDIM; k++ )
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}*/
}

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
	int i, j, k;
	float sum;
	int validated = 1;

	printf("Validating the result..\n");
	
	// C = AB
	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			sum = 0;
			for( k = 0; k < NDIM; k++ )
			{
				sum += a[i][k] * b[k][j];
			}

			if( c[i][j] != sum )
			{
				printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i][j], sum );
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

void print_mat( float mat[NDIM][NDIM] )
{
	int i, j;

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			printf("%8.2lf ", mat[i][j]);
		}
		printf("\n");
	}
}

void print_help(const char* prog_name)
{
	printf("Usage: %s [-pvh]\n", prog_name );
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

	parse_opt( argc, argv );

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			a[i][j] = k;
			b[i][j] = k;
			k++;
		}
	}

	timer_start(1);
	mat_mul( c, a, b );
	timer_stop(1);

	printf("Time elapsed : %lf sec\n", timer_read(1));


	if( validation )
		check_mat_mul( c, a, b );

	if( print_matrix )
	{
		printf("MATRIX A: \n");
		print_mat(a);

		printf("MATRIX B: \n");
		print_mat(b);

		printf("MATRIX C: \n");
		print_mat(c);
	}

	return 0;
}
