/*
 * Copyright 2010 Pawel Baran.
 * 
 * shatov33@gmail.com
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

/*
#define NUM_OF_ELEMENTS_PER_THREAD 2 // only even and > 0
#define NUM_OF_SHARED_ARRAY 6
#define NUM_OF_ELEMENTS 128
#define NUM_OF_THREADS_PER_BLOCK 512
*/

#define MAX_NUM_OF_THREADS_PER_BLOCK 512
#define MAX_NUM_OF_BLOCKS 512

// #define NUM_OF_ELEMENTS_PER_BLOCK 1024 // 2 to the power of k, where k = 1, 2, ...
// #define NUM_OF_THREADS_PER_BLOCK 256 // k, where k = 1, 2, ...
#define NUM_OF_ELEMENTS 4 // k, where k = 1, 2, ...
#define NUM_OF_ARRAYS_PER_BLOCK 6

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <time.h>

#include <cutil_inline.h>

#include <quicksort_kernel.cu>

#include "elem.h"
#include "elem.cu"

void runTest( int argc, char** argv);

extern "C" 
unsigned int compare( const char* reference, const float* data, 
                      const unsigned int len);
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    cutilExit(argc, argv);
}

void
runTest( int argc, char** argv) 
{
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int num_elements = NUM_OF_ELEMENTS;

		int num_elements_per_thread=2;
		int num_threads=num_elements/num_elements_per_thread;
		if(num_elements & 1)
			num_threads+=1;

		while(num_threads>MAX_NUM_OF_THREADS_PER_BLOCK*MAX_NUM_OF_BLOCKS){
			int a=1 & num_threads; // reszta z dzielenia przez 2
			num_threads>>=1;
			num_threads+=a;
			num_elements_per_thread<<=1;
		}

		unsigned int num_blocks=num_threads/MAX_NUM_OF_THREADS_PER_BLOCK;
		if(num_blocks*MAX_NUM_OF_THREADS_PER_BLOCK!=num_threads)
			num_blocks+=1;

		const unsigned int n = num_blocks*MAX_NUM_OF_THREADS_PER_BLOCK*num_elements_per_thread;

    unsigned int timer;
    cutilCheckError( cutCreateTimer(&timer));

		tab* table;
		table=make_tab(n);

		float* h_data=(float*)malloc(num_elements*sizeof(float));

		elem* d_elems;
      
    // initialize the input data on the host to be integer values
    // between 0 and 1000
		srand(time(NULL));
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        int el = 1000*(rand()/(float)RAND_MAX);
				if(rand() & 1) el*=-1;
				printf("el=%d\n",el);	

				table->elems[i].val = el;
				printf("h_data[%d] = %d;\n",i,table->elems[i].val);
    }
		for(unsigned int i=num_elements;i<n;++i){
			table->elems[i].val=0;
		}

    float* reference = (float*) malloc(num_elements*sizeof(float));
 		time_t start,end;
		time(&start);
    computeGold( reference, h_data, num_elements);
		time(&end);
		float time=end-start;

    cutilSafeCall( cudaMalloc( (void**) &d_elems, table->n*sizeof(elem)));

    cutilSafeCall( cudaMemcpy( d_elems, table->elems, table->n*sizeof(elem), cudaMemcpyHostToDevice) );

    dim3  grid(num_blocks, 1, 1); // 
    dim3  threads(MAX_NUM_OF_THREADS_PER_BLOCK, 1, 1);

		const unsigned int num_threads_per_block=MAX_NUM_OF_THREADS_PER_BLOCK;
		const unsigned int num_elements_per_block=n/num_blocks;


		// TO BE CHANGED
		const unsigned int shared_mem_size=16384;//NUM_OF_ARRAYS_PER_BLOCK*sizeof(float)*num_threads_per_block*2;

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("Kernel execution failed");

    printf("Running parallel quicksort for %d elements (n=%d)\n", num_elements,n);
  
    unsigned int numIterations = 1;
    
    printf("pbGPUQuicksort with params:\n- blocks=%d,\n- elements=%d,\n- elements2block=%d,\n- threads2block=%d\n"
			,num_blocks,num_elements,num_elements_per_block,num_threads_per_block);
    cutStartTimer(timer);
    for (unsigned int i = 0; i < numIterations; ++i)
    {
        quicksort_kernel<<< grid, threads, shared_mem_size >>>
            (d_elems, num_elements,num_elements_per_block,num_blocks);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Average time: %f ms\n\n", cutGetTimerValue(timer) / numIterations);
    printf("CPU time: %f ms\n\n", time);
    cutResetTimer(timer);

    // check for any errors
    cutilCheckMsg("Kernel execution failed");

		cutilSafeCall(cudaMemcpy( table->elems, d_elems,table->n*sizeof(elem),cudaMemcpyDeviceToHost));
		for( unsigned int i = 0; i < num_elements; ++i)
			printf("h_data[%d] = %d\n",i,table->elems[i]);
    printf("\nAuthor: Paweł Baran. e-mail: shatov33@gmail.com .\n");

    // cleanup memory
    free( h_data);
    free( reference);
    cutilSafeCall(cudaFree(d_elems));
		free_tab(table);
    cutilCheckError(cutDeleteTimer(timer));

    cudaThreadExit();
}
