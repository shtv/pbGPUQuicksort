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

// #define NUM_OF_ELEMENTS_PER_BLOCK 1024 // 2 to the power of k, where k = 1, 2, ...
// #define NUM_OF_THREADS_PER_BLOCK 256 // k, where k = 1, 2, ...
#define NUM_OF_ELEMENTS 128 // k, where k = 1, 2, ...
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

tab* table;

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

    unsigned int num_elements_per_block = 1; // NUM_OF_ELEMENTS_PER_BLOCK;
		while(num_elements_per_block*num_elements_per_block<num_elements) num_elements_per_block<<=1;

    unsigned int num_threads_per_block = 1; // NUM_OF_THREADS_PER_BLOCK;
		while(num_threads_per_block<MAX_NUM_OF_THREADS_PER_BLOCK && num_threads_per_block<2*num_elements_per_block) num_threads_per_block<<=1;

/*
		// nearest power of 2 greater than or equal num_elements_per_block
		unsigned int num_elements2;
		int i=1;
		while(i<num_elements_per_block) i<<=1;
		num_elements2=i;
*/
//    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int*)&num_elements);

    unsigned int timer;
    cutilCheckError( cutCreateTimer(&timer));
    
		const unsigned int num_blocks = 1+(int)ceil(num_elements/num_elements_per_block); // num_blocks - 1 should not be greater than num_elements_per_block, otherwise one thread in special block has more elements to computing
    const unsigned int mem_size = sizeof(float) * num_elements;

		const unsigned int shared_mem_size = NUM_OF_ARRAYS_PER_BLOCK*sizeof(float)*num_elements_per_block;

		const unsigned int n=(num_blocks-1)*num_elements_per_block;

    float* h_data = (float*) malloc( mem_size);
		table=make_tab(n);

		elem* d_elems;
      
    // initialize the input data on the host to be integer values
    // between 0 and 1000
		srand(time(NULL));
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
				
				if(rand() & 1) h_data[i]*=-1.0;

				table->elems[i].val = (int)h_data[i];
				printf("h_data[%d] = %f;\n",i,h_data[i]);
    }
		for(unsigned int i=num_elements;i<n;++i){
			table->elems[i].val=0;
		}

    // compute reference solution
    float* reference = (float*) malloc( mem_size); 
 		time_t start,end;
		time(&start);
    computeGold( reference, h_data, num_elements);
		time(&end);
		float time=end-start;

    // allocate device memory input and output arrays
//    float* d_idata;
//    float* d_odata;
//    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
//    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_elems, table->n*sizeof(elem)));

    // copy host memory to device input array
    cutilSafeCall( cudaMemcpy( d_elems, table->elems, table->n*sizeof(elem), cudaMemcpyHostToDevice) );

/*#ifndef __DEVICE_EMULATION__
    dim3  grid(1, 1, 1);  
#else*/
    dim3  grid(num_blocks, 1, 1); // 
/*#endif*/
    dim3  threads(num_threads_per_block, 1, 1);

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("Kernel execution failed");

    printf("Running parallel quicksort for %d elements\n", num_elements);
  
    // execute the kernels
    unsigned int numIterations = 1;
    
    printf("pbGPUQuicksort:\n");
    cutStartTimer(timer);
    for (unsigned int i = 0; i < numIterations; ++i)
    {
        quicksort_kernel<<< grid, threads, shared_mem_size >>>
            (d_elems, num_elements,num_elements_per_block);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Average time: %f ms\n\n", cutGetTimerValue(timer) / numIterations);
    printf("CPU time: %f ms\n\n", time);
    cutResetTimer(timer);

    // check for any errors
    cutilCheckMsg("Kernel execution failed");

        // copy result from device to host
        cutilSafeCall(cudaMemcpy( table->elems, d_elems,table->n*sizeof(elem),cudaMemcpyDeviceToHost));

        // If this is a regression test write the results to a file
        if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
        {
            // write file for regression test 
            cutWriteFilef( "./data/result.dat", h_data, num_elements, 0.0);
        }
        else 
        {
            // custom output handling when no regression test running
            // in this case check if the result is equivalent to the expected soluion
            
            // We can use an epsilon of 0 since values are integral and in a range 
            // that can be exactly represented
 /*           float epsilon = 0.0f;
            unsigned int result_regtest = cutComparefe( reference, h_data, num_elements, epsilon);
            char* names[] = {"quicksort"};
            printf( "%s: Test %s h[0]=%f\n", names[0], (1 == result_regtest) ? "PASSED" : "FAILED", h_data[1]);
    				for( unsigned int i = 0; i < num_elements; ++i) 
				    {
							printf("h_data[%d] = %f\n",i,h_data[i]);
				    }
*/
        }
    printf("\nAuthor: Paweł Baran. e-mail: shatov33@gmail.com .\n");

    // cleanup memory
    free( h_data);
    free( reference);
//    cutilSafeCall(cudaFree(d_idata));
//    cutilSafeCall(cudaFree(d_odata));
    cutilSafeCall(cudaFree(d_elems));
		free_tab(table);
    cutilCheckError(cutDeleteTimer(timer));

    cudaThreadExit();
}
