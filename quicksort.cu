/*
 * Copyright 2010 Pawel Baran.
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

// liczba tablic, ktorych rozmiar rowny n_1
#define NUM_OF_SHARED_ARRAY 6 // for num_of_elements=1024 should be 4 (unreachable)
#define NUM_OF_ELEMENTS 128
#define NUM_OF_ELEMENTS_PER_THREAD 2
#define NUM_OF_THREADS_PER_BLOCK 512

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <time.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <quicksort_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// regression test functionality
extern "C" 
unsigned int compare( const char* reference, const float* data, 
                      const unsigned int len);
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int num_elements = NUM_OF_ELEMENTS;

		// liczba elemnt√w 
		unsigned int num_elements_mem;
		int i=1;
		while(i<num_elements) i<<=1;
		num_elements_mem=i;
//    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int*)&num_elements);

    unsigned int timer;
    cutilCheckError( cutCreateTimer(&timer));
    
    const unsigned int num_threads = (int)ceil(NUM_OF_ELEMENTS
			/NUM_OF_ELEMENTS_PER_THREAD);
		const unsigned int num_blocks = 1+(int)ceil(num_threads/NUM_OF_THREADS_PER_BLOCK);
    const unsigned int mem_size = sizeof( float) * num_elements;
		const unsigned int num_elements_per_block = NUM_OF_ELEMENTS_PER_THREAD
			*NUM_OF_THREADS_PER_BLOCK;

		const unsigned int shared_mem_size = NUM_OF_SHARED_ARRAY*sizeof(float)*num_blocks
			*num_elements_per_block;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( mem_size);
      
    // initialize the input data on the host to be integer values
    // between 0 and 1000
		srand(time(NULL));
/*h_data[0] = 4;
h_data[1] = 3;
h_data[2] = 2;
h_data[3] = 1;
h_data[4] = 4;
h_data[5] = 5;
h_data[6] = 7;
h_data[7] = 6;
h_data[8] = 1;
h_data[9] = 10;
h_data[10] = 12;
h_data[11] = 13;
h_data[12] = 15;
h_data[13] = 14;
h_data[14] = 16;
h_data[15] = 11;*/
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
				printf("h_data[%d] = %f;\n",i,h_data[i]);
    }

    // compute reference solution
    float* reference = (float*) malloc( mem_size); 
 		time_t start,end;
		time(&start);
    computeGold( reference, h_data, num_elements);
		time(&end);
		float time=end-start;

    // allocate device memory input and output arrays
    float* d_idata;
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // copy host memory to device input array
    cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

    // setup execution parameters
    // Note that these scans only support a single thread-block worth of data,
    // but we invoke them here on many blocks so that we can accurately compare
    // performance
#ifndef __DEVICE_EMULATION__
    dim3  grid(1, 1, 1);  
#else
    dim3  grid(1, 1, 1); // only one run block in device emu mode or it will be too slow
#endif
    dim3  threads(num_threads, 1, 1);

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("Kernel execution failed");

    printf("Running parallel quicksort for %d elements\n", num_elements);
  
    // execute the kernels
    unsigned int numIterations = 1;
//    threads.x = num_threads;
    
    printf("quicksort:\n");
    cutStartTimer(timer);
    for (unsigned int i = 0; i < numIterations; ++i)
    {
        quicksort_kernel<<< grid, threads, shared_mem_size >>>
            (d_odata, d_idata, num_elements, num_elements_mem);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Average time: %f ms\n\n", cutGetTimerValue(timer) / numIterations);
    printf("CPU time: %f ms\n\n", time);
    cutResetTimer(timer);

    // check for any errors
    cutilCheckMsg("Kernel execution failed");

        // copy result from device to host
        cutilSafeCall(cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, 
                                   cudaMemcpyDeviceToHost));

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
            float epsilon = 0.0f;
            unsigned int result_regtest = cutComparefe( reference, h_data, num_elements, epsilon);
            char* names[] = {"quicksort"};
            printf( "%s: Test %s h[0]=%f\n", names[0], (1 == result_regtest) ? "PASSED" : "FAILED", h_data[1]);
    				for( unsigned int i = 0; i < num_elements; ++i) 
				    {
							printf("h_data[%d] = %f\n",i,h_data[i]);
				    }
        }

    printf("\nAuthor: Pawe≈Ç Baran. e-mail: shatov33@gmail.com .\n");

    // cleanup memory
    free( h_data);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));
    cutilCheckError(cutDeleteTimer(timer));

    cudaThreadExit();
}
