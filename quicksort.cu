/*
 * Copyright 2010 Pawel Baran.
 * 
 * shatov33@gmail.com
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#define MAX_NUM_OF_THREADS_PER_BLOCK 1
#define MAX_NUM_OF_BLOCKS 65536

#define NUM_OF_ELEMENTS 4  // k, where k = 1, 2, ...
#define NUM_OF_ARRAYS_PER_BLOCK 6
#define MAX_SHARED_MEMORY_SIZE 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include <time.h>

#include <cutil_inline.h>

#include <quicksort_kernel.cu>

#include "elem.h"
#include "elem.cu"

void runTest( int argc, char** argv);

/*
extern "C" 
unsigned int compare( const int* reference, const float* data, 
					  const unsigned int len);
extern "C" 
void computeGold( int* reference, int* idata, const unsigned int len);
*/

int main( int argc, char** argv){
	runTest( argc, argv);
	cutilExit(argc, argv);
}

void down_sweep_for_sum(sum* d_sums,int num_sums,int n){
	int blocks_num=num_sums/(MAX_NUM_OF_THREADS_PER_BLOCK*2);

	if(MAX_NUM_OF_THREADS_PER_BLOCK*2*blocks_num!=num_sums)
		blocks_num+=1;

	dim3 grid(blocks_num,1,1);
	int threads_num;

	if(blocks_num==1)
		threads_num=num_sums/2;
	else
		threads_num=MAX_NUM_OF_THREADS_PER_BLOCK;

	if(!threads_num) return;

	if(blocks_num>1){
		dim3 grid2(1,1,1);
		dim3 threads2(blocks_num/2,1,1);
		int offset=MAX_NUM_OF_THREADS_PER_BLOCK*2;
		printf("2:offset=%d threads to sum of sums = %d \n",offset,threads2.x);
		accumulate_sum_of_sums2<<<grid2,threads2,6*sizeof(int)*threads2.x>>> (d_sums,2,offset);
		cutilCheckMsg("accumulate_sum_of_sums2");
	}

	dim3 threads(threads_num,1,1);
	printf("second of the accumulating functions: blocks=%d threads in each one=%d\n",blocks_num,threads_num);
	accumulate_sums2<<<grid,threads,6*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK>>> (d_sums,2,blocks_num==1);
	cutilCheckMsg("accumulate_sums2");
}

void up_sweep_for_sum(sum* d_sums,int num_sums,int n){
	int blocks_num=num_sums/(MAX_NUM_OF_THREADS_PER_BLOCK*2);

	if(MAX_NUM_OF_THREADS_PER_BLOCK*2*blocks_num!=num_sums)
		blocks_num+=1;

	dim3 grid(blocks_num,1,1);
	int threads_num;

	if(blocks_num==1)
		threads_num=num_sums/2;
	else
		threads_num=MAX_NUM_OF_THREADS_PER_BLOCK;

	if(!threads_num) return;

	dim3 threads(threads_num,1,1);
	printf("first of the accumulating functions: blocks=%d threads in each one=%d\n",blocks_num,threads_num);
	accumulate_sums<<<grid,threads,4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK>>> (d_sums,2);
	cutilCheckMsg("accumulate_sums");

	if(blocks_num==1) return;

	dim3 grid2(1,1,1);
	dim3 threads2(blocks_num/2,1,1);
	int offset=MAX_NUM_OF_THREADS_PER_BLOCK*2;
	printf("offset=%d threads to sum of sums = %d \n",offset,threads2.x);
	accumulate_sum_of_sums<<<grid2,threads2,4*sizeof(int)*threads2.x>>> (d_sums,2,offset);
	cutilCheckMsg("accumulate_sum_of_sums");
}

void quicksort(elem* d_elems,sum* d_sums,int num_elements,int n,int num_elements_per_block,int num_blocks,int num_blocks2){

	dim3  grid(num_blocks, 1, 1); // 
//	dim3  threads(num_elements/2, 1, 1);
	dim3  threads(MAX_NUM_OF_THREADS_PER_BLOCK, 1, 1);

	int num_threads2=num_blocks2/2;
	num_threads2+=num_blocks2 & 1;
	while(num_threads2>MAX_NUM_OF_THREADS_PER_BLOCK)
		if(num_threads2 & 1){
			num_threads2>>=1;
			++num_threads2;
		}else
			num_threads2>>=1;

	dim3 grid2(1,1,1);
	dim3 threads2(num_threads2,1,1);

	printf("mikki: threads=%d elems=%d elems_per_block%d blocks=%d blocks2=%d n=%d\n",num_threads2,num_elements,num_elements_per_block,num_blocks,num_blocks2,n);

	// zakomentowane na jakis czas:
	/*
	check_order<<< grid, threads, sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements,num_elements_per_block,num_blocks,num_blocks2);

	check_order2<<< grid2, threads2,  sizeof(int)*num_threads2 >>>
		(d_sums,num_blocks2);
		*/

/*	
	make_pivots<<< grid, threads, sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements,num_elements_per_block,num_blocks,num_blocks2);
		*/
	make_pivots<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK);

	up_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);

	down_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);
	cutilCheckMsg("down_sweep");

	make_pivots2<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK,num_blocks2);

	make_offsets<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK);

	up_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);

	down_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);
	cutilCheckMsg("down_sweep");

	make_offsets2<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK,num_blocks2);

	make_idowns<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK);

	up_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);

	down_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);
	cutilCheckMsg("down_sweep");

	make_idowns2<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK,num_blocks2);

	make_iup1s<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK);

	up_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);

	down_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);
	cutilCheckMsg("down_sweep");

	make_iup1s2<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK,num_blocks2);

	printf("threads=%d\n",threads.x);
	printf("num_blocks=%d\n",num_blocks);
	make_iup2s<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK, num_blocks);

	printf("num_blocks2=%d num_elements_per_block=%d\n",num_blocks2,num_elements_per_block);
	up_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);

	printf("numBLOCKS=%d num_el=%d\n",num_blocks2,num_elements_per_block);

	down_sweep_for_sum(d_sums,num_blocks2,num_elements_per_block);
	cutilCheckMsg("down_sweep");

	make_iup2s2<<< grid, threads, 4*sizeof(int)*MAX_NUM_OF_THREADS_PER_BLOCK >>>
		(d_elems, d_sums, num_elements_per_block/MAX_NUM_OF_THREADS_PER_BLOCK,num_blocks2,num_blocks);
	
	cutilCheckMsg("move_elems1");
	move_elems1<<< grid, threads >>> (d_elems, 2, num_elements);
	
	cutilCheckMsg("move_elems2");
	move_elems2<<< grid, threads >>> (d_elems, 2, num_elements);
	
	cutilCheckMsg("move_elems3");
	move_elems3<<< grid, threads >>> (d_elems, 2, num_elements);
/*	
		*/
}

void
runTest( int argc, char** argv) 
{
	/*
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
*/
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

	int num_blocks2=1;
	while(num_blocks2<num_blocks) num_blocks2<<=1;
	
	table=make_tab(n,num_blocks2);

	elem* d_elems;
	sum* d_sums;
	  
	// initialize the input data on the host to be integer values
	// between 0 and 1000
	srand(time(NULL));
	printf("elems(n=%d) to be sorted:\n",num_elements);
	for( unsigned int i = 0; i < num_elements; ++i) 
	{
		int elval = 1000*(rand()/(float)RAND_MAX);
		if(rand() & 1) elval*=-1;

		/*
		// UWAGA: ponizej jest kod do testow:
		if(rand() & 1) elval=1;
		else elval=0;
		*/

		table->elems[i].val = elval;
		table->elems[i].seg_flag2=0;
		table->elems[i].pivot=0;
		printf(" %d ",table->elems[i].val);
	}
	printf(" ;\n");
	table->elems[0].seg_flag2=1;
	table->elems[0].val=-198;
	table->elems[1].val=-975;
	table->elems[2].val=62;
	table->elems[3].val=-234;
	for(unsigned int i=num_elements;i<n;++i){
		table->elems[i].val=INT_MAX;
		table->elems[i].seg_flag2=1;
		table->elems[i].pivot=0;
	}
	for(int i=0;i<num_blocks2;++i)
		table->sums[i].val=0;

	cutilSafeCall( cudaMalloc( (void**) &d_elems, table->n*sizeof(elem)));
	cutilSafeCall( cudaMalloc( (void**) &d_sums, num_blocks2*sizeof(sum)));
	cutilSafeCall( cudaMemcpy( d_elems, table->elems, table->n*sizeof(elem), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy( d_sums, table->sums, num_blocks2*sizeof(sum), cudaMemcpyHostToDevice) );

//	const unsigned int num_threads_per_block=MAX_NUM_OF_THREADS_PER_BLOCK;
	const unsigned int num_elements_per_block=n/num_blocks;


	// TO BE CHANGED
//	const unsigned int shared_mem_size=16384;//NUM_OF_ARRAYS_PER_BLOCK*sizeof(float)*num_threads_per_block*2;

	// make sure there are no CUDA errors before we start
	cutilCheckMsg("Kernel execution failed");

	printf("Running parallel quicksort for %d elements (n=%d)\n", num_elements,n);
  
	unsigned int numIterations = 1;
	
	printf("pbGPUQuicksort with params:\n- blocks=%d,\n- elements=%d,\n- elements2thread=%d\n"
			,num_blocks,num_elements,num_elements_per_thread);
	cutilCheckError(cutStartTimer(timer));
	for (unsigned int i = 0; i < numIterations; ++i)
	{
		quicksort(d_elems,d_sums,num_elements,n,num_elements_per_block,num_blocks,num_blocks2);
	}
	cudaThreadSynchronize();
	cutilCheckError(cutStopTimer(timer));
	printf("Average time: %f ms\n\n", cutGetTimerValue(timer) / numIterations);
//	printf("CPU time: %f ms\n\n", time);
	cutResetTimer(timer);

	// check for any errors
	cutilCheckMsg("Kernel execution failed");

	cutilSafeCall(cudaMemcpy( table->elems, d_elems,table->n*sizeof(elem),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy( table->sums, d_sums,num_blocks2*sizeof(sum),cudaMemcpyDeviceToHost));
	for( unsigned int i = 0; i < n; ++i)
		printf("val[%d] = %d seg_flag2 = %d pivot[%d] = %d offset=%d idown=%d iup=%d iup2=%d flag=%d\n",i,table->elems[i].val,table->elems[i].seg_flag2,i,table->elems[i].pivot,table->elems[i].offset,table->elems[i].idown,table->elems[i].iup1,table->elems[i].iup2,table->elems[i].seg_flag);
	for( unsigned int i = 0; i < num_blocks2; ++i)
		printf("sum[%d] = %d seg_flag=%d\n",i,table->sums[i].val,table->sums[i].seg_flag);
//	printf("sum[%d] = %d\n",0,table->sums[0].val);
//	printf("thread[%d] = %d\n",n-1,table->elems[n-1].val);
	printf("\nAuthor: Paweł Baran. e-mail: shatov33@gmail.com .\n");

	// cleanup memory
	printf("a3\n");
	cutilSafeCall(cudaFree(d_elems));
	printf("a4\n");
	cutilSafeCall(cudaFree(d_sums));
	printf("a5\n");
//	free_tab(table);
	printf("a6\n");
	cutilCheckError(cutDeleteTimer(timer));
	printf("a7\n");

	cudaThreadExit();
}
