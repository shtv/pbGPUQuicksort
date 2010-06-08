/*
 * Copyright 2010 Pawel Baran.
 *
 */

#ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
#define _SCAN_WORKEFFICIENT_KERNEL_H_

#include "elem.h"

#define MAX_REGISTERS_PER_THREAD 4

#define NORMAL 0
#define PIVOTS 1
#define KEYS 2
#define SEG_FLAGS 3

/*
__device__ float is_sorted(const float* keys,int n){
	int thid=threadIdx.x;
	int offset=1;
	extern __shared__ float temp[];
	temp[2*thid]=(keys[2*thid]>keys[2*thid+1])?1:0;
   __syncthreads();
	temp[2*thid+1]=((2*thid+2)<n && keys[2*thid+1]>keys[2*thid+2])?1:0;

  for(int d=n>>1;d>0;d>>=1){
    __syncthreads();
    if(thid<d){
      int ai=offset*(2*thid+1)-1;
      int bi=offset*(2*thid+2)-1;
			temp[bi]+=temp[ai];
    }
    offset<<=1;
  }
  __syncthreads();
	int res=!temp[n-1];
  __syncthreads();
	temp[2*thid]=0;
	temp[2*thid+1]=0;
  __syncthreads();

	return res;
}

__device__ void segmented_prescan(float* keys,float* seg_flags,int n){
  int thid=threadIdx.x;
	extern __shared__ float shared[];
	float* f=(float*) &shared[0];
	float* data=(float*) &shared[n];

	int offset=1;

	f[2*thid]=seg_flags[2*thid];
	f[2*thid+1]=seg_flags[2*thid+1];
	data[2*thid]=keys[2*thid];
	data[2*thid+1]=keys[2*thid+1];

	for(int d=n>>1;d>0;d>>=1){
		__syncthreads();
		if(thid<d){
			int ai=offset*(2*thid+1)-1;
			int bi=offset*(2*thid+2)-1;
			if(!f[bi])
				data[bi]+=data[ai];
			f[bi]=f[bi] || f[ai];
		}
		offset<<=1;
	}
	__syncthreads();

	if(thid==0) data[n-1]=0;

	for(int d=1;d<n;d<<=1){
		offset>>=1;
		__syncthreads();
		if(thid<d){
			int ai=offset*(2*thid+1)-1;
			int bi=offset*(2*thid+2)-1;
			float t=data[ai];
			data[ai]=data[bi];
			if(f[ai+1])
				data[bi]=0;
			else if(f[ai])
				data[bi]=t;
			else
				data[bi]+=t;
			f[ai]=0;
		}
	}
	__syncthreads();

// little correction ;-)
	if(thid==(n/2-1) && !seg_flags[n-1])
		keys[n-1]=data[n-2]+keys[2*thid];
	else if(!seg_flags[2*thid+1])
		keys[2*thid+1]=data[2*thid+1];
	else
		keys[2*thid+1]=0;
	keys[2*thid]=data[2*thid];
}

__device__ void seg_copy(float *f,float *pivots,float *keys,float *seg_flags,unsigned int n)
{
	// from arguments only f and pivots modified
  int thid=threadIdx.x;
	pivots[2*thid]=(seg_flags[2*thid])?keys[2*thid]:0;
	pivots[2*thid+1]=(seg_flags[2*thid+1])?keys[2*thid+1]:0;
  __syncthreads();

	segmented_prescan(pivots,seg_flags,n);
  __syncthreads();
	if(seg_flags[2*thid]) pivots[2*thid]=keys[2*thid];
	if(seg_flags[2*thid+1]) pivots[2*thid+1]=keys[2*thid+1];

	float di=keys[2*thid]-pivots[2*thid];
	f[2*thid]=(di<0)?0:1;
	di=keys[2*thid+1]-pivots[2*thid+1];
	f[2*thid+1]=(di<0)?0:1;
}

__device__ void make_offsets(float* offsets,float* seg_flags,int n){
  int thid=threadIdx.x;
	offsets[2*thid]=(seg_flags[2*thid])?2*thid:0;
	offsets[2*thid+1]=(seg_flags[2*thid+1])?2*thid+1:0;
  __syncthreads();
	segmented_prescan(offsets,seg_flags,n);
  __syncthreads();
	if(seg_flags[2*thid]) offsets[2*thid]=2*thid;
	if(seg_flags[2*thid+1]) offsets[2*thid+1]=2*thid+1;
}

__device__ void make_idown(float *idown,float* seg_flags,float* f,int n){
  int thid=threadIdx.x;
	idown[2*thid]=(f[2*thid])?0:1;
	idown[2*thid+1]=(f[2*thid+1])?0:1;
 	__syncthreads();
	segmented_prescan(idown,seg_flags,n);
}

__device__ void make_iup1(float* iup_half,float* seg_flags,int n){
  int thid=threadIdx.x;
	iup_half[2*thid]=1;
	iup_half[2*thid+1]=1;
	__syncthreads();
	segmented_prescan(iup_half,seg_flags,n);
}

__device__ void make_iup2(float *iup,float* seg_flags,float* f,int n){
  int thid=threadIdx.x;

	// reverse and negate
	float temp=f[thid];
	f[thid]=(f[n-thid-1])?0:1;
	f[n-thid-1]=temp?0:1;
	if(thid>0){
		temp=seg_flags[thid];
		seg_flags[thid]=seg_flags[n-thid];
		seg_flags[n-thid]=temp;
	}

	__syncthreads();
	iup[2*thid]=f[2*thid];
	iup[2*thid+1]=f[2*thid+1];
	__syncthreads();
	segmented_prescan(iup,seg_flags,n);
 	__syncthreads();

	// reverse and negate
	temp=f[thid];
	f[thid]=(f[n-thid-1])?0:1;
	f[n-thid-1]=temp?0:1;
	if(thid>0){
		temp=seg_flags[thid];
		seg_flags[thid]=seg_flags[n-thid];
		seg_flags[n-thid]=temp;
	}

	__syncthreads();
	temp=iup[thid];
	iup[thid]=iup[n-1-thid];
	iup[n-1-thid]=temp;
	__syncthreads();
}

__device__ void make_iup(float *iup,float* iup_half,float* seg_flags,float* f,int n){
  int thid=threadIdx.x;
	iup_half[2*thid]=1;
	iup_half[2*thid+1]=1;
	__syncthreads();
	segmented_prescan(iup_half,seg_flags,n);
	__syncthreads();

	// reverse and negate
	float temp=f[thid];
	f[thid]=(f[n-thid-1])?0:1;
	f[n-thid-1]=temp?0:1;
	if(thid>0){
		temp=seg_flags[thid];
		seg_flags[thid]=seg_flags[n-thid];
		seg_flags[n-thid]=temp;
	}

	__syncthreads();
	iup[2*thid]=f[2*thid];
	iup[2*thid+1]=f[2*thid+1];
	__syncthreads();
	segmented_prescan(iup,seg_flags,n);
 	__syncthreads();

	// reverse and negate
	temp=f[thid];
	f[thid]=(f[n-thid-1])?0:1;
	f[n-thid-1]=temp?0:1;
	if(thid>0){
		temp=seg_flags[thid];
		seg_flags[thid]=seg_flags[n-thid];
		seg_flags[n-thid]=temp;
	}

	__syncthreads();
	temp=iup[thid];
	iup[thid]=iup[n-1-thid];
	iup[n-1-thid]=temp;
	__syncthreads();
	iup[2*thid]+=iup_half[2*thid];
	iup[2*thid+1]+=iup_half[2*thid+1];
}

__device__ void segmented_prescan2(float* keys,float* seg_flags,int n){
  int thid=threadIdx.x;
	extern __shared__ float shared[];
	float* f=(float*) &shared[0];
	float* data=(float*) &shared[n];

	int offset=1;

	f[2*thid]=seg_flags[2*thid];
	f[2*thid+1]=seg_flags[2*thid+1];
	data[2*thid]=keys[2*thid];
	data[2*thid+1]=keys[2*thid+1];

	for(int d=n>>1;d>0;d>>=1){
		__syncthreads();
		if(thid<d){
			int ai=offset*(2*thid+1)-1;
			int bi=offset*(2*thid+2)-1;
			if(!f[bi]) data[bi]+=data[ai];
			if(f[bi] || f[ai]) f[bi]=1;
			else f[bi]=0;
		}
		offset<<=1;
	}
	keys[2*thid]=data[2*thid];
	keys[2*thid+1]=data[2*thid+1];
return;
	if(thid==0) data[n-1]=0;	
	__syncthreads();

	for(int d=1;d<n;d<<=1){
		offset>>=1;
		__syncthreads();
		if(thid<d){
			int ai=offset*(2*thid+1)-1;
			int bi=offset*(2*thid+2)-1;
			float t=data[ai];
			data[ai]=data[bi];
			if(f[ai]) data[bi]=t;
			else data[bi]+=t;
			if(f[ai] || f[bi]) f[bi]=1;
			else f[bi]=0;
			__syncthreads();
		}
		break;
	}

	__syncthreads();
	keys[2*thid]=data[2*thid];
	if(thid==(n/2-1) && !seg_flags[n-1])
		keys[n-1]=data[n-2];
	else
		keys[2*thid+1]=data[2*thid+1];
	__syncthreads();
}*/

__device__ void segmented_scan_down(int* val,int* f,int* f2,int thread_elems_num){
	const int threads_num=blockDim.x; // number of threads in each block
  const int thid=threadIdx.x; // thread's number in given block
	
	int offset=threads_num;
	for(int d=1;d<=threads_num;d<<=1){
		__syncthreads();
		if(thid<d){
			int ai=offset*(2*thid+1)-1;
			int bi=offset*(2*thid+2)-1;
			int t=val[ai];
			val[ai]=val[bi];
			if((f2==NULL && f[ai+1]) || (f2!=NULL && f2[ai]))
				val[bi]=0;
			else if(f[ai])
				val[bi]=0+t;
			else
				val[bi]+=t;
			f[ai]=0;
		}
		offset>>=1;
	}
}

__device__ void segmented_scan_up(int* val,int* f,int thread_elems_num){
	const int threads_num=blockDim.x; // number of threads in each block
  const int thid=threadIdx.x; // thread's number in given block
	
	int offset=1;
	for(int d=threads_num;d>0;d>>=1){
		__syncthreads();
		if(thid<d){
			int ai=offset*(2*thid+1)-1;
			int bi=offset*(2*thid+2)-1;
			if(!f[bi]){
				val[bi]+=val[ai];
				f[bi]=f[ai];
			}
		}
		offset<<=1;
	}
}

__global__ void accumulate_sum_of_sums2(sum* g_sums, int thread_elems_num, int offset){
	const int threads_num=blockDim.x; // number of threads in each block
	const int thid=threadIdx.x; // thread's number in given block
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];
	int* f2=(int*)&absolute_shared[2*threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_sums[offset*(begin2+i+1)-1].seg_flag;
		f2[begin2+i]=g_sums[offset*(begin2+i+1)-1].next_seg_flag;
		val[begin2+i]=g_sums[offset*(begin2+i+1)-1].val;
	}
	if(thid==threads_num-1)
		val[threads_num*thread_elems_num-1]=0;
	__syncthreads();
	segmented_scan_down(val,f,f2,thread_elems_num);
	__syncthreads();
	for(int i=0;i<thread_elems_num;++i){
		g_sums[offset*(begin2+i+1)-1].seg_flag=f[begin2+i];
		g_sums[offset*(begin2+i+1)-1].val=val[begin2+i];
	}
}

__global__ void accumulate_sums2(sum* g_sums, int thread_elems_num,short one_block){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];
	int* f2=(int*)&absolute_shared[2*threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_sums[begin+i].seg_flag;
		f2[begin2+i]=g_sums[begin+i].next_seg_flag;
		val[begin2+i]=g_sums[begin+i].val;
	}
	if(thid==threads_num-1 && one_block)
		val[threads_num*thread_elems_num-1]=0;
	__syncthreads();

	__syncthreads();
	segmented_scan_down(val,f,f2,thread_elems_num);
	__syncthreads();

	for(int i=0;i<thread_elems_num;++i){
		g_sums[begin+i].seg_flag=f[begin2+i];
		g_sums[begin+i].val=val[begin2+i];
	}
}

__global__ void accumulate_sum_of_sums(sum* g_sums, int thread_elems_num, int offset){
	const int threads_num=blockDim.x; // number of threads in each block
	const int thid=threadIdx.x; // thread's number in given block
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_sums[offset*(begin2+i+1)-1].seg_flag;
		val[begin2+i]=g_sums[offset*(begin2+i+1)-1].val;
	}
	__syncthreads();
	segmented_scan_up(val,f,thread_elems_num);
	__syncthreads();
	for(int i=0;i<thread_elems_num;++i){
		g_sums[offset*(begin2+i+1)-1].seg_flag=f[begin2+i];
		g_sums[offset*(begin2+i+1)-1].val=val[begin2+i];
	}
}

__global__ void accumulate_sums(sum* g_sums, int thread_elems_num){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_sums[begin+i].seg_flag;
		val[begin2+i]=g_sums[begin+i].val;
	}
	__syncthreads();
	segmented_scan_up(val,f,thread_elems_num);
	__syncthreads();

	for(int i=0;i<thread_elems_num;++i){
		g_sums[begin+i].seg_flag=f[begin2+i];
		g_sums[begin+i].val=val[begin2+i];
	}
}

__global__ void make_idowns2(elem *g_elems, sum* g_sums, int thread_elems_num,int num_blocks2){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+i].seg_flag;
		val[begin2+i]=g_elems[begin+i].idown;
	}

	__syncthreads();
	// zrzut z sumy bloków:
	if(thid==0){
		f[n-1]=g_sums[bid].seg_flag;
		val[n-1]=g_sums[bid].val;
	}

	__syncthreads();
	segmented_scan_down(val,f,NULL,thread_elems_num);
	__syncthreads();

	for(int i=0;i<thread_elems_num;++i){
		if(g_elems[begin+i].seg_flag2)
			g_elems[begin+i].idown=0;
		else
			g_elems[begin+i].idown=val[begin2+i];
	}

	__syncthreads();
	if(thid==threads_num-1 && bid==num_blocks2-1 && !g_elems[begin+thread_elems_num-1].seg_flag2)
		g_elems[begin+thread_elems_num-1].idown=1-g_elems[begin+thread_elems_num-2].pivot+g_elems[begin+thread_elems_num-2].idown;
}

__global__ void make_iup2s2(elem *g_elems, sum* g_sums, int thread_elems_num,int num_blocks2,int num_blocks){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=(num_blocks-1-bid)*n+(threads_num-1-thid)*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+thread_elems_num-1-i].seg_flag;
		val[begin2+i]=g_elems[begin+thread_elems_num-1-i].iup2;
	}

	__syncthreads();
	// zrzut z sumy bloków:
	if(thid==0){
		f[n-1]=g_sums[bid].seg_flag;
		val[n-1]=g_sums[bid].val;
	}

	__syncthreads();
	segmented_scan_down(val,f,NULL,thread_elems_num);
	__syncthreads();

	for(int i=0;i<thread_elems_num;++i){
		g_elems[begin+thread_elems_num-1-i].iup2=val[begin2+i];
		g_elems[begin+thread_elems_num-1-i].seg_flag=f[begin2+i];
	}

	__syncthreads();
	if(thid==threads_num-1 && bid==num_blocks2-1)
		g_elems[0].iup2=1-g_elems[1].pivot+g_elems[1].iup2;
	if(thid==0 && bid==0)
		g_elems[num_blocks2*threads_num*thread_elems_num-1].iup2=0;
}

__global__ void make_iup1s2(elem *g_elems, sum* g_sums, int thread_elems_num,int num_blocks2){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+i].seg_flag;
		val[begin2+i]=g_elems[begin+i].iup1;
	}

	__syncthreads();
	// zrzut z sumy bloków:
	if(thid==0){
		f[n-1]=g_sums[bid].seg_flag;
		val[n-1]=g_sums[bid].val;
	}

	__syncthreads();
	segmented_scan_down(val,f,NULL,thread_elems_num);
	__syncthreads();

	for(int i=0;i<thread_elems_num;++i){
		if(g_elems[begin+i].seg_flag2)
			g_elems[begin+i].iup1=0;
		else
			g_elems[begin+i].iup1=val[begin2+i];
	}

	__syncthreads();
	if(thid==threads_num-1 && bid==num_blocks2-1 && !g_elems[begin+thread_elems_num-1].seg_flag2)
		g_elems[begin+thread_elems_num-1].iup1=1+g_elems[begin+thread_elems_num-2].iup1;
}

__global__ void make_offsets2(elem *g_elems, sum* g_sums, int thread_elems_num,int num_blocks2){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+i].seg_flag;
		val[begin2+i]=g_elems[begin+i].offset;
	}
	__syncthreads();

	__syncthreads();
	// zrzut z sumy bloków:
	if(thid==0){
		f[n-1]=g_sums[bid].seg_flag;
		val[n-1]=g_sums[bid].val;
	}

	__syncthreads();
	segmented_scan_down(val,f,NULL,thread_elems_num);
	__syncthreads();

	for(int i=0;i<thread_elems_num;++i){
		if(g_elems[begin+i].seg_flag2)
			g_elems[begin+i].offset=begin+i;
		else
			g_elems[begin+i].offset=val[begin2+i];
	}

	__syncthreads();
	if(thid==threads_num-1 && bid==num_blocks2-1 && !g_elems[begin+thread_elems_num-1].seg_flag2)
		g_elems[begin+thread_elems_num-1].offset=g_elems[begin+thread_elems_num-2].offset;
}

__global__ void make_pivots2(elem *g_elems, sum* g_sums, int thread_elems_num,int num_blocks2){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+i].seg_flag;
		val[begin2+i]=g_elems[begin+i].pivot;
	}
	__syncthreads();

	// zrzut z sumy bloków:
	if(thid==0){
		f[n-1]=g_sums[bid].seg_flag;
		val[n-1]=g_sums[bid].val;
	}

	__syncthreads();
	segmented_scan_down(val,f,NULL,thread_elems_num);
	__syncthreads();

	for(int i=0;i<thread_elems_num;++i){
		if(g_elems[begin+i].seg_flag2)
			g_elems[begin+i].pivot=1;
		else
			g_elems[begin+i].pivot=val[begin2+i]<=g_elems[begin+i].val;
	}

	__syncthreads();
	if(thid==threads_num-1 && bid==num_blocks2-1 && !g_elems[begin+thread_elems_num-1].seg_flag2)
		g_elems[begin+thread_elems_num-1].pivot=val[begin2+thread_elems_num-2]<=g_elems[begin+thread_elems_num-1].val;

	/* ze starego projektu:
	// little correction ;-)
	if(thid==(n/2-1) && !seg_flags[n-1])
		keys[n-1]=data[n-2]+keys[2*thid];
	else if(!seg_flags[2*thid+1])
		keys[2*thid+1]=data[2*thid+1];
	else
		keys[2*thid+1]=0;
	keys[2*thid]=data[2*thid];
	*/
}

__global__ void make_offsets(elem *g_elems, sum* g_sums, int thread_elems_num){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+i].seg_flag2;
		if(f[begin2+i])
			val[begin2+i]=begin+i;
		else
			val[begin2+i]=0;
	}
	__syncthreads();
	segmented_scan_up(val,f,thread_elems_num);
	__syncthreads();
	for(int i=0;i<thread_elems_num;++i){
		g_elems[begin+i].seg_flag=f[begin2+i];
		g_elems[begin+i].offset=val[begin2+i];
	}
	if(thid==0){	// do sprawdzenia!
		g_sums[bid].val=val[n-1];
		g_sums[bid].seg_flag=f[n-1];
		if(bid>0)
			g_sums[bid-1].next_seg_flag=f[0];
		else
			g_sums[0].next_seg_flag=0;
	}
}

__global__ void make_idowns(elem *g_elems, sum* g_sums, int thread_elems_num){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+i].seg_flag2;
		if(g_elems[begin+i].pivot)
			val[begin2+i]=0;
		else
			val[begin2+i]=1;
	}
	__syncthreads();
	segmented_scan_up(val,f,thread_elems_num);
	__syncthreads();
	for(int i=0;i<thread_elems_num;++i){
		g_elems[begin+i].pivot2=f[begin2+i];
		g_elems[begin+i].idown=val[begin2+i];
	}
	if(thid==0){	// do sprawdzenia!
		g_sums[bid].val=val[n-1];
		g_sums[bid].seg_flag=f[n-1];
		if(bid>0)
			g_sums[bid-1].next_seg_flag=f[0];
		else
			g_sums[0].next_seg_flag=0;
	}
}

__global__ void make_iup2s(elem *g_elems, sum* g_sums, int thread_elems_num, int num_blocks){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=(num_blocks-1-bid)*n+(threads_num-1-thid)*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+thread_elems_num-1-i].seg_flag2;
		val[begin2+i]=1-g_elems[begin+thread_elems_num-1-i].pivot;
	}
	__syncthreads();
	if(thid==0){
		if(bid==0)
			f[0]=1;
		if(bid==num_blocks-1)
			f[n-1]=0;
	}

	__syncthreads();
	segmented_scan_up(val,f,thread_elems_num);
	__syncthreads();

	for(int i=0;i<thread_elems_num;++i){
		g_elems[begin-i-1+thread_elems_num].seg_flag=f[begin2+i];
		g_elems[begin-i-1+thread_elems_num].iup2=val[begin2+i];
	}
	if(thid==0){	// do sprawdzenia!
		g_sums[bid].val=val[n-1];
		g_sums[bid].seg_flag=f[n-1];
		if(bid>0)
			g_sums[bid-1].next_seg_flag=f[0];
		else
			g_sums[0].next_seg_flag=0;
	}
}

__global__ void move_elems3(elem* g_elems,int thread_elems_num,int num_elements){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;

	for(int i=0;i<thread_elems_num;++i){
		g_elems[begin+i].val=g_elems[begin+i].pivot2;
		if(g_elems[begin+i].seg_flag && begin+i<num_elements-1)
			g_elems[begin+i+1].seg_flag2=1;
	}
}

__global__ void move_elems2(elem* g_elems,int thread_elems_num,int num_elements){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;

	for(int i=0;i<thread_elems_num;++i){
		g_elems[begin+i].val=g_elems[begin+i].pivot2;
		if(g_elems[begin+i].seg_flag)
			g_elems[begin+i].seg_flag2=1;
	}
}

__global__ void move_elems1(elem* g_elems,int thread_elems_num,int num_elements){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;

	for(int i=0;i<thread_elems_num;++i){
		int ind=g_elems[begin+i].offset+(g_elems[begin+i].pivot?g_elems[begin+i].iup1+g_elems[begin+i].iup2:g_elems[begin+i].idown);
		g_elems[ind].pivot2=g_elems[begin+i].val;
		g_elems[ind].seg_flag=g_elems[begin+i].seg_flag2;
	}
}

__global__ void make_iup1s(elem *g_elems, sum* g_sums, int thread_elems_num){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+i].seg_flag2;
		val[begin2+i]=1;
	}
	__syncthreads();
	segmented_scan_up(val,f,thread_elems_num);
	__syncthreads();
	for(int i=0;i<thread_elems_num;++i){
		g_elems[begin+i].seg_flag=f[begin2+i];
		g_elems[begin+i].iup1=val[begin2+i];
	}
	if(thid==0){	// do sprawdzenia!
		g_sums[bid].val=val[n-1];
		g_sums[bid].seg_flag=f[n-1];
		if(bid>0)
			g_sums[bid-1].next_seg_flag=f[0];
		else
			g_sums[0].next_seg_flag=0;
	}
}

// n_real - number of elements to be sorted
// n - number of elements to be sorted in each block
__global__ void make_pivots(elem *g_elems, sum* g_sums, int thread_elems_num){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int n=threads_num*thread_elems_num;
	const int begin=bid*n+thid*thread_elems_num;
	const int begin2=thid*thread_elems_num;

	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	int* val=(int*)&absolute_shared[threads_num*thread_elems_num];

	for(int i=0;i<thread_elems_num;++i){
		f[begin2+i]=g_elems[begin+i].seg_flag2;
		if(f[begin2+i])
			val[begin2+i]=g_elems[begin+i].val;
		else
			val[begin2+i]=0;
	}
	__syncthreads();
	segmented_scan_up(val,f,thread_elems_num);
	__syncthreads();
	for(int i=0;i<thread_elems_num;++i){
		g_elems[begin+i].seg_flag=f[begin2+i];
		g_elems[begin+i].pivot=val[begin2+i];
	}
	if(thid==0){	// do sprawdzenia!
		g_sums[bid].val=val[n-1];
		g_sums[bid].seg_flag=f[n-1];
		if(bid>0)
			g_sums[bid-1].next_seg_flag=f[0];
		else
			g_sums[0].next_seg_flag=0;
	}
}

// n_real - number of elements to be sorted
// n - number of elements to be sorted in each block
//__global__ void quicksort_kernel(elem *g_elems, int n_real,int n,int num_blocks){
__global__ void check_order(elem *g_elems, sum* g_sums, int n_real,int n,int num_blocks,int num_blocks2){
	const int threads_num=blockDim.x; // number of threads in each block
	const int bid=blockIdx.x; // given block's number
  const int thid=threadIdx.x; // thread's number in given block
	const int thread_elems_num=n/threads_num; // number of elements in ech thread
	const int begin=bid*n+thid*thread_elems_num;
	extern __shared__ int absolute_shared[];
	int* f=(int*)&absolute_shared[0];
	f[thid]=0;
	int predecessor=g_elems[begin].val;
	int current;
	for(int i=1;i<=thread_elems_num;++i){
		if(thid==threads_num-1 && i==thread_elems_num && bid==num_blocks-1)
			current=INT_MAX;
		else
			current=g_elems[begin+i].val;
		if(predecessor>current){
			f[thid]=1;
		}else
			predecessor=current;
	}

	int offset=1;
	for(int d=threads_num>>1;d>0;d>>=1){
		__syncthreads();
		if(thid<d){
			int ai=offset*(2*thid+1)-1;
			int bi=offset*(2*thid+2)-1;
			f[bi]|=f[ai];
		}
		offset<<=1;
	}

	__syncthreads();
	if(thid==0)
		g_sums[bid].val=f[threads_num-1];
}

//	g_elems[begin]=is_sorted2();

	/*
	int i,i1,i2,f1,f2;
	float temp1,temp2;
	extern __shared__ float absolute_shared[];
	float* shared=(float*) &absolute_shared[2*n];

	float* data=(float*) &shared[0];
	float* seg_flags=(float*) &shared[n];
	float* f=(float*) &shared[2*n];
	float* pivots=(float*) &shared[3*n];
	float* offsets=(float*) &shared[3*n]; // 3*n sic!
	float* idown=(float*) &shared[3*n];
	float* iup=(float*) &shared[3*n];
	float* iup_half=(float*) &shared[3*n];

	// data initialization
	if(2*thid>n-n_real-1){
		data[2*thid]=g_idata[2*thid-n+n_real];
		seg_flags[2*thid]=0;
	}else{
		data[2*thid]=0; // min from input
		seg_flags[2*thid]=1;
	}
	if(2*thid+1>n-n_real-1){
		data[2*thid+1]=g_idata[2*thid+1-n+n_real];
		seg_flags[2*thid+1]=0;
	}else{
		data[2*thid+1]=0;
		seg_flags[2*thid+1]=1;
	}
	// seg_flags initialization
	if(2*thid==n-n_real)
		seg_flags[2*thid]=1;
	else if(2*thid+1==n-n_real)
		seg_flags[2*thid+1]=1;
	__syncthreads();

	i=0;
	while(!is_sorted(data,n) && i<n){
		++i;
		__syncthreads();
		seg_copy(f,pivots,data,seg_flags,n);
		__syncthreads();
		make_offsets(offsets,seg_flags,n);
		__syncthreads();
		float offset1=offsets[2*thid];
		float offset2=offsets[2*thid+1];
		__syncthreads();
		make_idown(idown,seg_flags,f,n);
		float idown1=idown[2*thid];
		float idown2=idown[2*thid+1];
		__syncthreads();
		make_iup1(iup_half,seg_flags,n);
		__syncthreads();
		float iup1=iup_half[2*thid];
		float iup2=iup_half[2*thid+1];
		__syncthreads();
		make_iup2(iup,seg_flags,f,n);
		__syncthreads();
		iup1+=iup[2*thid];
		iup2+=iup[2*thid+1];
		__syncthreads();
		temp1=data[2*thid];
		temp2=data[2*thid+1];
		i1=(f[2*thid])?iup1:idown1;
		i1+=offset1;
		i2=(f[2*thid+1])?iup2:idown2;
		i2+=offset2;
		
		__syncthreads();
		data[i1]=temp1;
		data[i2]=temp2;
		__syncthreads();
		f1=seg_flags[2*thid];
		f2=seg_flags[2*thid+1];
		__syncthreads();
		if(!seg_flags[i1] && f1) seg_flags[i1]=1;
		if(!seg_flags[i2] && f2) seg_flags[i2]=1;
		__syncthreads();
		if(f1 && i1+1<n && !seg_flags[i1+1]) seg_flags[i1+1]=1;
		if(f2 && i2+1<n && !seg_flags[i2+1]) seg_flags[i2+1]=1;
		__syncthreads();
		
	}

	if(2*thid>n-n_real-1)
		g_odata[2*thid-n+n_real]=data[2*thid];
	if(2*thid+1>n-n_real-1)
		g_odata[2*thid+1-n+n_real]=data[2*thid+1];

//	g_odata[2*thid]=data[2*thid];
//	g_odata[2*thid+1]=data[2*thid+1];
*/

#endif // #ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
