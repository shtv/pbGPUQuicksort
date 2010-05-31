/*
 * Copyright 2010 Pawel Baran.
 *
 */

#ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
#define _SCAN_WORKEFFICIENT_KERNEL_H_

#include "elem.h"

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
/*
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

__device__ int is_sorted2(int1* elems,int n,int thid,int bid,int threads,int* f,elem* g_elems){
	int sorted_thread=1;
	for(int i=0;i<n;++i){
		if(elems[i].x>elems[i+1].x){
			sorted_thread=0;
			break;
		}
	}

	f[thid]=sorted_thread;

	int offset=1;
	for(int d=threads>>1;d>0;d>>=1){
		__syncthreads();
		if(thid<d){
			int ai=offset*(2*thid+1)-1;
			int bi=offset*(2*thid+2)-1;
			f[bi]|=f[ai];
		}
		offset<<=1;
	}

	if(thid==0){ // po jednym wątku z każdego bloku, do zapisu danych na blok sumy
//		g_elems[bid*threads].
	}
}


// n_real - number of elements to be sorted
// n - number of elements to be sorted in each block
//__global__ void quicksort_kernel(elem *g_elems, int n_real,int n,int num_blocks){
__global__ void check_order2(elem *g_elems, sum* g_sums, int n_real,int n,int num_blocks,int num_blocks2){

}

// n_real - number of elements to be sorted
// n - number of elements to be sorted in each block
//__global__ void quicksort_kernel(elem *g_elems, int n_real,int n,int num_blocks){
__global__ void check_order(elem *g_elems, sum* g_sums, int n_real,int n,int num_blocks,int num_blocks2){
	const int threads=blockDim.x; // number of threads in each block
	int bid=blockIdx.x; // given block's number
  int thid=threadIdx.x; // thread's number in given block
	int thread_elems=n/threads; // number of elements in ech thread
	int begin=bid*n+thid*thread_elems;

	extern __shared__ float absolute_shared[];
	int* f=(int*)&absolute_shared[0];

	int1 elems[thread_elems+1];
	// przepisanie wartości z globalnej do rejestrów
	for(int i=0;i<thread_elems;++i){
		elems[i].x=g_elems[begin+i].val;
	}
	if(bid==num_blocks-1 && thid==threads-1)
		elems[thread_elems].x=INT_MAX;
	else
		elems[thread_elems].x=g_elems[begin+thread_elems].val;

//	__syncthreads();
	int sorted=is_sorted2(elems,thread_elems,thid,bid,threads,f,g_elems);
	if(thid==0)
		g_sums[bid].val=sorted;

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
}

#endif // #ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
