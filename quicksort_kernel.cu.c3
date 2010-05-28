/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
#define _SCAN_WORKEFFICIENT_KERNEL_H_

///////////////////////////////////////////////////////////////////////////////
//! Work-efficient compute implementation of scan, one thread per 2 elements
//! Work-efficient: O(log(n)) steps, and O(n) adds.
//! Also shared storage efficient: Uses n elements in shared mem -- no ping-ponging
//! Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
//! and Their Applications", or Prins and Chatterjee PRAM course notes:
//! https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
//!
//! Pro: Work Efficient
//! Con: Shared memory bank conflicts due to the addressing used.
//
//! @param g_odata  output data in global memory
//! @param g_idata  input data in global memory
//! @param n        input number of elements to scan from input data
///////////////////////////////////////////////////////////////////////////////

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

__device__ void seg_copy(float *f,float *keys,float *aseg_flags,float *pivots,unsigned int n)
{
// only pivots modified (exactly set)
  extern  __shared__ int seg_flags[];
	float last_pivot;

  int thid=threadIdx.x;
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];
///*	
  int offset=1;
	if(seg_flags[2*thid]) pivots[2*thid]=keys[2*thid];
	else pivots[2*thid]=0;
	if(seg_flags[2*thid+1]) pivots[2*thid+1]=keys[2*thid+1];
	else pivots[2*thid+1]=0;
//*/
///*
  for(int d=n>>1;d>=1;d>>=1){
    __syncthreads();
    if(thid<d){
      int ai=offset*(2*thid+1)-1;
      int bi=offset*(2*thid+2)-1;
      if(!seg_flags[bi])
        pivots[bi]+=pivots[ai];
			if(seg_flags[ai] || seg_flags[bi]) seg_flags[bi]=1;
			else seg_flags[bi]=0;
			
//      seg_flags[bi]=seg_flags[bi]|seg_flags[ai]; // nie rozumiem tego...
    } 
    offset<<=1;
  }

  __syncthreads();
// tego nie ma na slajdach!!!
//	seg_flags[2*thid]=aseg_flags[2*thid];
//	seg_flags[2*thid+1]=aseg_flags[2*thid+1];

  __syncthreads();
  if((thid+1)*2==n){
		last_pivot=pivots[n-1];
    pivots[n-1]=0;
	}

  for(int d=1;d<n;d<<=1){
    offset>>=1;
    __syncthreads();
	  if(thid<d){
	    int ai=offset*(2*thid+1)-1;
	    int bi=offset*(2*thid+2)-1;
	    float t=pivots[ai];
 	  	pivots[ai]=pivots[bi];
 	  	if(seg_flags[ai])
 	    	pivots[bi]=t;
 	   	else
 	    	pivots[bi]+=t;
			if(seg_flags[ai] || seg_flags[bi]) seg_flags[bi]=1;
			else seg_flags[bi]=0;
    	__syncthreads();
//	    seg_flags[ai]=0; // tego tez nie rozumiem
	  }
  }
return;
	aseg_flags[2*thid]=seg_flags[2*thid];
	aseg_flags[2*thid+1]=seg_flags[2*thid+1];
//*/
///*

  __syncthreads();
	if(2*(thid+1)==n)
		pivots[2*thid+1]=last_pivot;
	else if(aseg_flags[2*thid+1])
		pivots[2*thid+1]=keys[2*thid+1];
	if(aseg_flags[2*thid])
		pivots[2*thid]=keys[2*thid];
  __syncthreads();
return;
//*/

//	f:=pivots<=>keys
///*
	float di=keys[2*thid]-pivots[2*thid];
	if(di<0) f[2*thid]=0;
	else f[2*thid]=1;
	di=keys[2*thid+1]-pivots[2*thid+1];
	if(di<0) f[2*thid+1]=0;
	else f[2*thid+1]=1;
//*/
//	pivots[2*thid]=2;
//	pivots[2*thid+1]=3;
//  __syncthreads();
}
/*
__device__ void make_offsets(float* offsets,float* aseg_flags,int n){
  int thid=threadIdx.x;
  extern  __shared__ int seg_flags[];
	offsets[2*thid]=(aseg_flags[2*thid])?2*thid:0;
	offsets[2*thid+1]=(aseg_flags[2*thid+1])?2*thid+1:0;
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];
  int offset=1;

//
  for(int d=n>>1;d>0;d>>=1){
    __syncthreads();
    if(thid<d){
      int ai=offset*(2*thid+1)-1;
      int bi=offset*(2*thid+2)-1;
      if(!seg_flags[bi])
        offsets[bi]+=offsets[ai];
      seg_flags[bi]=seg_flags[bi]|seg_flags[ai]; // nie rozumiem tego...
    }
    offset<<=1;
  }

  __syncthreads();
// tego nie ma na slajdach!!!
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];

  __syncthreads();
  if((thid+1)*2==n)
    offsets[n-1]=0;

  for(int d=1;d<n;d<<=1){
    offset>>=1;
    __syncthreads();
	  if(thid<d){
	    int ai=offset*(2*thid+1)-1;
	    int bi=offset*(2*thid+2)-1;
	    float t=offsets[ai];
 	  	offsets[ai]=offsets[bi];
 	  	if(seg_flags[ai+1])
 	    	offsets[bi]=0;
 	   	else if(seg_flags[ai])
 	    	offsets[bi]=t;
 	   	else
 	    	offsets[bi]+=t;
	
    	__syncthreads();
	    seg_flags[ai]=0; // tego tez nie rozumiem
	  }
  }
//
//
  __syncthreads();
	if(aseg_flags[2*thid+1])
		offsets[2*thid+1]=2*thid+1;
	if(aseg_flags[2*thid])
		offsets[2*thid]=2*thid;
  __syncthreads();
//
}
*/
__device__ void make_idown(float *idown,float* aseg_flags,float* f,int n){
  int thid=threadIdx.x;
  extern  __shared__ int seg_flags[];
	idown[2*thid]=(f[2*thid])?0:1;
	idown[2*thid+1]=(f[2*thid+1])?0:1;
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];
///*	
  int offset=1;
//*/
///*
  for(int d=n>>1;d>0;d>>=1){
    __syncthreads();
    if(thid<d){
      int ai=offset*(2*thid+1)-1;
      int bi=offset*(2*thid+2)-1;
      if(!seg_flags[bi])
        idown[bi]+=idown[ai];
      seg_flags[bi]=seg_flags[bi]|seg_flags[ai]; // nie rozumiem tego...
    }
    offset<<=1;
  }

  __syncthreads();
// tego nie ma na slajdach!!!
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];

  __syncthreads();
  if((thid+1)*2==n)
    idown[n-1]=0;

  for(int d=1;d<n;d<<=1){
    offset>>=1;
    __syncthreads();
	  if(thid<d){
	    int ai=offset*(2*thid+1)-1;
	    int bi=offset*(2*thid+2)-1;
	    float t=idown[ai];
 	  	idown[ai]=idown[bi];
 	  	if(seg_flags[ai+1])
 	    	idown[bi]=0;
 	   	else if(seg_flags[ai])
 	    	idown[bi]=t;
 	   	else
 	    	idown[bi]+=t;
	
    	__syncthreads();
	    seg_flags[ai]=0; // tego tez nie rozumiem
	  }
  }
	if(aseg_flags[2*thid]) idown[2*thid]=0;
	if(aseg_flags[2*thid+1]) idown[2*thid+1]=0;
 	__syncthreads();
}

__device__ void make_iup(float *iup,float* aseg_flags,float* f,int n){
  int thid=threadIdx.x;
  extern  __shared__ int seg_flags[];
	float* iup2=(float*) &seg_flags[7*n];
	iup[2*thid]=1;
	iup[2*thid+1]=1;
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];
	__syncthreads();
///*	
  int offset=1;
//*/
///*
  for(int d=n>>1;d>0;d>>=1){
    __syncthreads();
    if(thid<d){
      int ai=offset*(2*thid+1)-1;
      int bi=offset*(2*thid+2)-1;
      if(!seg_flags[bi])
        iup[bi]+=iup[ai];
      seg_flags[bi]=seg_flags[bi]|seg_flags[ai]; // nie rozumiem tego...
    }
    offset<<=1;
  }

  __syncthreads();
// tego nie ma na slajdach!!!
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];

  __syncthreads();
  if((thid+1)*2==n)
    iup[n-1]=0;

  for(int d=1;d<n;d<<=1){
    offset>>=1;
    __syncthreads();
	  if(thid<d){
	    int ai=offset*(2*thid+1)-1;
	    int bi=offset*(2*thid+2)-1;
	    float t=iup[ai];
 	  	iup[ai]=iup[bi];
 	  	if(seg_flags[ai+1])
 	    	iup[bi]=0;
 	   	else if(seg_flags[ai])
 	    	iup[bi]=t;
 	   	else
 	    	iup[bi]+=t;
	
    	__syncthreads();
	    seg_flags[ai]=0; // tego tez nie rozumiem
	  }
  }
	if(aseg_flags[2*thid]) iup[2*thid]=0;
	if(aseg_flags[2*thid+1]) iup[2*thid+1]=0;
 	__syncthreads();

	iup2[2*thid]=iup[2*thid];
	iup2[2*thid+1]=iup[2*thid+1];

	// reverse and negate
	float temp=f[thid];
	f[thid]=(f[n-thid-1])?0:1;
	f[n-thid-1]=temp?0:1;
	if(thid>0){
		temp=aseg_flags[thid];
		aseg_flags[thid]=aseg_flags[n-thid];
		aseg_flags[n-thid]=temp;
	}

	__syncthreads();

	iup[2*thid]=f[2*thid];
	iup[2*thid+1]=f[2*thid+1];
	__syncthreads();
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];
	__syncthreads();
//	iup[2*thid]=seg_flags[2*thid];
//	iup[2*thid+1]=seg_flags[2*thid+1];
///*	
  offset=1;
//*/
///*
  for(int d=n>>1;d>0;d>>=1){
    __syncthreads();
    if(thid<d){
      int ai=offset*(2*thid+1)-1;
      int bi=offset*(2*thid+2)-1;
      if(!seg_flags[bi])
        iup[bi]+=iup[ai];
      seg_flags[bi]=seg_flags[bi]|seg_flags[ai]; // nie rozumiem tego...
    }
    offset<<=1;
  }

  __syncthreads();
// tego nie ma na slajdach!!!
	seg_flags[2*thid]=aseg_flags[2*thid];
	seg_flags[2*thid+1]=aseg_flags[2*thid+1];

  __syncthreads();
  if((thid+1)*2==n)
    iup[n-1]=0;

  for(int d=1;d<n;d<<=1){
    offset>>=1;
    __syncthreads();
	  if(thid<d){
	    int ai=offset*(2*thid+1)-1;
	    int bi=offset*(2*thid+2)-1;
	    float t=iup[ai];
 	  	iup[ai]=iup[bi];
 	  	if(seg_flags[ai+1])
 	    	iup[bi]=0;
 	   	else if(seg_flags[ai])
 	    	iup[bi]=t;
 	   	else
 	    	iup[bi]+=t;
	
    	__syncthreads();
	    seg_flags[ai]=0; // tego tez nie rozumiem
	  }
  }
	if(aseg_flags[2*thid]) iup[2*thid]=0;
	if(aseg_flags[2*thid+1]) iup[2*thid+1]=0;
 	__syncthreads();
	temp=iup[thid];
	iup[thid]=iup[n-1-thid];
	iup[n-1-thid]=temp;
 	__syncthreads();

	// reverse and negate
	temp=f[thid];
	f[thid]=(f[n-thid-1])?0:1;
	f[n-thid-1]=temp?0:1;
	if(thid>0){
		temp=aseg_flags[thid];
		aseg_flags[thid]=aseg_flags[n-thid];
		aseg_flags[n-thid]=temp;
	}

	__syncthreads();
	iup[2*thid]+=iup2[2*thid];
	iup[2*thid+1]+=iup2[2*thid+1];
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
			if(!f[bi]) data[bi]+=data[ai];
			if(f[bi] || f[ai]) f[bi]=1;
			else f[bi]=0;
		}
		offset<<=1;
	}
	__syncthreads();
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
	keys[2*thid+1]=data[2*thid+1];
	__syncthreads();
}

__global__ void quicksort_kernel(float *g_odata, float *g_idata, int n)
{
  int thid=threadIdx.x;
	extern __shared__ float shared[];
//	float* f=(float*) &shared[10*n];
	float* seg_flags=(float*) &shared[2*n];
//	float* pivots=(float*) &seg_flags[n];
//	float* offsets=(float*) &pivots[n];
//	float* idown=(float*) &offsets[n];
//	float* iup=(float*) &idown[n];
//	int* odata_tmp=(int*) &shared[8*n];
//	float* data_tmp=(float*) &shared[9*n];
g_odata[2*thid]=g_idata[2*thid];
g_odata[2*thid+1]=g_idata[2*thid+1];
	// seg_flags initialization
	if(thid==0){
		seg_flags[2*thid]=1;
	}
	else{
		seg_flags[2*thid]=0;
		g_odata[2*thid]=0;
	}
	seg_flags[2*thid+1]=0;
	g_odata[2*thid+1]=0;
	__syncthreads();

//	g_odata[2*thid+1]=is_sorted(g_idata,n);
//	g_odata[2*thid]=is_sorted(g_idata,n);
	
///*
segmented_prescan(g_odata,seg_flags,n);
/*
	data_tmp[2*thid]=g_idata[2*thid];
	data_tmp[2*thid+1]=g_idata[2*thid+1];
	int i=0;
	__syncthreads();
	while(!is_sorted(data_tmp,n) && i<1){
		++i;
	__syncthreads();
	seg_copy(f,data_tmp,seg_flags,pivots,n);
	g_odata[2*thid]=pivots[2*thid];
	g_odata[2*thid+1]=pivots[2*thid+1];
	return;
	make_offsets(offsets,seg_flags,n);
	make_idown(idown,seg_flags,f,n);
	make_iup(iup,seg_flags,f,n);
	odata_tmp[2*thid]=(f[2*thid])?iup[2*thid]:idown[2*thid];
	odata_tmp[2*thid+1]=(f[2*thid+1])?iup[2*thid+1]:idown[2*thid+1];
	odata_tmp[2*thid]+=offsets[2*thid];
	odata_tmp[2*thid+1]+=offsets[2*thid+1];
	__syncthreads();
	int has_been_set1,has_been_set2;
	if(seg_flags[2*thid]){
		has_been_set1=1;
	}else has_been_set1=0;
	if(seg_flags[2*thid+1]){
		has_been_set2=1;
	}else has_been_set2=0;
	__syncthreads();
	if(has_been_set1 && odata_tmp[2*thid]<n-1){
		seg_flags[odata_tmp[2*thid]+1]=1;
		seg_flags[odata_tmp[2*thid]]=1;
	}
	if(has_been_set2 && odata_tmp[2*thid+1]<n-1){
		seg_flags[odata_tmp[2*thid+1]]=1;
		seg_flags[odata_tmp[2*thid+1]+1]=1;
	}
	__syncthreads();
	float t1=data_tmp[2*thid];
	float t2=data_tmp[2*thid+1];
	__syncthreads();
	data_tmp[odata_tmp[2*thid]]=t1;
	data_tmp[odata_tmp[2*thid+1]]=t2;
//	g_idata[2*thid]=data_tmp[2*thid];
//	g_idata[2*thid+1]=data_tmp[2*thid+1];
	__syncthreads();
	}
//	if(thid==0) g_odata[0]=i;
	g_odata[2*thid]=data_tmp[2*thid];
	g_odata[2*thid+1]=data_tmp[2*thid+1];
	__syncthreads();

//
*/
}

#endif // #ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
