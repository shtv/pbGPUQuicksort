#ifndef _ELEM_TO_PBQUICKSORT
#define _ELEM_TO_PBQUICKSORT
typedef struct{
	int val;
	int pivot; // pivot, next flag
	int pivot2; // copy of pivot to be changed by idown generator
	int offset;
	int idown;
	int iup1;
	int iup2;
	short seg_flag;
	short seg_flag2;
} elem;

typedef struct{
	int val;
	short seg_flag;
	short next_seg_flag;
} sum;

typedef struct{
	int n;
	elem* elems;
	sum* sums;
} tab;

tab* make_tab(int);

void free_tab(tab*);
#endif
