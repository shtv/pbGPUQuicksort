#ifndef _ELEM_TO_PBQUICKSORT
#define _ELEM_TO_PBQUICKSORT
typedef struct{
	int val;
	int at_place;
	int pivot; // pivot, next flag
	int offset;
	short seg_flag;
	short seg_flag2;
	short f;
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
