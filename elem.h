#ifndef _ELEM_TO_PBQUICKSORT
#define _ELEM_TO_PBQUICKSORT
typedef struct{
	int val;
	int at_place;
} elem;

typedef struct{
	int val;
} sum;

typedef struct{
	int n;
	elem* elems;
	sum* sums;
} tab;

tab* make_tab(int);

void free_tab(tab*);
#endif
