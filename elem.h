#ifndef _ELEM_TO_PBQUICKSORT
#define _ELEM_TO_PBQUICKSORT
typedef struct{
	int val;
	int good_successor;
} elem;

typedef struct{
	int n;
	elem* elems;
} tab;

tab* make_tab(int a);

void free_tab(tab* a);
#endif
