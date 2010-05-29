#include <stdlib.h>

#include "elem.h"

tab* make_tab(int n){
	tab* table=(tab*)malloc(sizeof(tab));
	table->elems=(elem*)malloc(sizeof(elem)*n);
	table->n=n;
	return table;
}

void free_tab(tab* table){
	if(table==NULL) return;
	if(table->elems!=NULL)
		free(table->elems);
	free(table);
}
