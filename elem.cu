#include <stdlib.h>

#include "elem.h"

tab* make_tab(int n,int n2){
	tab* table=(tab*)malloc(sizeof(tab));
	table->elems=(elem*)malloc(sizeof(elem)*n);
	table->sums=(sum*)malloc(sizeof(sum)*n2);
	table->n=n;
	return table;
}

void free_tab(tab* table){
	if(table==NULL) return;
	if(table->elems!=NULL)
		free(table->elems);
	if(table->sums!=NULL)
		free(table->sums);
	free(table);
}
