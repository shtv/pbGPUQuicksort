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
		printf("check sums !\n");
	if(table->sums!=NULL){
		printf("sums isnt null\n");
		free(table->sums);
	}
	if(table->elems!=NULL){
		printf("elems isnt null\n");
		free(table->elems);
	}
	free(table);
}
