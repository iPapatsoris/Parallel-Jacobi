#include "util.h"
#include <stdio.h>
#include <math.h>

void divide2D(int total, int *rows, int *columns) {
	int r = (int) sqrt(total);
	const bool perfectSquare = (r * r == total);
	*rows = r;
	*columns = (perfectSquare ? r : total / r);
}


void parseInput(struct JacobiParams *jacobiParams) {
//    printf("Input n,m - grid dimension in x,y direction:\n");
    scanf("%d,%d", &jacobiParams->inputRows, &jacobiParams->inputColumns);
//    printf("Input checkConvergence - check if we are within error tolerance on each iteration:\n");
	int checkConvergence;
    scanf("%d", &checkConvergence);
	jacobiParams->checkConvergence = checkConvergence;
//    printf("Input alpha - Helmholtz constant:\n");
    scanf("%lf", &jacobiParams->alpha);
//    printf("Input relax - successive over-relaxation parameter:\n");
    scanf("%lf", &jacobiParams->relax);
//    printf("Input tol - error tolerance for the iterrative solver:\n");
    scanf("%lf", &jacobiParams->tol);
//    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", &jacobiParams->maxIterations);
}

void initArray(double *array, int size) {
	for (int i = 0 ; i < size ; i++) {
		array[i] =  i;
	}	
}