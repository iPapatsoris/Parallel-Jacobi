#include "util.h"
#include <stdio.h>
#include <math.h>

void divide2D(int total, int *rows, int *columns) {
	*columns = ceil(sqrt(total));
	*rows = ceil((float) total / *columns);
}

void parseInput(struct JacobiParams *jacobiParams) {
    scanf("%d,%d", &jacobiParams->inputRows, &jacobiParams->inputColumns);
	int checkConvergence;
    scanf("%d", &checkConvergence);
	jacobiParams->checkConvergence = checkConvergence;
    scanf("%lf", &jacobiParams->alpha);
    scanf("%lf", &jacobiParams->relax);
    scanf("%lf", &jacobiParams->tol);
    scanf("%d", &jacobiParams->maxIterations);
}

void initArray(double *array, int size) {
	for (int i = 0 ; i < size ; i++) {
		array[i] =  0;
	}	
}

void swap(double **a, double **b) {
	double *tmp = *a;
	*a = *b;
	*b = tmp;
}

JacobiConstants getJacobiConstants(int xLeft, int xRight, int yBottom, int yTop, int inputRows, int inputColumns, double alpha) {
	JacobiConstants jacobiConstants;
	const double deltaX = (double) (xRight - xLeft) / (inputColumns - 1);
	const double deltaY = (double) (yTop - yBottom) / (inputRows - 1);
	const double cx = 1.0/(deltaX*deltaX);
	const double cy = 1.0/(deltaY*deltaY);
	const double cc = -2.0*cx-2.0*cy-alpha;

	jacobiConstants.deltaX = deltaX;
	jacobiConstants.deltaY = deltaY;
	jacobiConstants.cc = cc;
	jacobiConstants.cy = cy;
	jacobiConstants.cx = cx;
	
	return jacobiConstants;
}