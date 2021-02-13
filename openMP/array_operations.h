#ifndef ARRAY_OPERATIONS
#define ARRAY_OPERATIONS

#include "mpi.h"
#include "omp.h"
#include <stdlib.h>
#include "util.h"
#include <stdio.h>

void initArrays(double *array, double *newArray, const int lines, const int columns);
void printArrays(const double *array, const double *newArray, const int lines, const int columns, const int processID, const int totalProcesses, MPI_Comm cartesianComm);
__attribute__((always_inline)) inline int at(const int i, const int j, const int columns) {
    return i * columns + j;
}
void printArray(const double *array, const int lines, const int columns);


__attribute__((always_inline)) inline void calculateOneElement(const int y, const int x, const double *array, double *newArray, 
						const int rows, const int columns, const struct JacobiParams *jacobiParams,
						const double yStart, const double xStart, const double deltaY, const double deltaX, 
						const double cy, const double cx, const double cc, double *error,
						const int yIncrement, const int xIncrement) {
	
	int globalY = y + yIncrement;
	int globalX = x + xIncrement;

	double fY = yStart + (globalY-1)*deltaY;
	double fYSquare = fY*fY;
	double fX = xStart + (globalX-1)*deltaX;
	double fXSquare = fX*fX;

	double f = -jacobiParams->alpha*(1.0-fXSquare)*(1.0-fYSquare) - 2.0*(1.0-fXSquare) - 2.0*(1.0-fYSquare);
	double curVal = array[at(y, x, columns)];
	double updateVal = ((array[at(y, x-1, columns)] + array[at(y, x+1, columns)])*cx +
					(array[at(y-1, x, columns)] + array[at(y+1, x, columns)])*cy +
					curVal*cc - f
				)/cc;
	newArray[at(y,x, columns)] = curVal - jacobiParams->relax*updateVal;
	
	#pragma omp atomic
	(*error) += updateVal*updateVal;
}

__attribute__((always_inline)) inline void calculateInnerElements(const int threads, const double *array, double *newArray, 
						const int rows, const int columns, const struct JacobiParams *jacobiParams,
						const double yStart, const double xStart, const double deltaY, const double deltaX, 
						const double cy, const double cx, const double cc, double *error, const int yIncrement, const int xIncrement) {
	
	double fY, fX, fYSquare, fXSquare, f, curVal, updateVal;

	#pragma omp for schedule(static) collapse(2)
	for (int y = 2 ; y < rows - 2 ; y++) {
		for (int x = 2 ; x < columns - 2 ; x++) {
			calculateOneElement(y, x, array, newArray, rows, columns, jacobiParams, yStart, xStart, deltaY, deltaX, cy, cx, cc, error, yIncrement, xIncrement);
		}
	}
}

#endif