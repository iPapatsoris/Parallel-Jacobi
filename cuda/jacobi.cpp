#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "util.h"

extern "C" float jacobiGPU(double *, int, int, int, const JacobiParams );

int main(int argc, char **argv) {
	struct JacobiParams jacobiParams;
	parseInput(&jacobiParams);
	int inputRows = jacobiParams.inputRows;
	int inputColumns = jacobiParams.inputColumns;
	int elements = inputRows * inputColumns;

	double *array = (double *) malloc(elements * sizeof(double));
	if (array == NULL) {
		fprintf(stderr, "CPU array allocation error\n");
		return -1;
	}

	initArray(array, elements);
	float msecs = jacobiGPU(array, elements, inputRows, inputColumns, jacobiParams);
    printf("Execution time %.2f msecs\n", msecs);

	free(array);
}