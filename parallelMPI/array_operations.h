#ifndef ARRAY_OPERATIONS
#define ARRAY_OPERATIONS

#include "mpi.h"

void initArrays(double *array, double *newArray, const int lines, const int columns);
void printArrays(const double *array, const double *newArray, const int lines, const int columns, const int processID, const int totalProcesses, MPI_Comm cartesianComm);
__attribute__((always_inline)) inline int at(const int i, const int j, const int columns) {
    return i * columns + j;
}
void printArray(const double *array, const int lines, const int columns);

#endif