#include "array_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "util.h"

void initArrays(double *array, double *newArray, const int rows, const int columns) {
    for (int i = 0 ; i < rows ; i++) {
        for (int j = 0 ; j < columns ; j++) {
            array[at(i,j, columns)] = 0;
            newArray[at(i,j, columns)] = 0;
        }
    }
}

void printArray(const double *array, const int rows, const int columns) {
    printf("%d %d\n", rows, columns);
    for (int i = 0 ; i < rows ; i++) {
        for (int j = 0 ; j < columns ; j++) {
            printf("%d", array[at(i,j, columns)]);
        }
        printf("\n");
    }
}

void printArrays(const double *array, const double *newArray, const int rows, const int columns, const int processID, const int totalProcesses, MPI_Comm cartesianComm) {
    for(int i = 0; i < totalProcesses; i++) {
        MPI_Barrier(cartesianComm);
        if (i == processID) {
            printf("\nProcess %d old\n", processID);
            printArray(array, rows, columns);
            printf("Process %d new\n", processID);
            printArray(newArray, rows, columns);
            printf("\n");
        }
    }
}