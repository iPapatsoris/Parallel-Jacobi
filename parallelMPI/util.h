#ifndef UTIL
#define UTIL

#include "mpi.h"
#include <stdbool.h>

enum {
    NORTH, SOUTH, EAST, WEST, NORTHEAST, SOUTHEAST, NORTHWEST, SOUTHWEST
};

struct Neighbours {
    int north, south, east, west;
};

struct JacobiParams {
	int maxIterations, inputRows, inputColumns;
	bool checkConvergence;
	double alpha, relax, tol;
};

struct Neighbours constructNeighbours(const int processID, const int columnsOfProcesses, const int totalProcesses);
void printNeighbours(const struct Neighbours n, const int processID);
void parseInput(struct JacobiParams *jacobiParams);
// void performParallelIO(char *inputFile, const int totalLines, const int totalColumns, const int processLines, const int columnsOfProcesses, const int lines, const int columns,
//                       MPI_Datatype *inputHandle, MPI_Datatype *outputHandle, MPI_Datatype *pixelDatatype, MPI_Datatype *memtype, MPI_Datatype *filetype);

__attribute__((always_inline)) inline void reverseDirection(double **array, double **newArray, MPI_Request **sendRequests, MPI_Request **recvRequests, MPI_Request *sendStraightRequests, MPI_Request *sendReverseRequests, MPI_Request *recvStraightRequests, MPI_Request *recvReverseRequests) {
    double *tmp = *array;
    *array = *newArray;
    *newArray = tmp;
    if (*sendRequests == sendStraightRequests) {
        *sendRequests = sendReverseRequests;
        *recvRequests = recvReverseRequests;
    } else {
        *sendRequests = sendStraightRequests;
        *recvRequests = recvStraightRequests;
    }
}

__attribute__((always_inline)) inline void resetCompletedOperations(int *completedOperations, int *remainingOperations) {
    completedOperations[NORTH] = 0; // Manual assignments instead of looping for performance
    completedOperations[SOUTH] = 0;
    completedOperations[EAST] = 0;
    completedOperations[WEST] = 0;
    completedOperations[NORTHEAST] = 0;
    completedOperations[NORTHWEST] = 0;
    completedOperations[SOUTHEAST] = 0;
    completedOperations[SOUTHWEST] = 0;
    *remainingOperations = 8;
}

#endif