#ifndef UTIL
#define UTIL

#include "mpi.h"
#include "omp.h"
#include <stdbool.h>

enum {
    NORTH, SOUTH, EAST, WEST
};

enum {
	NORTHEAST, SOUTHEAST, NORTHWEST, SOUTHWEST
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

__attribute__((always_inline)) inline void resetCompletedOperations(bool **completedOperations, int *operationsCountLocal,  bool *cornerOperations, int *cornerOperationsCount, int *completedRequests, int threads) {
	#pragma omp master
	{
		*cornerOperationsCount = 4;
		for (int i = 0 ; i < 4 ; i++) {
			completedRequests[i] = 0;
			cornerOperations[i] = false;
			for (int j = 0 ; j < threads ; j++) {
				completedOperations[i][j] = false;
			}
		}
	}

	*operationsCountLocal = 4;
}

#endif