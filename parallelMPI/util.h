#ifndef UTIL
#define UTIL

#include "mpi.h"
#include <stdbool.h>

enum {
    NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST, NORTH, SOUTH, EAST, WEST
};

struct Neighbours {
    int north, south, east, west, northeast, northwest, southeast, southwest;
};

struct JacobiParams {
	double alpha, relax, tol;
};

struct Neighbours constructNeighbours(const int processID, const int columnsOfProcesses, const int totalProcesses);
void printNeighbours(const struct Neighbours n, const int processID);
void parseInput(int *maxIterations, bool *checkConvergence, int *inputRows, int *inputColumns, struct JacobiParams *jacobiParams);
// void performParallelIO(char *inputFile, const int totalLines, const int totalColumns, const int processLines, const int columnsOfProcesses, const int lines, const int columns,
//                       MPI_Datatype *inputHandle, MPI_Datatype *outputHandle, MPI_Datatype *pixelDatatype, MPI_Datatype *memtype, MPI_Datatype *filetype);

__attribute__((always_inline)) inline void reverseDirection(unsigned char **array, unsigned char **newArray, MPI_Request **sendRequests, MPI_Request **recvRequests, MPI_Request *sendStraightRequests, MPI_Request *sendReverseRequests, MPI_Request *recvStraightRequests, MPI_Request *recvReverseRequests) {
    unsigned char *tmp = *array;
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

__attribute__((always_inline)) inline void resetCompletedRequests(int *completedRequests, int *remainingRequests, int *cornerFlags) {
    completedRequests[0] = 0; // Manual assignments instead of looping for performance
    completedRequests[1] = 0;
    completedRequests[2] = 0;
    completedRequests[3] = 0;
    completedRequests[4] = 0;
    completedRequests[5] = 0;
    completedRequests[6] = 0;
    completedRequests[7] = 0;
    *remainingRequests = 8;
    cornerFlags[0] = 0;
    cornerFlags[1] = 0;
    cornerFlags[2] = 0;
    cornerFlags[3] = 0;
}

#endif