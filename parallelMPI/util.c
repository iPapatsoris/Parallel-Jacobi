#include "util.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

struct Neighbours constructNeighbours(int processID, const int columnsOfProcesses, int totalProcesses) {
    struct Neighbours n;
    n.north = processID - columnsOfProcesses;
    n.south = processID + columnsOfProcesses;
    n.east = processID + 1;
    n.west = processID - 1;
    n.northeast = n.north + 1;
    n.northwest = n.north - 1;
    n.southeast = n.south + 1;
    n.southwest = n.south - 1;
    if (n.north < 0) {
        n.north = MPI_PROC_NULL;
        n.northeast = MPI_PROC_NULL;
        n.northwest = MPI_PROC_NULL;
    }
    if (n.south >= totalProcesses) {
        n.south = MPI_PROC_NULL;
        n.southeast = MPI_PROC_NULL;
        n.southwest = MPI_PROC_NULL;
    }
    if (n.east % columnsOfProcesses == 0) {
        n.east = MPI_PROC_NULL;
        n.northeast = MPI_PROC_NULL;
        n.southeast = MPI_PROC_NULL;
    }
    if (n.west < 0 || processID % columnsOfProcesses == 0) {
        n.west = MPI_PROC_NULL;
        n.northwest = MPI_PROC_NULL;
        n.southwest = MPI_PROC_NULL;
    }
    return n;
}

void printNeighbours(const struct Neighbours n, const int processID) {
    printf("Process %d\nnorth: %d\nsouth: %d\neast: %d\nwest: %d\nnortheast: %d\nnorthwest: %d\nsoutheast: %d\nsouthwest: %d\n\n", processID, n.north, n.south, n.east, n.west, n.northeast, n.northwest, n.southeast, n.southwest);
}

void parseInput(int *maxIterations, bool *checkConvergence, int *inputRows, int *inputColumns, struct JacobiParams *jacobiParams) {
//    printf("Input n,m - grid dimension in x,y direction:\n");
    scanf("%d,%d", inputRows, inputColumns);
//    printf("Input checkConvergence - check if we are within error tolerance on each iteration:\n");
    scanf("%d", checkConvergence);
//    printf("Input alpha - Helmholtz constant:\n");
    scanf("%lf", &jacobiParams->alpha);
//    printf("Input relax - successive over-relaxation parameter:\n");
    scanf("%lf", &jacobiParams->relax);
//    printf("Input tol - error tolerance for the iterrative solver:\n");
    scanf("%lf", &jacobiParams->tol);
//    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", maxIterations);
}