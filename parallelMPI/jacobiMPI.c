#include "mpi.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#define OUTPUTFILE "output"

int main(int argc, char **argv)
{
    int maxIterations, inputRows, inputColumns;
	bool checkConvergence;
	struct JacobiParams jacobiParams;
    parseInput(&maxIterations, &checkConvergence, &inputRows, &inputColumns, &jacobiParams);
	remove(OUTPUTFILE);

    int processID = -1;
    int totalProcesses = -1;
    int tag = 0;
    double local_start, local_finish, local_elapsed, elapsed;
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processID);

    // Block partitioning
    const int dimensionProcesses = (const int) sqrt(totalProcesses);
	const bool perfectSquare = (dimensionProcesses * dimensionProcesses == totalProcesses);
    const int linesOfProcesses = dimensionProcesses;
    const int columnsOfProcesses = (perfectSquare ? dimensionProcesses : totalProcesses / linesOfProcesses);
    const int rows = inputRows / linesOfProcesses + 2;
    const int columns = inputColumns / columnsOfProcesses + 2;

	if (!processID) {
		printf("%d X %d\nProcesses: %d X %d\nRows: %d\nColumns %d\n\n", inputRows, inputColumns, linesOfProcesses, columnsOfProcesses, rows, columns);
	}

	// Cartesian virtual topology
    MPI_Comm cartesianComm = MPI_COMM_WORLD;
    int cartesianDim[2] = {columnsOfProcesses, linesOfProcesses};
    int cartesianPeriod[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, cartesianDim, cartesianPeriod, 0, &cartesianComm);

    // Process' neighbours
    struct Neighbours n = constructNeighbours(processID, columnsOfProcesses, totalProcesses);
	printNeighbours(n, processID);

	
	
	MPI_Pcontrol(1);
	
	MPI_Pcontrol(0);
	MPI_Finalize();
}