#include "mpi.h"
#include "util.h"
#include "array_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#define OUTPUTFILE "output"
#define XLEFT -1.0
#define XRIGHT 1.0
#define YBOTTOM -1.0
#define YTOP 1.0


int main(int argc, char **argv)
{   
	int processID = -1;
    int totalProcesses = -1;
    int tag = 0;
    double local_start, local_finish, local_elapsed, elapsed;

    MPI_Status status;
	struct JacobiParams jacobiParams;

	MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processID);

	// Create data types to broadcast program params
	MPI_Datatype jacobiParamsDatatype;
	int jacobiParamsBlockLengths[3] = {3, 1, 3};
	const MPI_Aint jacobiParamsDisplacements[3] = {
		offsetof(struct JacobiParams, maxIterations), 
		offsetof(struct JacobiParams, checkConvergence), 
		offsetof(struct JacobiParams, alpha)
	};
	MPI_Datatype jacobiParamsTypes[3] = {MPI_INT, MPI_C_BOOL, MPI_DOUBLE}; 
	MPI_Type_create_struct(3, jacobiParamsBlockLengths, jacobiParamsDisplacements, jacobiParamsTypes, &jacobiParamsDatatype);
    MPI_Type_commit(&jacobiParamsDatatype);

	if (!processID) {
		parseInput(&jacobiParams);
		remove(OUTPUTFILE);
	}

	MPI_Bcast(&jacobiParams, 1, jacobiParamsDatatype, 0, MPI_COMM_WORLD);

	// Unpack struct for cleaner use
	const int inputRows = jacobiParams.inputRows;
	const int inputColumns = jacobiParams.inputColumns;
	int maxIterations = jacobiParams.maxIterations;
	const bool checkConvergence = jacobiParams.checkConvergence;

    // Block partitioning
    const int dimensionProcesses = (const int) sqrt(totalProcesses);
	const bool perfectSquare = (dimensionProcesses * dimensionProcesses == totalProcesses);
    const int linesOfProcesses = dimensionProcesses;
    const int columnsOfProcesses = (perfectSquare ? dimensionProcesses : totalProcesses / linesOfProcesses);
    const int rows = inputRows / linesOfProcesses + 2;
    const int columns = inputColumns / columnsOfProcesses + 2;

	// Jacobi constants
	const double deltaX = (XRIGHT - XLEFT) / (jacobiParams.inputColumns - 1);
    const double deltaY = (YTOP - YBOTTOM) / (jacobiParams.inputRows - 1);
    const double cx = 1.0/(deltaX*deltaX);
    const double cy = 1.0/(deltaY*deltaY);
    const double cc = -2.0*cx-2.0*cy-jacobiParams.alpha;
	
	double error = 0.0;
	double finalError = HUGE_VAL;

	if (!processID) {
		printf("(%d) %d X %d\nProcesses: %d X %d\nRows: %d\nColumns %d\n\n", processID, inputRows, inputColumns, linesOfProcesses, columnsOfProcesses, rows, columns);
	}

	// 2d arrays as 1d for contiguous space
    double *array;
    if ((array = malloc(rows * columns * sizeof(double))) == NULL) {
        printf("Memory allocation for first array failed\n");
        exit(EXIT_FAILURE);
    }
    double *newArray;
    if ((newArray = malloc(rows * columns * sizeof(double))) == NULL) {
        printf("Memory allocation for second array failed\n");
        exit(EXIT_FAILURE);
    }
	initArrays(array, newArray, rows, columns);

	// Cartesian virtual topology
    MPI_Comm cartesianComm = MPI_COMM_WORLD;
    int cartesianDim[2] = {columnsOfProcesses, linesOfProcesses};
    int cartesianPeriod[2] = {0, 0};
    // MPI_Cart_create(MPI_COMM_WORLD, 2, cartesianDim, cartesianPeriod, 0, &cartesianComm);

    // Process' neighbours
    struct Neighbours n = constructNeighbours(processID, columnsOfProcesses, totalProcesses);
	
	// Create rowElements MPI datatype
    MPI_Datatype rowElementsDatatype;
    MPI_Type_contiguous(columns - 2, MPI_DOUBLE, &rowElementsDatatype);
    MPI_Type_commit(&rowElementsDatatype);

	// Create columnElements MPI datatype
    MPI_Datatype columnElementsDatatype;
    MPI_Type_vector(rows - 2, 1, columns, MPI_DOUBLE, &columnElementsDatatype);
    MPI_Type_commit(&columnElementsDatatype);


	// Setup persistent communication requests 
    MPI_Request sendStraightRequests[4];
    MPI_Request recvStraightRequests[4];
    MPI_Request sendReverseRequests[4];
    MPI_Request recvReverseRequests[4];
    MPI_Status statuses[4];

    MPI_Request *sendRequests = sendStraightRequests;
    MPI_Request *recvRequests = recvStraightRequests;
    for (int i = 0 ; i < 2 ; i++) {
        MPI_Send_init(&array[at(1,1, columns)], 1, rowElementsDatatype, n.north, tag, cartesianComm, &sendRequests[NORTH]);
        MPI_Recv_init(&array[at(rows-1,1, columns)], 1, rowElementsDatatype, n.south, tag, cartesianComm, &recvRequests[SOUTH]);

        MPI_Send_init(&array[at(rows-2,1, columns)], 1, rowElementsDatatype, n.south, tag, cartesianComm, &sendRequests[SOUTH]);
        MPI_Recv_init(&array[at(0,1, columns)], 1, rowElementsDatatype, n.north, tag, cartesianComm, &recvRequests[NORTH]);

        MPI_Send_init(&array[at(1,1, columns)], 1, columnElementsDatatype, n.west, tag, cartesianComm, &sendRequests[WEST]);
        MPI_Recv_init(&array[at(1,columns-1, columns)], 1, columnElementsDatatype, n.east, tag, cartesianComm, &recvRequests[EAST]);

        MPI_Send_init(&array[at(1,columns-2, columns)], 1, columnElementsDatatype, n.east, tag, cartesianComm, &sendRequests[EAST]);
        MPI_Recv_init(&array[at(1,0, columns)], 1, columnElementsDatatype, n.west, tag, cartesianComm, &recvRequests[WEST]);

        reverseDirection(&array, &newArray, &sendRequests, &recvRequests, sendStraightRequests, sendReverseRequests, recvStraightRequests, recvReverseRequests);
    }
	    
	// For doing work based on which halo points arrive */
    int completedOperations[8] = {0,0,0,0,0,0,0,0};
    int remainingOperations = 8;

	// Parallel program start 
    MPI_Barrier(cartesianComm);
	MPI_Pcontrol(1);
    local_start = MPI_Wtime();

    while (maxIterations-- || (false && checkConvergence && finalError > jacobiParams.tol)) {
        // Start sending and receiving halo points (non-blocking)
        MPI_Start(&recvRequests[SOUTH]);
        MPI_Start(&recvRequests[NORTH]);
        MPI_Start(&recvRequests[EAST]);
        MPI_Start(&recvRequests[WEST]);

        MPI_Start(&sendRequests[NORTH]);
        MPI_Start(&sendRequests[SOUTH]);
        MPI_Start(&sendRequests[WEST]);
        MPI_Start(&sendRequests[EAST]);

		error = 0.0;
		calculateInnerElements(array, newArray, rows, columns, &jacobiParams,
								YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);

		// Calculate outer elements as halo points arrive
        while (remainingOperations) {
            if (!completedOperations[NORTH]) {
                MPI_Test(&recvRequests[NORTH], &completedOperations[NORTH], &status);
                if (completedOperations[NORTH]) {
                    remainingOperations--;

                    for (int j = 2 ; j < columns - 2 ; j++) {
						calculateOneElement(1, j, array, newArray, rows, columns, &jacobiParams,
											YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);
                    }
                }
            }
            if (!completedOperations[SOUTH]) {
                MPI_Test(&recvRequests[SOUTH], &completedOperations[SOUTH], &status);
                if (completedOperations[SOUTH]) {
                    remainingOperations--;

                    for (int j = 2 ; j < columns - 2 ; j++) {
						calculateOneElement(rows - 2, j, array, newArray, rows, columns, &jacobiParams,
											YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);                   
					}
                }
            }
            if (!completedOperations[EAST]) {
                MPI_Test(&recvRequests[EAST], &completedOperations[EAST], &status);
                if (completedOperations[EAST]) {
                    remainingOperations--;

                    for (int i = 2 ; i < rows - 2 ; i++) {
						calculateOneElement(i, columns - 2, array, newArray, rows, columns, &jacobiParams,
											YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);                   
					}
                }
            }
            if (!completedOperations[WEST]) {
                MPI_Test(&recvRequests[WEST], &completedOperations[WEST], &status);
				if (completedOperations[WEST]) {
                    remainingOperations--;

                    for (int i = 2 ; i < rows - 2 ; i++) {
						calculateOneElement(i, 1, array, newArray, rows, columns, &jacobiParams,
											YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);                   
					}
                }
            }

			if (!completedOperations[NORTHEAST]) {
                if (completedOperations[NORTH] && completedOperations[EAST]) {
                    remainingOperations--;
                    completedOperations[NORTHEAST] = 1;
					calculateOneElement(1, columns-2, array, newArray, rows, columns, &jacobiParams,
											YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);                
					}
            }
            if (!completedOperations[NORTHWEST]) {
                if (completedOperations[NORTH] && completedOperations[WEST]) {
                    remainingOperations--;
                    completedOperations[NORTHWEST] = 1;
					calculateOneElement(1, 1, array, newArray, rows, columns, &jacobiParams,
											YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);               
					}
            }
            if (!completedOperations[SOUTHEAST]) {
                if (completedOperations[SOUTH] && completedOperations[EAST]) {
                    remainingOperations--;
                    completedOperations[SOUTHEAST] = 1;
					calculateOneElement(rows - 2, columns - 2, array, newArray, rows, columns, &jacobiParams,
											YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);                
					}
            }
            if (!completedOperations[SOUTHWEST]) {
                if (completedOperations[SOUTH] && completedOperations[WEST]) {
                    remainingOperations--;
                    completedOperations[SOUTHWEST] = 1;					
					calculateOneElement(rows - 2, 1, array, newArray, rows, columns, &jacobiParams,
											YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error);                
					}
            }
        }

        resetCompletedOperations(completedOperations, &remainingOperations);

        if (checkConvergence) {
			printf("Process %d error %f\n", processID, error);
            MPI_Allreduce(&error, &finalError, 1, MPI_DOUBLE, MPI_SUM, cartesianComm);
			finalError = sqrt(finalError)/(((inputColumns))*((inputRows)));
        }
		// 840 1 and 4 process, correct is inputColumns

        MPI_Waitall(4, sendRequests, statuses);
        reverseDirection(&array, &newArray, &sendRequests, &recvRequests, sendStraightRequests, sendReverseRequests, recvStraightRequests, recvReverseRequests);
	}

	local_finish = MPI_Wtime();
	MPI_Pcontrol(0);
    // MPI_File_write_all(outputHandle, array, 1, memtype, &status);
    local_elapsed = local_finish - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartesianComm);
    if (!processID) {
        printf("Elapsed time: %.2lf\n", elapsed);
		if (jacobiParams.checkConvergence) {
			printf("Error is %g\n",finalError);
		}
	}

	if (processID == 0) {
		printf("Process %d\n", processID);
		for (int i = 0 ; i < rows ; i++) {
		printf("\nLine %d\n", i);
		for (int j = 0 ; j < columns ; j++) {
			printf("%f ", array[at(i, j, columns)]);
		}
		printf("\n");
	}
	}

    // MPI_File_close(&outputHandle);
    MPI_Type_free(&rowElementsDatatype);
    MPI_Type_free(&columnElementsDatatype);
    MPI_Finalize();
    free(array);
    free(newArray);
	return 0;
}