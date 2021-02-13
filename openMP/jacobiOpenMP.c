#include "mpi.h"
#include "util.h"
#include "array_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#else
int omp_get_thread_num(void) {return 0; }
int omp_get_num_threads(void) { return 1; }
printf("not defined LOL");
#endif

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
	int prov, threads;
    double local_start, local_finish, local_elapsed, elapsed;

    MPI_Status status;
	struct JacobiParams jacobiParams;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);		
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
	const double deltaX = (XRIGHT - XLEFT) / (inputColumns - 1);
    const double deltaY = (YTOP - YBOTTOM) / (inputRows - 1);
    const double cx = 1.0/(deltaX*deltaX);
    const double cy = 1.0/(deltaY*deltaY);
    const double cc = -2.0*cx-2.0*cy-jacobiParams.alpha;
	const int yIncrement = (processID / columnsOfProcesses) * (rows - 2);
	const int xIncrement = (processID % columnsOfProcesses) * (columns - 2);
	
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

	/* Parallel I/O */
    MPI_File outputHandle;
    int gSizes[2], lSizes[2], memSizes[2], startIndices[2];
    gSizes[0] = inputRows;
    gSizes[1] = inputColumns;
    lSizes[0] = rows - 2;
    lSizes[1] = columns - 2;
    memSizes[0] = rows;
    memSizes[1] = columns;
    startIndices[0] = startIndices[1] = 1;

    MPI_Datatype memtype;
    MPI_Type_create_subarray(2, memSizes, lSizes, startIndices, MPI_ORDER_C, MPI_DOUBLE, &memtype);
    MPI_Type_commit(&memtype);

    MPI_Datatype filetype;
    startIndices[0] = (processID / columnsOfProcesses) * lSizes[0];
    startIndices[1] = (processID % columnsOfProcesses) * lSizes[1];

	MPI_Type_create_subarray(2, gSizes, lSizes, startIndices, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File_open(cartesianComm, OUTPUTFILE, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &outputHandle);
    MPI_File_set_view(outputHandle, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);


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
	    
	// For monitoring which corner elements have been processed
	bool cornerOperations[4] = {0,0,0,0};
	int cornerOperationsCount = 4;

	// For monitoring which MPI Receives are ready
	int completedRequests[4] = {0,0,0,0}; 

	// For monitoring which threads have finished working on which sides of the array
	bool *completedOperations[4];


	#pragma omp parallel
	{	
		threads = omp_get_num_threads();
		int threadID = omp_get_thread_num();
		int operationsCountLocal = 4;
		#pragma omp master 
		{
			for (int i = 0 ; i < operationsCountLocal ; i++) {
				completedOperations[i] = malloc(threads * sizeof(bool));
				for (int j = 0 ; j < threads ; j++) {
					completedOperations[i][j] = false;
				}
			}	
			// Parallel program start 
			MPI_Barrier(cartesianComm);
			MPI_Pcontrol(1);
			local_start = MPI_Wtime();
			printf("Total threads in process %d: %d\n", processID, threads);
		}
		#pragma omp barrier

		while (maxIterations) {
			#pragma omp master 
			{
				// Start sending and receiving halo points (non-blocking)
				MPI_Start(&recvRequests[SOUTH]);
				MPI_Start(&recvRequests[NORTH]);
				MPI_Start(&recvRequests[EAST]);
				MPI_Start(&recvRequests[WEST]);

				MPI_Start(&sendRequests[NORTH]);
				MPI_Start(&sendRequests[SOUTH]);
				MPI_Start(&sendRequests[WEST]);
				MPI_Start(&sendRequests[EAST]);
				MPI_Barrier(cartesianComm);
			}

			calculateInnerElements(threads, array, newArray, rows, columns, &jacobiParams,
									YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);

			// Calculate outer elements as halo points arrive
			while (operationsCountLocal || cornerOperationsCount) {
				if (!completedOperations[NORTH][threadID]) {
					MPI_Test(&recvRequests[NORTH], &completedRequests[NORTH], &status);
					completedOperations[NORTH][threadID] = completedRequests[NORTH];
					if (completedOperations[NORTH][threadID]) {
						operationsCountLocal--;

						#pragma omp for schedule(static)
						for (int j = 2 ; j < columns - 2 ; j++) {
							calculateOneElement(1, j, array, newArray, rows, columns, &jacobiParams,
												YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);
						}
					}
				}
				if (!completedOperations[SOUTH][threadID]) {
					MPI_Test(&recvRequests[SOUTH], &completedRequests[SOUTH], &status);
					completedOperations[SOUTH][threadID] = completedRequests[SOUTH];
					if (completedOperations[SOUTH][threadID]) {
						operationsCountLocal--;

						#pragma omp for schedule(static)
						for (int j = 2 ; j < columns - 2 ; j++) {
							calculateOneElement(rows - 2, j, array, newArray, rows, columns, &jacobiParams,
												YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);                   
						}
					}
				}
				if (!completedOperations[EAST][threadID]) {
					MPI_Test(&recvRequests[EAST], &completedRequests[EAST], &status);
					completedOperations[EAST][threadID] = completedRequests[EAST];
					if (completedOperations[EAST][threadID]) {
						operationsCountLocal--;

						#pragma omp for schedule(static)
						for (int i = 2 ; i < rows - 2 ; i++) {
							calculateOneElement(i, columns - 2, array, newArray, rows, columns, &jacobiParams,
												YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);                   
						}
					}
				}
				if (!completedOperations[WEST][threadID]) {
					MPI_Test(&recvRequests[WEST], &completedRequests[WEST], &status);
					completedOperations[WEST][threadID] = completedRequests[WEST];
					if (completedOperations[WEST][threadID]) {
						operationsCountLocal--;

						#pragma omp for schedule(static)
						for (int i = 2 ; i < rows - 2 ; i++) {
							calculateOneElement(i, 1, array, newArray, rows, columns, &jacobiParams,
												YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);                   
						}
					}
				}

				#pragma omp critical(NORTHEAST)
				if (!cornerOperations[NORTHEAST]) {
					if (completedRequests[NORTH] && completedRequests[EAST]) {
						cornerOperationsCount--;
						cornerOperations[NORTHEAST] = 1;
						calculateOneElement(1, columns-2, array, newArray, rows, columns, &jacobiParams,
												YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);                
						}
				}

				#pragma omp critical(NORTHWEST)
				if (!cornerOperations[NORTHWEST]) {
					if (completedRequests[NORTH] && completedRequests[WEST]) {
						cornerOperationsCount--;
						cornerOperations[NORTHWEST] = 1;
						calculateOneElement(1, 1, array, newArray, rows, columns, &jacobiParams,
												YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);               
						}
				}

				#pragma omp critical(SOUTHEAST)
				if (!cornerOperations[SOUTHEAST]) {
					if (completedRequests[SOUTH] && completedRequests[EAST]) {
						cornerOperationsCount--;
						cornerOperations[SOUTHEAST] = 1;
						calculateOneElement(rows - 2, columns - 2, array, newArray, rows, columns, &jacobiParams,
												YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);                
						}
				}

				#pragma omp critical(SOUTHWEST)
				if (!cornerOperations[SOUTHWEST]) {
					if (completedRequests[SOUTH] && completedRequests[WEST]) {
						cornerOperationsCount--;
						cornerOperations[SOUTHWEST] = 1;					
						calculateOneElement(rows - 2, 1, array, newArray, rows, columns, &jacobiParams,
												YBOTTOM, XLEFT, deltaY, deltaX, cy, cx, cc, &error, yIncrement, xIncrement);                
						}
				}
			}

			#pragma omp barrier
			resetCompletedOperations(completedOperations, &operationsCountLocal, cornerOperations, &cornerOperationsCount, completedRequests, threads);

			#pragma omp master
			{
				if (checkConvergence) {
					MPI_Allreduce(&error, &finalError, 1, MPI_DOUBLE, MPI_SUM, cartesianComm);
					finalError = sqrt(finalError)/(inputColumns * inputRows);
					error = 0.0;
				}

				MPI_Waitall(4, sendRequests, statuses);
				reverseDirection(&array, &newArray, &sendRequests, &recvRequests, sendStraightRequests, sendReverseRequests, recvStraightRequests, recvReverseRequests);
				maxIterations--;
			}
			#pragma omp barrier
			if (checkConvergence && finalError <= jacobiParams.tol) {
				break;
			}
		}
	}

	if (checkConvergence && finalError <= jacobiParams.tol) {
		printf("Finished with %d iterations left!\n", maxIterations);
	}

	local_finish = MPI_Wtime();
	MPI_Pcontrol(0);
    local_elapsed = local_finish - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartesianComm);
	MPI_File_write_all(outputHandle, array, 1, memtype, &status);
    if (!processID) {
        printf("Elapsed time: %.2lf\n", elapsed);
		if (jacobiParams.checkConvergence) {
			printf("Error is %g\n",finalError);
		}
	}

    MPI_File_close(&outputHandle);
    MPI_Type_free(&rowElementsDatatype);
    MPI_Type_free(&columnElementsDatatype);
	MPI_Type_free(&jacobiParamsDatatype);
	MPI_Type_free(&memtype);
	MPI_Type_free(&filetype);
    MPI_Finalize();
    free(array);
    free(newArray);
	return 0;
}