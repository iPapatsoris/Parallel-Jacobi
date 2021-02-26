#include <timestamp.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "util.h"

#define BLOCK_SIZE 256

#define XRIGHT 1
#define XLEFT -1
#define YTOP 1
#define YBOTTOM -1

#define ROUND_UP(XX, YY) ((double) (XX + YY - 1) / YY);

struct Neighbors {
	int north;
	int south;
	int west;
	int east;
	int center;
};

__host__ cudaError_t allocateMemory(double *array, int arrayBytes, int extraHaloBytes, double **arrayDevice, double **newArrayDevice, double **errorArrayDevice, double **newErrorArrayDevice, double **extraHaloDevice);
__global__ void kjacobi(double *array, double *newArray, int inputRows, int inputColumns, const JacobiParams jacobiParams, const JacobiConstants jacobiConstants, double *errorArray, double *extraHalo, int deviceId);
__global__ void ksum(double *array, double *newArray);
__device__ __attribute__((always_inline)) inline Neighbors constructNeighbors(int i, int columns);
__device__ __attribute__((always_inline)) inline void calculateOneElement(const int y, const int x, 
	const struct Neighbors *neighbors, const double *sharedArray, double *array, 
	const struct JacobiParams jacobiParams, const struct JacobiConstants, double *errorArray, const int inputRows, const int deviceId);

extern "C" float jacobiGPU(double *array, int elements, int inputRows, int inputColumns, JacobiParams jacobiParams, double *extraHaloDevice[2], int threadId, double *finalError) {
	int totalGPUs, deviceId, otherDeviceId;
	double localGridError;
	cudaError_t err;
	
	cudaGetDeviceCount (&totalGPUs);
	cudaSetDevice(threadId % totalGPUs);
	cudaGetDevice(&deviceId);
	otherDeviceId = (deviceId ? 0 : 1);
	cudaDeviceEnablePeerAccess(otherDeviceId, 0);

	// Calculate grid/block dimensions
	int rowsOfBlocks, columnsOfBlocks, rowsOfBlockThreads, columnsOfBlockThreads;
	divide2D(BLOCK_SIZE, &rowsOfBlockThreads, &columnsOfBlockThreads);
	rowsOfBlocks = ceil((float) inputRows / rowsOfBlockThreads);
	columnsOfBlocks = ceil((float) inputColumns / columnsOfBlockThreads);

	// Shared memory for fast operations per block
	int sharedMemorySize = (rowsOfBlockThreads + 2) * (columnsOfBlockThreads + 2);
	int sharedMemoryBytes = sharedMemorySize * sizeof(double);
	
	dim3 dimBl(columnsOfBlockThreads, rowsOfBlockThreads);
	dim3 dimGr(columnsOfBlocks, rowsOfBlocks);

	printf("Elements %d\nGrid %d X %d\nBlock threads %d X %d\n", elements, rowsOfBlocks, columnsOfBlocks, rowsOfBlockThreads, columnsOfBlockThreads);

	int arrayBytes = elements * sizeof(double);
	int extraHaloBytes = inputColumns * sizeof(double);
	double *arrayDevice, *newArrayDevice, *errorArrayDevice, *newErrorArrayDevice;
	if ((err = allocateMemory(array, arrayBytes, extraHaloBytes, &arrayDevice, &newArrayDevice, &errorArrayDevice, &newErrorArrayDevice, &(extraHaloDevice[deviceId]) )) != cudaSuccess) {
		return err;
	}
	int haloToSendIndex = (deviceId ? 0 : (inputRows-1) * inputColumns);

	double *srcArray = arrayDevice;
	double *dstArray = newArrayDevice;
	JacobiConstants jacobiConstants = getJacobiConstants(XLEFT, XRIGHT, YBOTTOM, YTOP, 2*inputRows, inputColumns, jacobiParams.alpha);

	// For parallel error sum reduction
	dim3 dimBlSum(BLOCK_SIZE);
	int sumSharedMemoryBytes = BLOCK_SIZE * sizeof(double); 
	int globalElements = elements * 2;

	int iterations = jacobiParams.maxIterations;
	timestamp t_start;
	t_start = getTimestamp();

	#pragma omp barrier
	while(iterations--) {
		*finalError = 0;
		cudaMemcpyPeer(extraHaloDevice[otherDeviceId], otherDeviceId, &(srcArray[haloToSendIndex]), deviceId, extraHaloBytes); // debug extra halo that each gpu receives
		#pragma omp barrier
		cudaDeviceSynchronize();
		kjacobi<<<dimGr, dimBl, sharedMemoryBytes>>>(srcArray, dstArray, inputRows, inputColumns, jacobiParams, jacobiConstants, errorArrayDevice, extraHaloDevice[deviceId], deviceId);

		if (jacobiParams.checkConvergence) {
			double *srcErrorArray = errorArrayDevice;
			double *dstErrorArray = newErrorArrayDevice;
			int errorElements = elements;

			while (errorElements > 1) {
				int errorBlocks = ROUND_UP((double) errorElements, BLOCK_SIZE);
				
				dim3 dimGrSum(errorBlocks);
				ksum<<<dimGrSum, dimBlSum, sumSharedMemoryBytes>>>(srcErrorArray, dstErrorArray);
				
				err = cudaDeviceSynchronize();
				if (err != cudaSuccess){
					fprintf(stderr, "GPUassert: %s\n",err);
					return err;
				}

				errorElements = errorBlocks;
				if (errorElements > 1) {
					swap(&srcErrorArray, &dstErrorArray);
				}
			}

			err = cudaMemcpy(&localGridError, dstErrorArray, sizeof(double), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess){
				fprintf(stderr, "GPUassert: %s\n",err);
				return err;
			}

			#pragma omp atomic
				(*finalError) += localGridError;
			#pragma omp barrier
			#pragma omp master 
				(*finalError) = sqrt(*finalError) / globalElements;
			#pragma omp barrier
			if (*finalError <= jacobiParams.tol) {
				break;
			}
		}

		if (iterations) {
			swap(&srcArray, &dstArray);
		}
		
		#pragma omp barrier
	}

	float msecs = getElapsedtime(t_start);
	#pragma omp master
	{
		printf("Error is %g\n", *finalError);
		cudaMemcpy(array, dstArray, arrayBytes, cudaMemcpyDeviceToHost);
	}
	cudaFree(arrayDevice);
	cudaFree(newArrayDevice);
	cudaFree(errorArrayDevice);
	cudaFree(newErrorArrayDevice);

	return msecs;
}

__global__ void kjacobi(double *array, double *newArray, int inputRows, int inputColumns, const JacobiParams jacobiParams, const JacobiConstants jacobiConstants, double *errorArray, double *extraHalo, int deviceId) {
	const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int i = row * inputColumns + column;  

	extern __shared__ double sharedArray[];
	const unsigned int sharedDimX = blockDim.x + 2;

	const unsigned int columnShared = threadIdx.x + 1;
	const unsigned int rowShared = threadIdx.y + 1;
	const unsigned int iShared = rowShared * sharedDimX + columnShared; 

	struct Neighbors arrayNeighbors = constructNeighbors(i, inputColumns);
	struct Neighbors sharedArrayNeighbors = constructNeighbors(iShared, sharedDimX);

	if (column < inputColumns && row < inputRows) {
		sharedArray[iShared] = array[i];

		// Compute halo points. Set 0 when there isn't neighbor block, or when we're at the end of array within the last block 
		if (!threadIdx.y) { // First row of block
			if (blockIdx.y > 0) {
				sharedArray[sharedArrayNeighbors.north] = array[arrayNeighbors.north];
			} else {
				sharedArray[sharedArrayNeighbors.north] = (deviceId ? extraHalo[column] : 0);
			}
			
		}
		if (threadIdx.y == blockDim.y - 1 || row == inputRows - 1) { // Last row of block OR last row of array 
			if (blockIdx.y < gridDim.y - 1 && row < inputRows - 1 ) {
				sharedArray[sharedArrayNeighbors.south] = array[arrayNeighbors.south];
			} else {
				sharedArray[sharedArrayNeighbors.south] = (!deviceId ? extraHalo[column] : 0);
			}
		}
		if (!threadIdx.x) { // First column of block
			sharedArray[sharedArrayNeighbors.west] = (blockIdx.x > 0 ? array[arrayNeighbors.west] : 0);
		}
		if (threadIdx.x == blockDim.x - 1 || column == inputColumns - 1) { // Last column of block OR last column of array
			sharedArray[sharedArrayNeighbors.east] = (blockIdx.x < gridDim.x - 1 && column < inputColumns - 1 ? array[arrayNeighbors.east] : 0);
		}
	}

	__syncthreads();

	if (column < inputColumns && row < inputRows) {
		calculateOneElement(row, column, &sharedArrayNeighbors, sharedArray, &newArray[i], jacobiParams, jacobiConstants, &errorArray[i], inputRows, deviceId);
	}

	__syncthreads();
}

__global__ void ksum(double *array, double *newArray) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iShared = threadIdx.x;
	extern __shared__ double sharedArray[];
	sharedArray[iShared] = array[i];
	__syncthreads();

	int threads = ROUND_UP(blockDim.x, 2);
	while (threads > 0) {
		if (threadIdx.x < threads) {
			int first = iShared * 2;
			sharedArray[iShared] = sharedArray[first] + sharedArray[first + 1];
		}		
		__syncthreads();
		threads /= 2;
	}
	
	if (!threadIdx.x) {
		newArray[blockIdx.x] = sharedArray[iShared];
	}
}

__device__ __attribute__((always_inline)) inline void calculateOneElement(const int y, const int x, 
	const struct Neighbors *neighbors, const double *sharedArray, double *array, 
	const struct JacobiParams jacobiParams, const struct JacobiConstants jacobiConstants, double *errorArray, const int inputRows, const int deviceId) {

	int globalY = (deviceId ? y + inputRows : y);

	double fY = YBOTTOM + (globalY)*jacobiConstants.deltaY;
	double fYSquare = fY*fY;
	double fX = XLEFT + (x)*jacobiConstants.deltaX;
	double fXSquare = fX*fX;

	double f = -jacobiParams.alpha*(1.0-fXSquare)*(1.0-fYSquare) - 2.0*(1.0-fXSquare) - 2.0*(1.0-fYSquare);
	double curVal = sharedArray[neighbors->center];
	double updateVal = ((sharedArray[neighbors->west] + sharedArray[neighbors->east])*jacobiConstants.cx +
	(sharedArray[neighbors->north] + sharedArray[neighbors->south])*jacobiConstants.cy + curVal*jacobiConstants.cc - f
	)/jacobiConstants.cc;
	
	*array = curVal - jacobiParams.relax*updateVal;
	*errorArray = updateVal*updateVal;
}

__device__ __attribute__((always_inline)) inline Neighbors constructNeighbors(int i, int columns) {
	struct Neighbors neighbors;
	neighbors.center = i;
	neighbors.north = i - columns;
	neighbors.south = i + columns;
	neighbors.west = i - 1;
	neighbors.east = i + 1;

	return neighbors;
}

__host__ cudaError_t allocateMemory(double *array, int arrayBytes, int extraHaloBytes, double **arrayDevice, double **newArrayDevice, double **errorArrayDevice, double **newErrorArrayDevice, double **extraHaloDevice) {
	cudaError_t err = cudaMalloc((void **)arrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	err = cudaMemcpy(*arrayDevice, array, arrayBytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	err = cudaMalloc((void **)newArrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	err = cudaMalloc((void **)errorArrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	err = cudaMalloc((void **)newErrorArrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	err = cudaMalloc((void **)extraHaloDevice, extraHaloBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	return err;
}