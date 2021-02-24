#include <lcutil.h>
#include <timestamp.h>
#include <stdio.h>
#include <math.h>
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

__device__ Neighbors constructNeighbors(int i, int columns);
__device__ __attribute__((always_inline)) inline void calculateOneElement(const int y, const int x, 
	const struct Neighbors *neighbors, const double *sharedArray, double *array, 
	const struct JacobiParams jacobiParams, double *errorArray, const int inputColumns, const int inputRows);

__global__ void kjacobi(double *array, double *newArray, int inputRows, int inputColumns, const JacobiParams jacobiParams, const int sharedMemorySize, double *errorArray) {
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
			sharedArray[sharedArrayNeighbors.north] = (blockIdx.y > 0 ? array[arrayNeighbors.north] : 0);
		}
		if (threadIdx.y == blockDim.y - 1 || row == inputRows - 1) { // Last row of block OR last row of array 
			sharedArray[sharedArrayNeighbors.south] = (blockIdx.y < gridDim.y - 1 && row < inputRows - 1 ? array[arrayNeighbors.south] : 0);
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
		calculateOneElement(row, column, &sharedArrayNeighbors, sharedArray, &newArray[i], jacobiParams, &errorArray[i], inputColumns, inputRows);
	}

	__syncthreads();
}

__global__ void ksum(double *array, double *newArray) {
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int iShared = threadIdx.x;
	extern __shared__ double sharedArray[];
	sharedArray[iShared] = array[i];
	//printf("error[%d] %.15f\n",  i, array[i]);
	__syncthreads();

	int threads = ROUND_UP(blockDim.x, 2);
	while (threads > 0) {
		if (threadIdx.x < threads) {
			int first = iShared * 2;
			sharedArray[iShared] = sharedArray[first] + sharedArray[first + 1];
			//printf("Block %d thread %d sum %.15f + %.15f\n", blockIdx.x, threadIdx.x, sharedArray[first], sharedArray[first+1]);
		}		
		__syncthreads();
		threads /= 2;
	}
	
	if (!threadIdx.x) {
		newArray[blockIdx.x] = sharedArray[iShared];
		//printf("Block %d %.15f\n", blockIdx.x, newArray[i]);
	}
}

extern "C" float jacobiGPU(double *array, int elements, int inputRows, int inputColumns, JacobiParams jacobiParams) {
	double *arrayDevice;
	double finalError;
	cudaError_t err;
	int arrayBytes = elements * sizeof(double);

	int rowsOfBlocks, columnsOfBlocks, rowsOfBlockThreads, columnsOfBlockThreads;
	divide2D(BLOCK_SIZE, &rowsOfBlockThreads, &columnsOfBlockThreads);
	rowsOfBlocks = ceil((float) inputRows / rowsOfBlockThreads);
	columnsOfBlocks = ceil((float) inputColumns / columnsOfBlockThreads);

	int sharedMemorySize = (rowsOfBlockThreads + 2) * (columnsOfBlockThreads + 2);
	
	dim3 dimBl(columnsOfBlockThreads, rowsOfBlockThreads);
	dim3 dimGr(columnsOfBlocks, rowsOfBlocks);

	printf("Elements %d\nGrid %d X %d\nBlock threads %d X %d\n", elements, rowsOfBlocks, columnsOfBlocks, rowsOfBlockThreads, columnsOfBlockThreads);

	err = cudaMalloc((void **)&arrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	// Copy data to device memory
	err = cudaMemcpy(arrayDevice, array, arrayBytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	double *newArrayDevice;

	err = cudaMalloc((void **)&newArrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	// Copy data to device memory
	err = cudaMemcpy(newArrayDevice, array, arrayBytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	double *errorArray = (double *) malloc(arrayBytes);
	double *errorArrayDevice;
	err = cudaMalloc((void **)&errorArrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	double *newErrorArrayDevice;
	err = cudaMalloc((void **)&newErrorArrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	double *srcArray = arrayDevice;
	double *dstArray = newArrayDevice;

	int iterations = jacobiParams.maxIterations;
	timestamp t_start;
	t_start = getTimestamp();

	while(iterations--) {
		kjacobi<<<dimGr, dimBl, sharedMemorySize * sizeof(double)>>>(srcArray, dstArray, inputRows, inputColumns, jacobiParams, sharedMemorySize, errorArrayDevice);
		err = cudaGetLastError();
		if (err != cudaSuccess){
			fprintf(stderr, "GPUassert: %s\n",err);
			return err;
		}
	
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess){
			fprintf(stderr, "GPUassert: %s\n",err);
			return err;
		}

		err = cudaMemcpy(errorArray, errorArrayDevice, arrayBytes, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess){
			fprintf(stderr, "GPUassert: %s\n",err);
			return err;
		}
		
		for (int i = 0 ; i < elements ; i++) {
			;//printf("CPU error[%d]: %.15f\n", i, errorArray[i]);
		}

		if (jacobiParams.checkConvergence) {
			double *srcErrorArray = errorArrayDevice;
			double *dstErrorArray = newErrorArrayDevice;
			int errorElements = elements;

			while (errorElements > 1) {
				int errorBlocks = ceil((double) errorElements / BLOCK_SIZE);
				//printf("blocks %d elements %d \n", errorBlocks, errorElements);
				
				dim3 dimBlSum(BLOCK_SIZE);
				dim3 dimGrSum(errorBlocks);
				ksum<<<dimGrSum, dimBlSum, BLOCK_SIZE * sizeof(double)>>>(srcErrorArray, dstErrorArray);
				
				err =cudaGetLastError();
				if (err != cudaSuccess){
						fprintf(stderr, "GPUassert: %s\n",err);
						return err;
				}
				err = cudaDeviceSynchronize();
				if (err != cudaSuccess){
					fprintf(stderr, "GPUassert: %s\n",err);
					return err;
				}

				errorElements = errorBlocks;
				//return 0; ///////

				if (errorElements > 1) {
					double *tmp = srcErrorArray;
					srcErrorArray = dstErrorArray;
					dstErrorArray = tmp;
				}
			}

			err = cudaMemcpy(errorArray, dstErrorArray, sizeof(double), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess){
				fprintf(stderr, "GPUassert: %s\n",err);
				return err;
			}

			finalError = sqrt(errorArray[0])/(inputColumns * inputRows);
			if (finalError <= jacobiParams.tol) {
				break;
			}
		}

		if (iterations) {
			double *tmp = srcArray;
			srcArray = dstArray;
			dstArray = tmp;
		}
	}

	float msecs = getElapsedtime(t_start);
	

	err = cudaMemcpy(array, dstArray, arrayBytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	cudaFree(arrayDevice);

	printf("Error is %g\n", finalError);

	return msecs;
}

__device__ __attribute__((always_inline)) inline void calculateOneElement(const int y, const int x, 
	const struct Neighbors *neighbors, const double *sharedArray, double *array, 
	const struct JacobiParams jacobiParams, double *errorArray, const int inputColumns, const int inputRows) {

	// Jacobi constants
	const double deltaX = (double) (XRIGHT - XLEFT) / (inputColumns - 1);
	const double deltaY = (double) (YTOP - YBOTTOM) / (inputRows - 1);
	const double cx = 1.0/(deltaX*deltaX);
	const double cy = 1.0/(deltaY*deltaY);
	const double cc = -2.0*cx-2.0*cy-jacobiParams.alpha;

	double fY = YBOTTOM + (y)*deltaY;
	double fYSquare = fY*fY;
	double fX = XLEFT + (x)*deltaX;
	double fXSquare = fX*fX;

	double f = -jacobiParams.alpha*(1.0-fXSquare)*(1.0-fYSquare) - 2.0*(1.0-fXSquare) - 2.0*(1.0-fYSquare);
	double curVal = sharedArray[neighbors->center];
	double updateVal = ((sharedArray[neighbors->west] + sharedArray[neighbors->east])*cx +
	(sharedArray[neighbors->north] + sharedArray[neighbors->south])*cy + curVal*cc - f
	)/cc;
	
	*array = curVal - jacobiParams.relax*updateVal;
	*errorArray = updateVal*updateVal;
	//printf("%f\n", *errorArray);
}

__device__ Neighbors constructNeighbors(int i, int columns) {
	struct Neighbors neighbors;
	neighbors.center = i;
	neighbors.north = i - columns;
	neighbors.south = i + columns;
	neighbors.west = i - 1;
	neighbors.east = i + 1;

	return neighbors;
}