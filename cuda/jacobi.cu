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

__global__ void kjacobi(double *array, double *newArray, int N, int inputRows, int inputColumns, const JacobiParams jacobiParams, const int sharedMemorySize, double *errorArray, double *debug) {
	const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int i = row * inputColumns + column;  // change to input; array size is equal to input, however we have more blocks. we calculate i based on input, ignore outside threads
	
	//printf("bx %d by %d tx %d ty %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
	//printf("row %d column %d i %d\n", row, column, i);
	

	extern __shared__ double shared[];
	double *sharedArray = shared;
	double *newSharedArray = &shared[sharedMemorySize];


	const unsigned int sharedDimX = blockDim.x + 2;

	const unsigned int columnShared = threadIdx.x + 1;
	const unsigned int rowShared = threadIdx.y + 1;
	const unsigned int iShared = rowShared * sharedDimX + columnShared; 

	struct Neighbors arrayNeighbors = constructNeighbors(i, inputColumns);
	struct Neighbors sharedArrayNeighbors = constructNeighbors(iShared, sharedDimX);
	
	if (column < inputColumns && row < inputRows) {
		sharedArray[iShared] = array[i];

		int blockIndex = blockIdx.y * sharedMemorySize * gridDim.x + blockIdx.x * sharedMemorySize;
		debug[blockIndex + iShared] = sharedArray[iShared];

		
		// Compute halo points. Set 0 when there isn't neighbor block, or when we're at the end of array within the last block 
		if (!threadIdx.y) { // First row of block
			sharedArray[sharedArrayNeighbors.north] = (blockIdx.y > 0 ? array[arrayNeighbors.north] : 0);
			debug[blockIndex + sharedArrayNeighbors.north] = sharedArray[sharedArrayNeighbors.north];
		}
		if (threadIdx.y == blockDim.y - 1 || row == inputRows - 1) { // Last row of block OR last row of array 
			sharedArray[sharedArrayNeighbors.south] = (blockIdx.y < gridDim.y - 1 && row < inputRows - 1 ? array[arrayNeighbors.south] : 0);
			debug[blockIndex + sharedArrayNeighbors.south] = sharedArray[sharedArrayNeighbors.south];
		}
		if (!threadIdx.x) { // First column of block
			sharedArray[sharedArrayNeighbors.west] = (blockIdx.x > 0 ? array[arrayNeighbors.west] : 0);
			debug[blockIndex + sharedArrayNeighbors.west] = sharedArray[sharedArrayNeighbors.west];
		}
		if (threadIdx.x == blockDim.x - 1 || column == inputColumns - 1) { // Last column of block OR last column of array
			sharedArray[sharedArrayNeighbors.east] = (blockIdx.x < gridDim.x - 1 && column < inputColumns - 1 ? array[arrayNeighbors.east] : 0);
			debug[blockIndex + sharedArrayNeighbors.east] = sharedArray[sharedArrayNeighbors.east];
		}
	}

	__syncthreads();

	// DEBUG:
	// - error
	// - output array on 50 iterations: correct for 16 X 16
	//   incorrect for 840 X 840
	// - if wrong: can check sharedarray 4x4 grid
	// so far have verified sharedArray 1x1 grid with 16*16 input

	// After program is correct, optimize error calculation 

	if (column < inputColumns && row < inputRows) {
		calculateOneElement(row, column, &sharedArrayNeighbors, sharedArray, &newArray[i], jacobiParams, &errorArray[i], inputColumns, inputRows);
	}

	__syncthreads();
}

extern "C" float jacobiGPU(double *array, int elements, int inputRows, int inputColumns, JacobiParams jacobiParams) {
	double *arrayDevice;
	cudaError_t err;
	int arrayBytes = elements * sizeof(double);

	// int blocks = ceil((float) elements / BLOCK_SIZE);
	int rowsOfBlocks, columnsOfBlocks, rowsOfBlockThreads, columnsOfBlockThreads;
	// divide2D(blocks, &rowsOfBlocks, &columnsOfBlocks);
	divide2D(BLOCK_SIZE, &rowsOfBlockThreads, &columnsOfBlockThreads);
	rowsOfBlocks = ceil((float) inputRows / rowsOfBlockThreads);
	columnsOfBlocks = ceil((float) inputColumns / columnsOfBlockThreads);


	int sharedMemorySize = (rowsOfBlockThreads + 2) * (columnsOfBlockThreads + 2);
	
	dim3 dimBl(columnsOfBlockThreads, rowsOfBlockThreads);
	dim3 dimGr(columnsOfBlocks, rowsOfBlocks);

	printf("Elements %d\nGrid %d X %d\nBlock threads %d X %d\n", elements, rowsOfBlocks, columnsOfBlocks, rowsOfBlockThreads, columnsOfBlockThreads);

	double *debug = (double *) malloc(rowsOfBlocks * columnsOfBlocks * sharedMemorySize * sizeof(double ));
	if (debug == NULL) {
		fprintf(stderr, "CPU debug array allocation error\n");
		return -1;
	}

	for (int i = 0 ; i < rowsOfBlocks ; i++) {
		for (int j = 0 ; j < columnsOfBlocks ; j++) {
			int blockIndex = i * sharedMemorySize * columnsOfBlocks + j * sharedMemorySize;
			for (int z = 0 ; z < sharedMemorySize ; z++) {
				debug[blockIndex + z] = -1;
			}
		}
	}

	double *debugDevice;

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

	err = cudaMalloc((void **)&debugDevice, rowsOfBlocks * columnsOfBlocks * sharedMemorySize * sizeof(double));
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	// Copy data to device memory
	err = cudaMemcpy(debugDevice, debug, rowsOfBlocks * columnsOfBlocks * sharedMemorySize * sizeof(double ), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	} // optional

	double *errorArray = (double *) malloc(arrayBytes);
	for (int i = 0 ; i < elements ; i++) {
		errorArray[i] = 0;
	} // optional
	
	double *errorArrayDevice;
	err = cudaMalloc((void **)&errorArrayDevice, arrayBytes);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	// Copy data to device memory
	err = cudaMemcpy(errorArrayDevice, errorArray, arrayBytes, cudaMemcpyHostToDevice);
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
		kjacobi<<<dimGr, dimBl, 2 * sharedMemorySize * sizeof(double)>>>(srcArray, dstArray, elements, inputRows, inputColumns, jacobiParams, sharedMemorySize, errorArrayDevice, debugDevice);
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
		if (iterations) {
			double *tmp = srcArray;
			srcArray = dstArray;
			dstArray = tmp;
		}
	}

	float msecs = getElapsedtime(t_start);

	// Copy results back to host memory
	err = cudaMemcpy(debug, debugDevice, rowsOfBlocks * columnsOfBlocks * sharedMemorySize * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	err = cudaMemcpy(errorArray, errorArrayDevice, rowsOfBlocks * columnsOfBlocks * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	err = cudaMemcpy(array, dstArray, arrayBytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}

	cudaFree(arrayDevice);

	double errorSum = 0;
	for (int i = 0 ; i < elements ; i++) {
		errorSum += errorArray[i];
	}
	double finalError = sqrt(errorSum)/(inputColumns * inputRows);

	printf("Error is %g\n", finalError);

	/*
	for (int i = 0 ; i < rowsOfBlocks ; i++) {
		for (int j = 0 ; j < columnsOfBlocks ; j++) {
			int blockIndex = i * sharedMemorySize * columnsOfBlocks + j * sharedMemorySize;
			for (int z = 0 ; z < sharedMemorySize ; z++) {
				printf("block row %d block column %d element %d: %f\n", i, j, z, debug[blockIndex + z]);
			}
		}
	}*/

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

	//printf("updateVal %.15f cc %.15f f %.15f", updateVal, curVal, f);
	
	//printf("%f %f %f %f\n", sharedArray[neighbors->west], sharedArray[neighbors->east], sharedArray[neighbors->north], sharedArray[neighbors->south]);
	*array = curVal - jacobiParams.relax*updateVal;
	(*errorArray) = updateVal*updateVal;
	//printf("adding %f to %f\n", updateVal*updateVal, errorArray[blockIdx.y * gridDim.x + blockIdx.x]);
	//printf("%.15f\n", *array);
	if (y == 0 && x == 255)
		;//printf("y %d x %d %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n", y, x, updateVal, sharedArray[neighbors->north], sharedArray[neighbors->south], sharedArray[neighbors->west], sharedArray[neighbors->east], curVal, f);

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