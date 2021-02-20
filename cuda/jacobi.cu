#include <lcutil.h>
#include <timestamp.h>
#include <stdio.h>
#include <math.h>
#include "util.h"

#define BLOCK_SIZE 256


__global__ void kjacobi(double *array, int N, int inputRows, int inputColumns) {
	const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int i = row * blockDim.x * gridDim.x + column;  // change to input; array size is equal to input, however we have more blocks. we calculate i based on input, ignore outside threads
	/*if(i < N)
		printf("row %d column %d i %d: %f\n", row, column, i, array[i]);*/

	extern __shared__ double sharedArray[]; 
	const unsigned int sharedDimX = blockDim.x + 2;

	const unsigned int columnShared = threadIdx.x + 1;
	const unsigned int rowShared = threadIdx.y + 1;
	const unsigned int iShared = rowShared * sharedDimX + columnShared; 

	printf("i row %d i column %d i:%d iShared row %d iShared column %d iShared %d\n", row, column, i, rowShared, columnShared, iShared);

	if (column < inputColumns && row < inputRows && i < N) // add check for end of array before end of block
		sharedArray[iShared] = array[i];
/*
	if (!threadIdx.y) {
		sharedArray[iShared - sharedDimX] = (blockIdx.y > 0 ? array[i - blockDim.x * gridDim.x] : 0);
	}
	if (threadIdx.y == blockDim.y - 1) {
		sharedArray[iShared + sharedDimX] = (blockIdx.y < gridDim.y ? array[i + blockDim.x * gridDim.x] : 0);
	}
	if (!threadIdx.x) {
		sharedArray[iShared - 1] = (blockIdx.x > 0 ? array[i - 1] : 0);
	}
	if (threadIdx.x == blockDim.x - 1) {
		sharedArray[iShared + 1] = (blockIdx.x < blockDim.x ? array[i + 1] : 0);
	}*/
	
	__syncthreads();

	/*if(i < N)
		printf("row %d column %d i %d: %f\n", row, column, i, array[i]);*/

}

extern "C" float jacobiGPU(double *array, int elements, int inputRows, int inputColumns) {
	double *arrayDevice;
	cudaError_t err;
	int arrayBytes = elements * sizeof(double);

	int blocks = elements / BLOCK_SIZE;
	int rowsOfBlocks, columnsOfBlocks, rowsOfBlockThreads, columnsOfBlockThreads;
	divide2D(blocks, &rowsOfBlocks, &columnsOfBlocks);
	divide2D(BLOCK_SIZE, &rowsOfBlockThreads, &columnsOfBlockThreads);

	int sharedMemorySize = (rowsOfBlockThreads + 2) * (columnsOfBlockThreads + 2);
	
	
	//int netiBlocks = FRACTION_CEILING(elements, BLOCK_SIZE);
	//divide2D(netiBlocks, &rowsOfBlocks, &columnsOfBlocks);
	
	
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

	timestamp t_start;
	t_start = getTimestamp();

	kjacobi<<<dimGr, dimBl, sharedMemorySize * sizeof(double)>>>(arrayDevice, elements, inputRows, inputColumns);
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

	float msecs = getElapsedtime(t_start);

	// Copy results back to host memory
	/*err = cudaMemcpy(c, cd, sizeof(float)*N, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n",err);
		return err;
	}*/

	cudaFree(arrayDevice);
	return msecs;
}