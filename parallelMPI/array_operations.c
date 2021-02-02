#include "array_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "util.h"

void initArrays(double *array, double *newArray, const int rows, const int columns) {
    for (int i = 0 ; i < rows ; i++) {
        for (int j = 0 ; j < columns ; j++) {
            array[at(i,j, columns)] = 0;
            newArray[at(i,j, columns)] = 0;
        }
    }
}
