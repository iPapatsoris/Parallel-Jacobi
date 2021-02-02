#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	int dim = atoi(argv[1]);
	double array[dim * dim];
	while (fread(array, sizeof(double), dim * dim, stdin));

	for (int i = 0; i < dim; i++) {
		printf("\nLine %d\n", i);
		for (int j = 0 ; j < dim ; j++) {
			printf("%f ", array[dim * i + j]);
		}
	}
	printf("\n");
}
