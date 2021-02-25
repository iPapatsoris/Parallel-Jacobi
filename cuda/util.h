struct JacobiParams {
	int maxIterations, inputRows, inputColumns;
	bool checkConvergence;
	double alpha, relax, tol;
};

void divide2D(int total, int *rows, int *columns);
void parseInput(struct JacobiParams *jacobiParams);
void initArray(double *array, int size);
void swap(double **a, double **b);