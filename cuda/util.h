struct JacobiParams {
	int maxIterations, inputRows, inputColumns;
	bool checkConvergence;
	double alpha, relax, tol;
};

struct JacobiConstants {
	double deltaX;
	double deltaY;
	double cx;
	double cy;
	double cc;
};

void divide2D(int total, int *rows, int *columns);
void parseInput(struct JacobiParams *jacobiParams);
void initArray(double *array, int size);
void swap(double **a, double **b);
JacobiConstants getJacobiConstants(int xLeft, int xRight, int yBottom, int yTop, int inputRows, int inputColumns, double alpha);