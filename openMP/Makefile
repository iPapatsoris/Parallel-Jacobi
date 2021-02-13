OBJS =  util.o array_operations.o jacobiOpenMP.o
HEADER = util.h array_operations.h

CC 	= mpicc
MPIP = -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind -fopenmp
FLAGS	= -g -O3 ${MPIP} -c 

all: jacobi read_array

read_array: read_array.c
	gcc read_array.c -o read_array

jacobi: $(OBJS) $(HEADER)
	$(CC) -O3 -lm ${MPIP} -o jacobiOpenMP.x $(OBJS)

util.o: util.c
	$(CC) $(FLAGS) util.c 

array_operations.o: array_operations.c
	$(CC) $(FLAGS) array_operations.c 

jacobiOpenMP.o: jacobiOpenMP.c
	$(CC) $(FLAGS) jacobiOpenMP.c
	
clean:
	rm -f all $(OBJS)