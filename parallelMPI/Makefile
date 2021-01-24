OBJS =  util.o array_operations.o jacobiMPI.o
HEADER = util.h array_operations.h

CC 	= mpicc
MPIP = -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind
FLAGS	= -g -O3 ${MPIP} -c 

all: $(OBJS) $(HEADER)
	$(CC) -O3 -lm ${MPIP} -o jacobiMPI.x $(OBJS)

util.o: util.c
	$(CC) $(FLAGS) util.c 

array_operations.o: array_operations.c
	$(CC) $(FLAGS) array_operations.c 

jacobiMPI.o: jacobiMPI.c
	$(CC) $(FLAGS) jacobiMPI.c
	
clean:
	rm -f all $(OBJS)