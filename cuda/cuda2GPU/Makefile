CUDA_INSTALL_PATH = /usr/local/cuda-11.1
CC = g++
OPTFLAG = -O3 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops -fopenmp
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common/inc/
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR} -fopenmp
NVFLAGS = -O3 -Xcompiler -fopenmp  -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR} 
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
        LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -fopenmp
CLEAN_FILES = jacobi gpu_jacobi.o jacobi.o util.o

jacobi: jacobi.o util.o gpu_jacobi.o
	${CC} ${LFLAGS} -o $@ $^
	cp $@ ../release

util.o: util.cpp
	${CC} -c ${FLAGS} -o $@ $^

jacobi.o: jacobi.cpp
	${CC} -c ${FLAGS} -o $@ $^

gpu_jacobi.o: jacobi.cu
	${NVCC} ${NVFLAGS} -DUNIX -c $^ -o $@

clean:
	\rm -f $(CLEAN_FILES)
	\rm -f ../release/jacobi

rebuild: clean jacobi

