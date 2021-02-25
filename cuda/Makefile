CUDA_INSTALL_PATH = /usr/local/cuda-11.1
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -Icommon/inc/
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
        LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart
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

