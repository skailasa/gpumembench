# Makefile: This file is part of the gpumembench micro-benchmark suite.
#
# Contact: Elias Konstantinidis <ekondis@gmail.com>

include ../common.mk
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops -ffast-math -fassociative-math
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -lrt
NVCODE = -gencode arch=compute_89,code=sm_89 \
         -gencode arch=compute_90,code=sm_90

shmembench: main.o shmem_kernels.o
	${CC} -o $@ $^ ${LFLAGS}

main.o: main.cpp shmem_kernels.h
	${CC} -c ${FLAGS} $<

shmem_kernels.o: shmem_kernels.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $<

clean:
	\rm -f shmembench main.o shmem_kernels.o

rebuild: clean shmembench

