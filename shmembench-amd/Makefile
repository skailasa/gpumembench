include ../common.mk
ROCM_PATH ?= /opt/rocm
HIPCC     := ${ROCM_PATH}/bin/hipcc
HIPCODE   := --offload-arch=gfx1100
OPTFLAGS  := -O2 -fomit-frame-pointer -funroll-loops -ffast-math -fassociative-math
INCDIR    := -I../common -I${ROCM_PATH}/include
HIPFLAGS  := ${OPTFLAGS} ${INCDIR} -Wall -g -fno-strict-aliasing -DUNIX
LFLAGS    := -L${ROCM_PATH}/lib${LIBSUFFIX} -lamdhip64 -lm -lstdc++ -lrt

all: shmembench

shmembench: main.o shmem_kernels.o
	@echo "Linking $@ ..."
	${HIPCC} ${HIPCODE} -o $@ $^ ${LFLAGS} -no-pie

main.o: main.cpp shmem_kernels.h
	${HIPCC} ${HIPCODE} ${HIPFLAGS} -c $< -o $@

shmem_kernels.o: shmem_kernels.cpp
	${HIPCC} ${HIPCODE} ${HIPFLAGS} -c $< -o $@

clean:
	@echo "Cleaning..."
	rm -f shmembench main.o shmem_kernels.o

rebuild: clean all
