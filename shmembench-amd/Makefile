include ../common.mk
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops -ffast-math -fassociative-math
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common
FLAGS = ${OPTFLAG} -I${ROCM_INSTALL_PATH}/include -Wall -g ${INCDIR}
HIPFLAGS = -O2 -I${ROCM_PATH}/include ${INCDIR} \
           -fno-strict-aliasing -g

LFLAGS = -L${ROCM_PATH}/lib${LIBSUFFIX} \
             -lm -lstdc++ -lamdhip64 -lrt

