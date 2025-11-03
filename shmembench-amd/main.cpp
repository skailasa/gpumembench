#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <lhiputil.h>
#include "shmem_kernels.h"


#define VECTOR_SIZE (1024*1024)
int main(int argc, char* argv[]) {

    printf("AMD shmembench (shared memory bandwidth microbenchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(double);

	HIP_SAFE_CALL(hipSetDevice(0)); // set first device as default

	StoreDeviceInfoHIP(stdout);

	size_t freeHIPMem, totalHIPMem;
    HIP_SAFE_CALL(hipMemGetInfo(&freeHIPMem, &totalHIPMem));
	printf("Total GPU memory %lu, free %lu\n", totalHIPMem, freeHIPMem);

	printf("Buffer sizes: %dMB\n", datasize/(1024*1024));

	double *c;
	c = (double*)malloc(datasize);
	memset(c, 0, sizeof(int)*VECTOR_SIZE);

	// benchmark execution
	shmembenchGPU(c, VECTOR_SIZE);

	free(c);

	return 0;
}