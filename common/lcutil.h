/**
 * lcutil.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#ifndef _CUTIL_H_
#define _CUTIL_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/(denominator))

static inline int _ConvertSMVer2Cores(int major, int minor){
	switch(major){
		case 1:  return 8;
		case 2:  switch(minor){
			case 1:  return 48;
			default: return 32;
		}
		case 3:  return 192;
		case 6:  switch(minor){
			case 0:  return 64;
			default: return 128;
		}
		default: return 128;
	}
}

// Return theoretical peak limits in compute and memory bandwidth from cuda property information
static inline void GetDevicePeakInfo(double *aGIPS, double *aGBPS, cudaDeviceProp *aDeviceProp = NULL){
	cudaDeviceProp deviceProp;
	int current_device;
	if( aDeviceProp )
		deviceProp = *aDeviceProp;
	else{
		CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	}

    int coreClock = 0;
    int memClock = 0;
    int memBusWidth = 0;

    cudaDeviceGetAttribute(&coreClock, cudaDevAttrClockRate, current_device);
    cudaDeviceGetAttribute(&memClock, cudaDevAttrMemoryClockRate, current_device);
    cudaDeviceGetAttribute(&memBusWidth, cudaDevAttrGlobalMemoryBusWidth, current_device);

	const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
    *aGIPS = 1000.0 * coreClock * TotalSPs / 1.0e9;                     // Giga-instructions/sec
    *aGBPS = 2.0 * (double)memClock * 1000.0 * (double)memBusWidth / 8.0; // GBytes/sec

}

// Get CUDA device properties
static inline cudaDeviceProp GetDeviceProperties(void){
	cudaDeviceProp deviceProp;
	int current_device;
	CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	return deviceProp;
}

// Produce basic device information output to a file stream
static void StoreDeviceInfo(FILE *fout){
	cudaDeviceProp deviceProp;
	int current_device, driver_version;
	CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	CUDA_SAFE_CALL( cudaDriverGetVersion(&driver_version) );

    // Get attributes for CUDA 12+
    int coreClock = 0;
    int memClock = 0;
    int memBusWidth = 0;

    cudaDeviceGetAttribute(&coreClock, cudaDevAttrClockRate, current_device);
    cudaDeviceGetAttribute(&memClock, cudaDevAttrMemoryClockRate, current_device);
    cudaDeviceGetAttribute(&memBusWidth, cudaDevAttrGlobalMemoryBusWidth, current_device);


	fprintf(fout, "------------------------ Device specifications ------------------------\n");
	fprintf(fout, "Device:              %s\n", deviceProp.name);
	fprintf(fout, "CUDA driver version: %d.%d\n", driver_version/1000, driver_version%1000);
    fprintf(fout, "GPU clock rate:      %d MHz\n", coreClock/1000);
    fprintf(fout, "Memory clock rate:   %d MHz\n", memClock/1000/2);
    fprintf(fout, "Memory bus width:    %d bits\n", memBusWidth);
	fprintf(fout, "WarpSize:            %d\n", deviceProp.warpSize);
	fprintf(fout, "L2 cache size:       %d KB\n", deviceProp.l2CacheSize/1024);
	fprintf(fout, "Total global mem:    %d MB\n", (int)(deviceProp.totalGlobalMem/1024/1024));
	fprintf(fout, "ECC enabled:         %s\n", deviceProp.ECCEnabled?"Yes":"No");
	fprintf(fout, "Compute Capability:  %d.%d\n", deviceProp.major, deviceProp.minor);
	const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
	fprintf(fout, "Total SPs:           %d (%d MPs x %d SPs/MP)\n", TotalSPs, deviceProp.multiProcessorCount, _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
	double InstrThroughput, MemBandwidth;
	GetDevicePeakInfo(&InstrThroughput, &MemBandwidth, &deviceProp);
	fprintf(fout, "Compute throughput:  %.2f GFlops (theoretical single precision FMAs)\n", 2.0*InstrThroughput);
	fprintf(fout, "Memory bandwidth:    %.2f GB/sec\n", MemBandwidth/(1000.0*1000.0*1000.0));
	fprintf(fout, "-----------------------------------------------------------------------\n");
}

#endif
