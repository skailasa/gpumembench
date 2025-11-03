#ifndef _HIPUTIL_H_
#define _HIPUTIL_H_

#include <stdio.h>
#include <hip/hip_runtime.h>


#define HIP_SAFE_CALL(call) {                                               \
    hipError_t err = call;                                                  \
    if (err != hipSuccess) {                                                \
        fprintf(stderr, "HIP error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, hipGetErrorString(err));                \
        exit(EXIT_FAILURE);                                                 \
    } }

#define FRACTION_CEILING(numerator, denominator) (((numerator)+(denominator)-1)/(denominator))

static inline int ConvertCUToCores(const hipDeviceProp_t *prop) {
    // For ROCm, "multiProcessorCount" = number of Compute Units
    return prop->multiProcessorCount * 64;
}

// Return theoretical peak limits in compute and memory bandwidth from cuda property information
static inline void GetDevicePeakInfoHIP(double *aGIPS, double *aGBPS, hipDeviceProp_t *aDeviceProp = NULL) {
    hipDeviceProp_t deviceProp;
    int current_device = 0;

    if (aDeviceProp)
        deviceProp = *aDeviceProp;
    else {
        HIP_SAFE_CALL(hipGetDevice(&current_device));
        HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, current_device));
    }

    int coreClock = 0;
    int memClock = 0;
    int memBusWidth = 0;

    HIP_SAFE_CALL(hipDeviceGetAttribute(&coreClock, hipDeviceAttributeClockRate, current_device));
    HIP_SAFE_CALL(hipDeviceGetAttribute(&memClock, hipDeviceAttributeMemoryClockRate, current_device));
    HIP_SAFE_CALL(hipDeviceGetAttribute(&memBusWidth, hipDeviceAttributeMemoryBusWidth, current_device));

    const int TotalSPs = ConvertCUToCores(&deviceProp);
    *aGIPS = 1000.0 * coreClock * TotalSPs / 1.0e9;  // Giga-instructions/sec
    *aGBPS = 2.0 * (double)memClock * 1000.0 * (double)memBusWidth / 8.0; // GB/s
}


// Return HIP device properties
static inline hipDeviceProp_t GetDevicePropertiesHIP(void) {
    hipDeviceProp_t deviceProp;
    int current_device = 0;
    HIP_SAFE_CALL(hipGetDevice(&current_device));
    HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, current_device));
    return deviceProp;
}


// Produce basic device information output to a file stream
static inline void StoreDeviceInfoHIP(FILE *fout) {
    hipDeviceProp_t deviceProp;
    int current_device, driver_version;
    HIP_SAFE_CALL(hipGetDevice(&current_device));
    HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, current_device));
    HIP_SAFE_CALL(hipDriverGetVersion(&driver_version));

    int coreClock = 0;
    int memClock = 0;
    int memBusWidth = 0;
    HIP_SAFE_CALL(hipDeviceGetAttribute(&coreClock, hipDeviceAttributeClockRate, current_device));
    HIP_SAFE_CALL(hipDeviceGetAttribute(&memClock, hipDeviceAttributeMemoryClockRate, current_device));
    HIP_SAFE_CALL(hipDeviceGetAttribute(&memBusWidth, hipDeviceAttributeMemoryBusWidth, current_device));

    fprintf(fout, "------------------------ Device specifications ------------------------\n");
    fprintf(fout, "Device:              %s\n", deviceProp.name);
    fprintf(fout, "ROCm driver version: %d.%d\n", driver_version/1000, driver_version%1000);
    fprintf(fout, "GPU clock rate:      %d MHz\n", coreClock / 1000);
    fprintf(fout, "Memory clock rate:   %d MHz\n", memClock / 1000 / 2);
    fprintf(fout, "Memory bus width:    %d bits\n", memBusWidth);
    fprintf(fout, "Wavefront size:      %d\n", deviceProp.warpSize);
    fprintf(fout, "L2 cache size:       %d KB\n", deviceProp.l2CacheSize / 1024);
    fprintf(fout, "Total global mem:    %d MB\n", (int)(deviceProp.totalGlobalMem / 1024 / 1024));
    fprintf(fout, "Compute Units:       %d\n", deviceProp.multiProcessorCount);

    const int TotalSPs = ConvertCUToCores(&deviceProp);
    fprintf(fout, "Total SPs:           %d (%d CUs x 64 SPs/CU)\n",
            TotalSPs, deviceProp.multiProcessorCount);

    double InstrThroughput, MemBandwidth;
    GetDevicePeakInfoHIP(&InstrThroughput, &MemBandwidth, &deviceProp);
    fprintf(fout, "Compute throughput:  %.2f GFlops (theoretical single precision FMAs)\n",
            2.0 * InstrThroughput);
    fprintf(fout, "Memory bandwidth:    %.2f GB/sec\n",
            MemBandwidth / (1000.0 * 1000.0 * 1000.0));
    fprintf(fout, "-----------------------------------------------------------------------\n");
}
#endif