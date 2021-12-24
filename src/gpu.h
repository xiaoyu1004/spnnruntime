#ifndef GPU_HPP
#define GPU_HPP

#include "log.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

namespace spnnruntime
{
    using ComplexType = cufftDoubleComplex;

    // CUDA: use 512 threads per block
    const std::uint32_t CUDA_NUM_THREADS = 256;

    // CUDA: number of blocks for threads.
    inline std::uint32_t CUDA_GET_BLOCKS(const std::uint32_t N)
    {
        return (N - 1) / CUDA_NUM_THREADS + 1;
    }

    /// <summary>
    /// Get the number of CUDA support compute-capable devices
    /// </summary>
    /// <returns>The number of devices</returns>
    int get_cuda_device_count();

    /// <summary>
    /// Get the ASCII string identifying of the device_id
    /// </summary>
    /// <param name="device_id">The 0-indexed id of the cuda device</param>
    /// <returns>The number of compute-capable devices</returns>
    std::string get_cuda_device_name(int device_id);

    /// <summary>
    /// Get the latest version of CUDA supported by the driver.
    /// The version is returned as (1000 * major + 10 * minor).
    /// For example, CUDA 9.2 would be represented by 9020.
    /// If no driver is installed, then 0 is returned as the driver version.
    /// </summary>
    /// <param name="device_id">The 0-indexed id of the cuda device</param>
    /// <returns>Returns the CUDA driver version.</returns>
    int get_cuda_device_driver_version(int device_id);

    /// <summary>
    /// Get the version number of the current CUDA Runtime instance
    /// </summary>
    /// <param name="device_id">The 0-indexed id of the cuda device</param>
    /// <returns>Returns the CUDA Runtime version.</returns>
    int get_cuda_device_runtime_version(int device_id);

    /// <summary>
    /// Get are the major and minor revision numbers defining the device's compute capability.
    /// </summary>
    /// <param name="device_id">The 0-indexed id of the cuda device</param>
    /// <returns>Returns 10 * major + minor</returns>
    int get_cuda_device_capability(int device_id);

    /// <summary>
    /// Returns free and total respectively, the free and total amount of  memory available for allocation by the device in Mbytes.
    /// </summary>
    /// <param name="device_id">The 0-indexed id of the cuda device</param>
    /// <param name="total_size">Total memory in Mb</param>
    /// <param name="free_size">Free memory in Mb</param>
    void get_cuda_device_memory(int device_id, std::size_t &total_size, std::size_t &free_size);

    int get_cuda_device_cuda_core_num(int device_id);

    const char *cudaGetErrorEnum(cufftResult error);

#define GET_THREAD_ID() (std::uint32_t)(blockDim.x * blockIdx.x + threadIdx.x)

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                             \
    /* Code block avoids redefinition of cudaError_t error */             \
    do                                                                    \
    {                                                                     \
        cudaError_t error = condition;                                    \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

#define CUDA_DRIVER_CHECK(condition)                                       \
    /* Code block avoids redefinition of cudaError_t error */              \
    do                                                                     \
    {                                                                      \
        cudaError_enum error = condition;                                  \
        CHECK_EQ(error, CUDA_SUCCESS) << " " << cudaGetErrorString(error); \
    } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define CUFFT_CHECK(condition)                                            \
    do                                                                    \
    {                                                                     \
        cufftResult error = condition;                                    \
        CHECK_EQ(error, CUFFT_SUCCESS) << " " << cudaGetErrorEnum(error); \
    } while (0)
} // spnnruntime

#endif