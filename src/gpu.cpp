#include "gpu.h"

namespace spnnruntime
{
    int get_cuda_device_count()
    {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        return deviceCount;
    }

    std::string get_cuda_device_name(int dev)
    {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        return std::string(deviceProp.name);
    }

    int get_cuda_device_driver_version(int dev)
    {
        int driverVersion;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
        return driverVersion;
    }

    int get_cuda_device_runtime_version(int dev)
    {
        int runtimeVersion;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
        return runtimeVersion;
    }

    int get_cuda_device_capability(int dev)
    {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        return deviceProp.major * 10 + deviceProp.minor;
    }

    void get_cuda_device_memory(int dev, std::size_t &total_size, std::size_t &free_size)
    {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaMemGetInfo(&free_size, &total_size));
        total_size /= 1048576;
        free_size /= 1048576;
    }

    int get_cuda_device_cuda_core_num(int dev)
    {
        /// TODO: _ConvertSMVer2Cores should be re-write
        //CUDA_CHECK(cudaSetDevice(dev));
        //cudaDeviceProp deviceProp;
        //cudaGetDeviceProperties(&deviceProp, dev);
        //return _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
        return -1;
    }

    // CUFFT API errors
    const char *cudaGetErrorEnum(cufftResult error)
    {
        switch (error)
        {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";

        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
        }

        return "<unknown>";
    }
} // namespace spnnruntime
