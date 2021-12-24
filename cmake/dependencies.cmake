find_package(CUDA REQUIRED)
if(${CUDA_VERSION} VERSION_LESS 11.1)
	# message(FATAL_ERROR "CUDA version is too lower(${CUDA_VERSION} vs 11.1)")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(CUDA_INCLUDE_DIRS $ENV{CUDA_PATH}/include)
	set(CUDA_LIBRARY_DIRS $ENV{CUDA_PATH}/lib/x64)
else(GNU OR Clang)
	set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
	set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64)
endif()

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()