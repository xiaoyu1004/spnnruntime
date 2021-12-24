configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SPNNDefine.h.in ${CMAKE_CURRENT_BINARY_DIR}/SPNNDefine.h)

set(spnnruntime_SRCS 
        allocator.cpp
        blob.cpp
        datareader.cpp
        extractor.cpp
        layer.cpp
        log.cpp
        modelbin.cpp
        paramdict.cpp
        pipeline.cpp
        singleton.hpp
        tensor.cpp
        option.cpp
        gpu.cpp
        syncedmem.cpp
        setting/radar.cpp
        command.cpp)

include(${CMAKE_MODULE_PATH}/spnn_add_layer.cmake)

set(__LAYER_TYPE_ENUM_INDEX 0)

# layer implementation
spnn_add_layer(Input)
spnn_add_layer(AbsVal)
spnn_add_layer(RemoveDC)
spnn_add_layer(PulseCompress)
spnn_add_layer(Hamming)
spnn_add_layer(MoveDetector)
spnn_add_layer(PaddingMirror)
spnn_add_layer(Integral)
spnn_add_layer(CfarOp)
spnn_add_layer(BinaryOp)
spnn_add_layer(ClusterOp)

# exclude clusterop
list(REMOVE_ITEM spnnruntime_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clusterop.cu)

configure_file(layer_declaration.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
configure_file(layer_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
configure_file(layer_type_enum.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)

if(SPNN_BUILD_SHARED)
    cuda_add_library(spnnruntime SHARED ${spnnruntime_SRCS})
else()
    cuda_add_library(spnnruntime STATIC ${spnnruntime_SRCS})
endif()
set_target_properties(spnnruntime PROPERTIES DEBUG_POSTFIX "d")

target_include_directories(spnnruntime
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

target_link_directories(spnnruntime PRIVATE ${CUDA_LIBRARY_DIRS})

target_link_libraries(spnnruntime cufft)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_definitions(spnnruntime PRIVATE -DEXPORT_SPNNRUNTIME)
endif()