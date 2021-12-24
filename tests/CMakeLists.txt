macro(spnn_add_test name)
    add_executable(test_${name} test_${name}.cpp)
    target_include_directories(test_${name} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_directories(test_${name} PRIVATE ${CUDA_LIBRARY_DIRS})
    target_link_libraries(test_${name} PRIVATE spnnruntime cufft)
    add_test(NAME test_${name} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:test_${name}> -P ${CMAKE_MODULE_PATH}/run_test.cmake)
    # add test to a virtual project group
    set_property(TARGET test_${name} PROPERTY FOLDER "tests")
endmacro()

macro(spnn_add_layer_test class)
    string(TOLOWER ${class} name)
    # enable if WITH_LAYER_xxx option ON
    spnn_add_test(${name})
endmacro()

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

# spnn_add_layer_test(log)
spnn_add_layer_test(pipeline)
# spnn_add_layer_test(async)