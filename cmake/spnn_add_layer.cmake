macro(spnn_add_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    if(SPNN_CMAKE_VERBOSE)
        message(STATUS "WITH_LAYER_${name} = ${WITH_LAYER_${name}}")
    endif()

    if(WITH_LAYER_${name})
        list(APPEND spnnruntime_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp)
        list(APPEND spnnruntime_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/${name}.cu)
    endif()

    if(WITH_LAYER_${name})
        set(layer_registry "${layer_registry} {\"${class}\", ${class}LayerCreator},\n")
    else()
        set(layer_registry "${layer_registry}{\"${class}\", 0},\n")
    endif()

    if(WITH_LAYER_${name})
        set(layer_declaration "${layer_declaration} Layer *${class}LayerCreator(void *); \n")
    else()
        set(layer_declaration "${layer_declaration}")
    endif()

    # generate layer_type_enum file
    set(layer_type_enum "${layer_type_enum}${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()
