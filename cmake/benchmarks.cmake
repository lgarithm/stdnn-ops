INCLUDE(${CMAKE_SOURCE_DIR}/cmake/gbench.cmake)

ADD_CUSTOM_TARGET(benchmarks)

FUNCTION(ADD_BENCH target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_USE_GBENCH(${target})
    IF(UNIX)
        FIND_PACKAGE(Threads REQUIRED)
        TARGET_LINK_LIBRARIES(${target} Threads::Threads)
    ENDIF()
    ADD_DEPENDENCIES(benchmarks ${target})
    TARGET_USE_STDTENSOR(${target})
    IF(USE_EXTERN)
        TARGET_LINK_LIBRARIES(${target} stdnn-ops)
    ENDIF()
    IF(USE_OPENBLAS)
        TARGET_USE_BLAS(${target})
    ENDIF()
ENDFUNCTION()

FILE(GLOB benchmarks benchmarks/bench_*.cpp)
FOREACH(f ${benchmarks})
    GET_FILENAME_COMPONENT(name ${f} NAME_WE)
    STRING(REPLACE "_" "-" name ${name})
    ADD_BENCH(${name} ${f})
ENDFOREACH()
