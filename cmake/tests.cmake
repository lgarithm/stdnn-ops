INCLUDE(${CMAKE_SOURCE_DIR}/cmake/gtest.cmake)

FUNCTION(ADD_GTEST target)
    ADD_EXECUTABLE(${target} ${ARGN} tests/main.cpp)
    TARGET_USE_GTEST(${target})
    TARGET_USE_STDTENSOR(${target})
    IF(USE_EXTERN)
        TARGET_LINK_LIBRARIES(${target} stdnn-ops)
    ENDIF()
    TARGET_INCLUDE_DIRECTORIES(${target}
                               PRIVATE ${CMAKE_SOURCE_DIR}/tests/include)
    ADD_TEST(NAME ${target} COMMAND ${target})
ENDFUNCTION()

FILE(GLOB tests tests/test_*.cpp)
FOREACH(t ${tests})
    GET_FILENAME_COMPONENT(name ${t} NAME_WE)
    STRING(REPLACE "_"
                   "-"
                   name
                   ${name})
    ADD_GTEST(${name} ${t})
ENDFOREACH()
