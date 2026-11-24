cmake_minimum_required(VERSION 3.16..3.31)
project(rena LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RENA_BUILD_TESTS "Build test executables" ON)

add_subdirectory(third_party)

add_compile_options(-Wall -Wextra)

add_subdirectory(core)
add_subdirectory(classifier_backend)

if(RENA_BUILD_TESTS)
    add_subdirectory(test)
endif()
