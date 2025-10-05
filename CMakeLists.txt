cmake_minimum_required(VERSION 3.24)
project(tt-example LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPM_SOURCE_CACHE ${CMAKE_BINARY_DIR}/cpm_modules)
include(cpm/get_cpm.cmake)

CPMAddPackage(
    NAME tt-metal
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/../tt-metal # tt-metal 폴더의 상대 경로
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
