add_executable(sfpu_remainder ${CMAKE_CURRENT_SOURCE_DIR}/sfpu_remainder.cpp)
target_link_libraries(sfpu_remainder PRIVATE TT::Metalium)
target_include_directories(sfpu_remainder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)