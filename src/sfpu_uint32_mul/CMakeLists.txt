add_executable(sfpu_uint32_mul ${CMAKE_CURRENT_SOURCE_DIR}/sfpu_uint32_mul.cpp)
target_link_libraries(sfpu_uint32_mul PRIVATE TT::Metalium)
target_include_directories(sfpu_uint32_mul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)