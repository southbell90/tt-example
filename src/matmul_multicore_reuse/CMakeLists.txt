add_executable(matmul_multicore_reuse ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse.cpp)
target_link_libraries(matmul_multicore_reuse PRIVATE TT::Metalium)
target_include_directories(matmul_multicore_reuse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)