add_executable(matmul_multi_core ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multi_core.cpp)
target_link_libraries(matmul_multi_core PRIVATE TT::Metalium)
target_include_directories(matmul_multi_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)