add_executable(matmul_single_core ${CMAKE_CURRENT_SOURCE_DIR}/matmul_single_core.cpp)
target_link_libraries(matmul_single_core PRIVATE TT::Metalium)
target_include_directories(matmul_single_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)