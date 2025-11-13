add_executable(matmul_uint8 ${CMAKE_CURRENT_SOURCE_DIR}/matmul_uint8.cpp)
target_link_libraries(matmul_uint8 PRIVATE TT::Metalium)
target_include_directories(matmul_uint8 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)