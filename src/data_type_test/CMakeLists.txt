add_executable(data_type_test ${CMAKE_CURRENT_SOURCE_DIR}/data_type_test.cpp)
target_link_libraries(data_type_test PRIVATE TT::Metalium)
target_include_directories(data_type_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)