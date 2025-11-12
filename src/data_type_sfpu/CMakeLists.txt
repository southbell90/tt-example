add_executable(data_type_sfpu ${CMAKE_CURRENT_SOURCE_DIR}/data_type_sfpu.cpp)
target_link_libraries(data_type_sfpu PRIVATE TT::Metalium)
target_include_directories(data_type_sfpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)