add_executable(sfpu_right_shift ${CMAKE_CURRENT_SOURCE_DIR}/sfpu_right_shift.cpp)
target_link_libraries(sfpu_right_shift PRIVATE TT::Metalium)
target_include_directories(sfpu_right_shift PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)