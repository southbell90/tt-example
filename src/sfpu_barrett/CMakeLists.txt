add_executable(sfpu_barrett ${CMAKE_CURRENT_SOURCE_DIR}/sfpu_barrett.cpp)
target_link_libraries(sfpu_barrett PRIVATE TT::Metalium)
target_include_directories(sfpu_barrett PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)