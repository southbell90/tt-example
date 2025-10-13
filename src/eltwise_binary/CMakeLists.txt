add_executable(eltwise_binary ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_binary.cpp)
target_link_libraries(eltwise_binary PRIVATE TT::Metalium)