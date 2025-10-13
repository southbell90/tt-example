add_executable(eltwise_sfpu ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_sfpu.cpp)
target_link_libraries(eltwise_sfpu PRIVATE TT::Metalium)