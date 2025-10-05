add_executable(loopback ${CMAKE_CURRENT_SOURCE_DIR}/loopback.cpp)
target_link_libraries(loopback PRIVATE TT::Metalium)