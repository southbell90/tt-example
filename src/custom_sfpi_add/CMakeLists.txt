add_executable(custom_sfpi_add ${CMAKE_CURRENT_SOURCE_DIR}/custom_sfpi_add.cpp)
target_link_libraries(custom_sfpi_add PRIVATE TT::Metalium)