add_executable(noc_tile_transfer ${CMAKE_CURRENT_SOURCE_DIR}/noc_tile_transfer.cpp)
target_link_libraries(noc_tile_transfer PRIVATE TT::Metalium)
target_include_directories(noc_tile_transfer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_common)