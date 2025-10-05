// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Read parameters from the kernel arguments
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);

    // Address and the DRAM bank ID of the source buffer
    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);

    // Address and the DRAM bank ID of the destination buffer
    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(2);

    // Size of the buffer in bytes
    std::uint32_t num_tiles = get_arg_val<uint32_t>(3);

    // Each tile is 32x32 elements of bfloat16, which is 2 bytes per element.
    // So the tile size in bytes is 32 * 32 * 2 = 2048 bytes.
    // Note that this is the same as the tile size used in the host code
    // when creating the buffers.
    const uint32_t tile_size_bytes = 32 * 32 * 2;
    constexpr auto in0_args = TensorAccessorArgs<0>();

    // TensorAccessor 객체는 bank addressing과 page size를 자동으로 다룬다.
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr, tile_size_bytes);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr, tile_size_bytes);

    /*
        데이터의 이동은 비동기적으로 일어나게 돼서 kernel이 multiple request를 날릴 수 있다.
        barrier 함수가 필요한 이유가 이것 때문이다.
    */
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Issue a read to the NoC and write to the L1 buffer. This operation is asynchronous.
        // thus a barrier is needed to ensure that the read is complete before the write.
        noc_async_read_tile(i, in0, l1_buffer_addr);
        noc_async_read_barrier();
        // Write back the tile to the destination DRAM buffer.
        // Again, this is an asynchronous operation, so we need a barrier to ensure the write
        // is complete before the next iteration.
        noc_async_write_tile(i, out0, l1_buffer_addr);
        noc_async_write_barrier();
    }
}