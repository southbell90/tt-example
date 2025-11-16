#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t src3_addr = get_arg_val<uint32_t>(3);
    uint32_t src4_addr = get_arg_val<uint32_t>(4);
    uint32_t src5_addr = get_arg_val<uint32_t>(5);
    uint32_t Nt = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;
    constexpr uint32_t cb_id_in3 = 3;
    constexpr uint32_t cb_id_in4 = 4;
    constexpr uint32_t cb_id_in5 = 5;

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr, get_tile_size(cb_id_in0));
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr, get_tile_size(cb_id_in1));
    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    const auto s2 = TensorAccessor(s2_args, src2_addr, get_tile_size(cb_id_in2));
    constexpr auto s3_args = TensorAccessorArgs<s2_args.next_compile_time_args_offset()>();
    const auto s3 = TensorAccessor(s3_args, src3_addr, get_tile_size(cb_id_in3));
    constexpr auto s4_args = TensorAccessorArgs<s3_args.next_compile_time_args_offset()>();
    const auto s4 = TensorAccessor(s4_args, src4_addr, get_tile_size(cb_id_in4));
    constexpr auto s5_args = TensorAccessorArgs<s4_args.next_compile_time_args_offset()>();
    const auto s5 = TensorAccessor(s5_args, src5_addr, get_tile_size(cb_id_in5));

    for(uint32_t i = 0; i < Nt; i++) {
                                        
        cb_reserve_back(cb_id_in0, 1);
        cb_reserve_back(cb_id_in1, 1);
        cb_reserve_back(cb_id_in2, 1);
        cb_reserve_back(cb_id_in3, 1);
        cb_reserve_back(cb_id_in4, 1);
        cb_reserve_back(cb_id_in5, 1);

        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3);
        uint32_t l1_write_addr_in4 = get_write_ptr(cb_id_in4);
        uint32_t l1_write_addr_in5 = get_write_ptr(cb_id_in5);

        noc_async_read_tile(i, s0, l1_write_addr_in0);
        noc_async_read_tile(i, s1, l1_write_addr_in1);
        noc_async_read_tile(i, s2, l1_write_addr_in2);
        noc_async_read_tile(i, s3, l1_write_addr_in3);
        noc_async_read_tile(i, s4, l1_write_addr_in4);
        noc_async_read_tile(i, s5, l1_write_addr_in5);

        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
        cb_push_back(cb_id_in1, 1);
        cb_push_back(cb_id_in2, 1);
        cb_push_back(cb_id_in3, 1);
        cb_push_back(cb_id_in4, 1);
        cb_push_back(cb_id_in5, 1);

    }
}