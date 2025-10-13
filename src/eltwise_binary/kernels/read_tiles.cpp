#include <cstdint>

/*
    reader kernel은 2개의 DRAM의 input buffers에서 데이터를 읽어 circular buffers에 집어넣는다.

*/
void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    // The circular buffers to read the tiles into (same as the ones in the host program)
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    // circular buffers에서 사용되는 tile size를 얻는다.
    // circular buffers의 tile size는 DRAM buffer의 tile size와 같다고 가정한다.
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    /*
        input buffers를 위한 address generators를 만든다.
        이것들은 interleaved buffers의 pointer라고 생각하면 된다.
        page size는 tile size로 맞춘다.
    */
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_size_bytes);

    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto in1 = TensorAccessor(in1_args, in1_addr, tile_size_bytes);

    for (uint32_t i = 0; i < n_tiles; i++) {
        //먼저 circular buffes에 공간이 있다는 것을 확인해야 된다.
        //cb_reserve_back(cb_id, num_tiles)
        //blocking으로 num_tiles만큼의 타일이 circular buffer에서 free가 될때까지 기다린다.
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);


        noc_async_read_tile(i, in0, get_write_ptr(cb_in0));
        noc_async_read_tile(i, in1, get_write_ptr(cb_in1));

        // Wait until tile reads are done
        noc_async_read_barrier();
        // cb_push_back(cb_id, num_tiles)
        // num_tiles만큼을 CB's queue에다가 push 한다.circular buffer의 available space가 줄어든다.
        // 이 함수는 producer가 CB의 consumer에게 tile이 접근 가능하게 만들어 준다. (visible)
        // forward kernels에서 cb_wait_front 함수를 호출하면 이 tile이 보이게 된다.
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);
    }
}