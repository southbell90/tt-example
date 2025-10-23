#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    /*
        Tensix core의 compute unit은 한 번에 최대 8개의 타일을 처리할 수 있기 때문에
        output block이 8 타일보다 크면 이를 sub-block으로 나눠 처리하는게 효율적이다
        이 예제에서도 타일의 크기가 8인 sub-block으로 나누어 처리한다.
        본 예제에서 out_subblock_h = 4, out_subblock_w = 2 가 된다.
    */
    uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);        // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);        // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);           // out_subblock_w*in1_num_subblocks
    uint32_t num_blocks = get_compile_time_arg_val(7);               // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(8);           // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(9);           // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);  // out_subblock_h * out_subblock_w;
    uint32_t batch = get_compile_time_arg_val(11);                   // batch dim

    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
            cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    if (enable_reload) {
                        // bring prior partial C tiles back from c_24
                        copy_tile_to_dst_init_short(tt::CBIndex::c_24);
                        cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(tt::CBIndex::c_24, i, i);     // reload partials into dst registers
                        }
                        cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                        mm_init_short(tt::CBIndex::c_0, tt::CBIndex::c_1);
                    }

                    // Compute output sub-block from in0_subblock x in1_subblock
                    /*
                        (모든 개수는 타일 단위)
                        output block size = 20 * 2
                        output sub-block size = 4*2 = 8
                        output block 하나당 sub-block의 개수는 5개
                        아래의 for문은 sub-block 1개에서의 데이터를 읽어온다.

                        위의 in0_subblock, in1_subblock 2중 for문은 block내의 sub-block 5개를 도는 for문이다.
                    */ 
                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                matmul_tiles(           // dst register에 값을 누적해서 더한다. (overwrite 하는게 아님)
                                    tt::CBIndex::c_0,
                                    tt::CBIndex::c_1,
                                    in0_index,
                                    in1_index,
                                    dst_index,
                                    false /* transpose */);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    if (last_out) {
                        // Pack out to output buffer
                        cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, tt::CBIndex::c_16);
                        }
                        cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                    } else {
                        // Wait for tiles in output buffer to be written out since interm and output share memory
                        if (block == 0) {
                            cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Move partial result to interm buffer
                        cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, tt::CBIndex::c_24);
                        }
                        cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                    }

                    release_dst();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            /*
                k-block의 개수가 2개 이상이면 spill은 true가 된다.
                block for문에서 block=0일 때 cb24에 20x2 크기의 타일을 임시로 저장해둔다.
                그 이후 block >=1 이면은 enable_reload 구문에서
                cb24에 있는 데이터를 dst register에 subblock 단위로 불러서 누산한다.
                최종 결과는 20x2 행렬 블록 하나를 생성한다.
            */
            if (spill) {
                enable_reload = true;
            }

            // 계산을 마친 데이터는 pop을 해서 cb에서 제거한다.
            cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
            cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
        }
    }
}
}  // namespace NAMESPACE