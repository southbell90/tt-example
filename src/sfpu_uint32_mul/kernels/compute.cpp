#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/mul_int32_sfpu.h"
#include "compute_kernel_api/mul_int_sfpu.h"


namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    tt::CBIndex cb_out = tt::CBIndex::c_16;


    tile_regs_acquire();

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    copy_tile_init(cb_in0);
    copy_tile(cb_in0, 0, 0);
    copy_tile_init(cb_in1);
    copy_tile(cb_in1, 0, 1);


    init_sfpu(cb_in0, cb_out);

    mul_int32_tile_init();

    mul_uint32_tile(0, 1, 2);

    tile_regs_commit();
    tile_regs_wait();
    // Wait for space in the circular buffer to be available for us to write
    cb_reserve_back(cb_out, 1);
    pack_tile(2, cb_out);  // copy tile 0 from the registers to the CB
    // We don't need the input tile anymore, mark it as consumed
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);

    // Mark the tile as ready for the writer kernel to write to DRAM
    cb_push_back(cb_out, 1);
    tile_regs_release();

}
}  