#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/remainder.h"


namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t q_bits = get_arg_val<uint32_t>(1);
    uint32_t q_rcp_bits = get_arg_val<uint32_t>(2);

    tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    tt::CBIndex cb_out = tt::CBIndex::c_16;

    tile_regs_acquire();

    init_sfpu(cb_in0, cb_out);

    cb_wait_front(cb_in0, 1);

    copy_tile_init(cb_in0);
    copy_tile(cb_in0, 0, 0);

    // remainder_tile_init(q_bits, q_rcp_bits);
    remainder_tile(0, q_bits, q_rcp_bits);

    tile_regs_commit();
    tile_regs_wait();
    // Wait for space in the circular buffer to be available for us to write
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);  // copy tile 0 from the registers to the CB
    // We don't need the input tile anymore, mark it as consumed
    cb_pop_front(cb_in0, 1);

    // Mark the tile as ready for the writer kernel to write to DRAM
    cb_push_back(cb_out, 1);
    tile_regs_release();

}
}  