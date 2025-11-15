#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/remainder.h"
#include "compute_kernel_api/mul_int32_sfpu.h"
#include "compute_kernel_api/mul_int_sfpu.h"
#include "compute_kernel_api/eltwise_unary/right_shift.h"
#include "compute_kernel_api/sub_int_sfpu.h"


#ifdef TRISC_MATH
inline void my_remainder_tile_face(uint32_t q) {
    constexpr size_t vectors_per_face = 8;

    for (size_t i = 0; i < vectors_per_face; i++) {
        vUInt x = dst_reg[i];
        v_if(x >= q) { x -= q ;}
        v_endif;
        dst_reg[i] = x;
    }
}
#endif


inline void my_remainder_tile(uint32_t idx_dst0, uint32_t q) {
    MATH(_llk_math_eltwise_unary_sfpu_params_<false>(
        my_remainder_tile_face, idx_dst0, VectorMode::RC, q));
}

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t q = get_arg_val<uint32_t>(1);

    tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    tt::CBIndex cb_in2 = tt::CBIndex::c_2;
    tt::CBIndex cb_out = tt::CBIndex::c_16;

    tile_regs_acquire();
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    cb_wait_front(cb_in2, 1);

    init_sfpu(cb_in0, cb_out);
    init_sfpu(cb_in1, cb_out);
    init_sfpu(cb_in2, cb_out);

    // dst register 0 에는 a, 1에는 mu, 2에는 q
    copy_tile_init(cb_in0);
    copy_tile(cb_in0, 0, 0);
    copy_tile_init(cb_in1);
    copy_tile(cb_in1, 0, 1);
    copy_tile_init(cb_in2);
    copy_tile(cb_in2, 0, 2);
    
    mul_int32_tile_init();
    mul_uint32_tile(0,1,3);     // 3번 레지스터에 a * mu

    right_shift_tile_init();
    right_shift_tile(3, 16);    // 3번 레지스터에 t = a * mu >> 32 
    right_shift_tile(3, 16);

    mul_int32_tile_init();
    mul_uint32_tile(2, 3, 3);   // 3번 레지스터에 t * q

    sub_int_tile_init();
    sub_uint32_tile(0,3,0);     // 0번 레지스터에 r = a - t * q

    my_remainder_tile(0, q);    // if r >= q : r -= q

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