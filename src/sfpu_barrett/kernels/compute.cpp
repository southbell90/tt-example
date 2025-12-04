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
#include "compute_kernel_api/binary_shift.h"


#ifdef TRISC_MATH
inline void split_32_to_16_face(uint32_t a_idx, uint32_t hi_idx, uint32_t lo_idx) {
    constexpr size_t vectors_per_face = 8;
    constexpr uint32_t n_vector_in_tile = 32;

    uint32_t a_idx_base = a_idx * n_vector_in_tile;
    uint32_t hi_idx_base = hi_idx * n_vector_in_tile;
    uint32_t lo_idx_base = lo_idx * n_vector_in_tile;

    uint32_t amount = 65535;

    for (size_t i = 0; i < vectors_per_face; i++) {
        vUInt a = dst_reg[a_idx_base + i];
        dst_reg[hi_idx_base + i] = a >> 16;
        dst_reg[lo_idx_base + i] = a & amount;
    }
}

inline void mul32x32_to_64_face(uint32_t w0_hi_idx, uint32_t w0_lo_idx, uint32_t w1_idx, uint32_t w2_idx, uint32_t w3_idx, uint32_t t_hi_idx, uint32_t t_lo_idx) {
    constexpr size_t vectors_per_face = 8;
    constexpr uint32_t n_vector_in_tile = 32;

    uint32_t w0_hi_idx_base = w0_hi_idx * n_vector_in_tile;
    uint32_t w0_lo_idx_base = w0_lo_idx * n_vector_in_tile;
    uint32_t w1_idx_base = w1_idx * n_vector_in_tile;
    uint32_t w2_idx_base = w2_idx * n_vector_in_tile;
    uint32_t w3_idx_base = w3_idx * n_vector_in_tile;
    uint32_t t_hi_idx_base = t_hi_idx * n_vector_in_tile;
    uint32_t t_lo_idx_base = t_lo_idx * n_vector_in_tile;

    uint32_t amount = 65535;

    for (size_t i = 0; i < vectors_per_face; i++) {
        vUInt mid_lo = (vUInt(dst_reg[w1_idx_base + i]) & amount) + (vUInt(dst_reg[w2_idx_base + i]) & amount);
        vUInt carry0 = mid_lo >> 16;
        mid_lo = mid_lo & amount;
        vUInt mid_hi = (vUInt(dst_reg[w1_idx_base + i]) >> 16) + (vUInt(dst_reg[w2_idx_base + i]) >> 16) + carry0;

        vUInt sum = vUInt(dst_reg[w0_hi_idx_base + i]) + mid_lo;
        vUInt sum_lo = sum & amount;
        vUInt sum_hi = sum >> 16;

        vUInt lo = vUInt(dst_reg[w0_lo_idx_base + i]);   // low 16 bits already zero-extended
        vUInt hi = (sum_lo << 16);
        vUInt out = lo | hi;
        dst_reg[t_lo_idx_base + i] = hi;
        dst_reg[t_hi_idx_base + i] = vUInt(dst_reg[w3_idx_base + i]) + mid_hi + sum_hi;
        
    }
}

inline void add128_face(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3){
    constexpr size_t vectors_per_face = 8;
    constexpr uint32_t n_vector_in_tile = 32;

    uint32_t w0_idx = w0 * n_vector_in_tile;
    uint32_t w1_idx = w1 * n_vector_in_tile;
    uint32_t w2_idx = w2 * n_vector_in_tile;
    uint32_t w3_idx = w3 * n_vector_in_tile;
    uint32_t a0_idx = a0 * n_vector_in_tile;
    uint32_t a1_idx = a1 * n_vector_in_tile;
    uint32_t a2_idx = a2 * n_vector_in_tile;
    uint32_t a3_idx = a3 * n_vector_in_tile;

    for (size_t i = 0; i < vectors_per_face; i++) {
        // w0 += a0
        vUInt tmp = vUInt(dst_reg[w0_idx + i]) + vUInt(dst_reg[a0_idx + i]);
        vUInt carry = 0;
        v_if(tmp < vUInt(dst_reg[w0_idx + i])) {
            carry = 1;
        }
        v_endif;
        dst_reg[w0_idx + i] = tmp;

        // w1 += a1 + carry
        tmp = vUInt(dst_reg[w1_idx + i]) + vUInt(dst_reg[a1_idx + i]);
        vUInt carry2 = 0;
        v_if(tmp < vUInt(dst_reg[w1_idx + i])) {
            carry2 = 1;
        }
        v_endif;
        tmp = tmp + carry;
        vUInt carry3 = 0;
        v_if(tmp < carry) {
            carry3 = 1;
        }
        v_endif;
        dst_reg[w1_idx + i] = tmp;
        carry = (carry2 | carry3);

        // w2 += a2 + array
        tmp = vUInt(dst_reg[w2_idx + i]) + vUInt(dst_reg[a2_idx + i]);
        carry2 = 0;
        v_if(tmp < vUInt(dst_reg[w2_idx + i])) {
            carry2 = 1;
        }
        v_endif;
        tmp = tmp + carry;
        carry3 = 0;
        v_if(tmp < carry) {
            carry3 = 1;
        }
        v_endif;
        dst_reg[w2_idx + i] = tmp;
        carry = (carry2 | carry3);

        // w3 += a3 + array
        tmp = vUInt(dst_reg[w3_idx + i]) + vUInt(dst_reg[a3_idx + i]);
        carry2 = 0;
        v_if(tmp < vUInt(dst_reg[w3_idx + i])) {
            carry2 = 1;
        }
        v_endif;
        tmp = tmp + carry;
        carry3 = 0;
        v_if(tmp < carry) {
            carry3 = 1;
        }
        v_endif;
        dst_reg[w3_idx + i] = tmp;

    }
}

inline void copy_reg_face(uint32_t src, uint32_t dst, uint32_t trash) {
    constexpr size_t vectors_per_face = 8;
    constexpr uint32_t n_vector_in_tile = 32;

    uint32_t src_idx = src * n_vector_in_tile;
    uint32_t dst_idx = dst * n_vector_in_tile; 

    for (size_t i = 0; i < vectors_per_face; i++) {
        dst_reg[dst_idx + i] = dst_reg[src_idx + i];
    }
}

inline void barrett_reduce_face(uint32_t r_hi, uint32_t r_lo, uint32_t q_hi, uint32_t q_lo, uint32_t q_) {
    constexpr size_t vectors_per_face = 8;
    constexpr uint32_t n_vector_in_tile = 32;

    uint32_t r_hi_idx = r_hi * n_vector_in_tile;
    uint32_t r_lo_idx = r_lo * n_vector_in_tile; 
    uint32_t q_hi_idx = q_hi * n_vector_in_tile;
    uint32_t q_lo_idx = q_lo * n_vector_in_tile; 

    vUInt q = q_;

    for (size_t i = 0; i < vectors_per_face; i++) {
        v_if(vUInt(dst_reg[r_lo_idx + i]) < vUInt(dst_reg[q_lo_idx + i])) {
            dst_reg[r_hi_idx + i] = vUInt(dst_reg[r_hi_idx + i]) - 1;
        }
        v_endif;

        dst_reg[r_lo_idx + i] = vUInt(dst_reg[r_lo_idx + i]) - vUInt(dst_reg[q_lo_idx + i]);
        dst_reg[r_hi_idx + i] = vUInt(dst_reg[r_hi_idx + i]) - vUInt(dst_reg[q_hi_idx + i]);

        v_if(vUInt(dst_reg[r_hi_idx + i]) > 0 || vUInt(dst_reg[r_lo_idx + i]) >= q) {
            v_if(vUInt(dst_reg[r_lo_idx + i]) >= q) {
                dst_reg[r_lo_idx + i] = vUInt(dst_reg[r_lo_idx + i]) - q;
            }
            v_else {
                dst_reg[r_hi_idx + i] = vUInt(dst_reg[r_hi_idx + i]) - 1;
                dst_reg[r_lo_idx + i] = vUInt(dst_reg[r_lo_idx + i]) - q;
            }
            v_endif;
        }
        v_endif;

        v_if(vUInt(dst_reg[r_hi_idx + i]) > 0 || vUInt(dst_reg[r_lo_idx + i]) >= q) {
            v_if(vUInt(dst_reg[r_lo_idx + i]) >= q) {
                dst_reg[r_lo_idx + i] = vUInt(dst_reg[r_lo_idx + i]) - q;
            }
            v_else {
                dst_reg[r_hi_idx + i] = vUInt(dst_reg[r_hi_idx + i]) - 1;
                dst_reg[r_lo_idx + i] = vUInt(dst_reg[r_lo_idx + i]) - q;
            }
            v_endif;
        }
        v_endif;

        // 결과 값 0번 레지스터에 저장
        dst_reg[0 + i] = dst_reg[r_lo_idx + i];
    }
}


#endif


inline void split_32_to_16(uint32_t a_idx, uint32_t hi_idx, uint32_t lo_idx) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(
        split_32_to_16_face, a_idx, hi_idx, lo_idx, (int)ckernel::VectorMode::RC));
}

inline void mul32x32_to_64(uint32_t w0_hi_idx, uint32_t w0_lo_idx, uint32_t w1_idx, uint32_t w2_idx, uint32_t w3_idx, uint32_t t_hi_idx, uint32_t t_lo_idx) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(
        mul32x32_to_64_face, w0_hi_idx, w0_lo_idx, w1_idx, (int)ckernel::VectorMode::None, w2_idx, w3_idx, t_hi_idx, t_lo_idx));
}

inline void add128(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(
        add128_face, w0, w1, w2, (int)ckernel::VectorMode::RC, w3, a0, a1, a2, a3));
}

inline void copy_reg(uint32_t src, uint32_t dst) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(copy_reg_face, src, dst, dst, (int)ckernel::VectorMode::RC));
}

inline void barrett_reduce(uint32_t r_hi, uint32_t r_lo, uint32_t q_hi, uint32_t q_lo, uint32_t q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(barrett_reduce_face, r_hi, r_lo, q_hi, VectorMode::RC, q_lo, q));
}

void mul32x32(uint32_t a, uint32_t b, 
    uint32_t a1, uint32_t a0, uint32_t b1, uint32_t b0,
    uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3
) {
    split_32_to_16(a, a1, a0);
    split_32_to_16(b, b1, b0);

    ckernel::mul_int32_tile_init();
    ckernel::mul_uint32_tile(a0, b0, w0);
    ckernel::mul_uint32_tile(a0, b1, w1);
    ckernel::mul_uint32_tile(a1, b0, w2);
    ckernel::mul_uint32_tile(a1, b1, w3);


    uint32_t w0_hi = b1;
    uint32_t w0_lo = b0;
    split_32_to_16(w0,w0_hi,w0_lo);

    uint32_t t1 = a1;
    uint32_t t0 = a0;

    // a0, a1 인덱스에 결과 값이 저장된다. a0 = result_lo , a1 = result_hi
    mul32x32_to_64(w0_hi,w0_lo,w1,w2,w3,t1,t0);
}

namespace NAMESPACE {

void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t q = get_arg_val<uint32_t>(1);

    tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    tt::CBIndex cb_in2 = tt::CBIndex::c_2;
    tt::CBIndex cb_in3 = tt::CBIndex::c_3;
    tt::CBIndex cb_in4 = tt::CBIndex::c_4;
    tt::CBIndex cb_in5 = tt::CBIndex::c_5;
    tt::CBIndex cb_out = tt::CBIndex::c_16;

    tile_regs_acquire();
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    cb_wait_front(cb_in2, 1);
    cb_wait_front(cb_in3, 1);
    cb_wait_front(cb_in4, 1);
    cb_wait_front(cb_in5, 1);

    init_sfpu(cb_in0, cb_out);
    init_sfpu(cb_in1, cb_out);
    init_sfpu(cb_in2, cb_out);
    init_sfpu(cb_in3, cb_out);
    init_sfpu(cb_in4, cb_out);
    init_sfpu(cb_in5, cb_out);

    // dst register 0: a, 1: b, 2: mu_hi, 3: mu_lo, 4: q , 5: 0
    copy_tile_init(cb_in0);
    copy_tile(cb_in0, 0, 0);
    copy_tile_init(cb_in1);
    copy_tile(cb_in1, 0, 1);
    copy_tile_init(cb_in2);
    copy_tile(cb_in2, 0, 2);
    copy_tile_init(cb_in3);
    copy_tile(cb_in3, 0, 3);
    copy_tile_init(cb_in4);
    copy_tile(cb_in4, 0, 4);
    copy_tile_init(cb_in5);
    copy_tile(cb_in5, 0, 5);
    
    // t = a * b
    // 6: t_hi , 7: t_lo
    uint32_t t_hi = 6;
    uint32_t t_lo = 7;
    mul32x32(0,1,6,7,8,9,10,11,12,13);

    // barret reduce
    // t mod q
    // q_hat = floor((t * mu) / 2^64)
    // 8: qhat_hi , 9: qhat_lo
    // mul32x32(7, 3, 10, 11, 12, 13, 14, 15 ,16 ,17); // 10: hi0 , 11: lo0
    // mul32x32(7, 2, 12,13,14,15,16,17,18,19);        // 12: hi1 , 13: lo1
    // mul32x32(6,3,14,15,16,17,18,19,20,21);          // 14: hi2 , 15: lo2
    // mul32x32(6, 2, 16,17,18,19,20,21,22,23);         // 16: hi3 , 17: lo3

    uint32_t w0 = 18;
    uint32_t w1 = 19;
    uint32_t w2 = 20;
    uint32_t w3 = 21;

    uint32_t hi0 = 10;
    uint32_t lo0 = 11;
    uint32_t hi1 = 12;
    uint32_t lo1 = 13;
    uint32_t hi2 = 14;
    uint32_t lo2 = 15;
    uint32_t hi3 = 16;
    uint32_t lo3 = 17;
    uint32_t zero = 5;

    add128(w0,w1,w2,w3,lo0,hi0, zero,zero);
    add128(w0,w1,w2,w3,zero,lo1, hi1, zero);
    add128(w0,w1,w2,w3,zero,lo2, hi2, zero);
    add128(w0,w1,w2,w3,zero,zero, lo3,hi3);

    uint32_t qhat_hi = 8;
    uint32_t qhat_lo = 9;

    // src --> dst
    copy_reg(w3, qhat_hi);
    copy_reg(w2, qhat_lo);

    uint32_t q_hat = qhat_lo;
    // r = t - q_hat * q
    mul32x32(q_hat, 4, 10,11,12,13,14,15,16,17);    // 10: q_hi, 11: q_lo
    uint32_t r_hi = t_hi;
    uint32_t r_lo = t_lo;
    uint32_t q_hi = 10;
    uint32_t q_lo = 11;

    barrett_reduce(r_hi, r_lo, q_hi, q_lo, q);     // 결과값 0번 레지스터에 저장

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