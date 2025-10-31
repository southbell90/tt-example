#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

namespace NAMESPACE {

void MAIN {
    const uint32_t Nt = get_compile_time_arg_val(0);
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    mm_init(cb_in0, cb_in1, cb_out);


    for(uint32_t i = 0; i < Nt; i++) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < Nt; ++j) {

            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);


        }
        // Commit and wait for the registers are populated with the results from the FPU
        tile_regs_commit();
        tile_regs_wait();

        // Ensure the output circular buffer has space for the result tile.
        cb_reserve_back(cb_out, 1);
        // Pack the result tile into the output circular buffer.
        pack_tile(0, cb_out);
        // Mark the output tile as ready so the writer can read it.
        cb_push_back(cb_out, 1);

        // We don't need the registers anymore, so we can release them and prepare for the next output tile.
        tile_regs_release();
    }
        
}
}  // namespace NAMESPACE