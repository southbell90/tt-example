#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

/*
    FPU가 지원하는 data type에 대한 테스트 파일이다.
    FPU에서 int8로 연산을 하고 output은 int32로 나온다.
*/
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

void golden_matmul(
    const std::vector<uint8_t>& a,
    const std::vector<uint8_t>& b,
    std::vector<uint32_t>& output,
    uint32_t N
) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                for(int k = 0; k < N; k++) {
                    output.at(i * N + j) += (a.at(i * N + k) * b.at(k*N + j));
                }
            }
    }
}

void matmul_single_core(
    const std::vector<uint8_t>& W,
    const std::vector<uint8_t>& a,
    std::vector<uint32_t>& output,
    uint32_t N,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up mesh command queue, workload, device range, and program. This is a single-core example using core {0,0}.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    // Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
    CoreCoord core({0, 0});

    // Calcaulate the number of tiles for each dimension.
    uint32_t Nt = N / TILE_WIDTH;

    // Create DRAM buffers for the input and output data.
    uint32_t single_tile_size = sizeof(uint8_t) * TILE_HEIGHT * TILE_WIDTH;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::DeviceLocalBufferConfig out_dram_config{
        .page_size = single_tile_size * 4, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_A{.size = sizeof(uint8_t) * W.size()};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = sizeof(uint8_t) * a.size()};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = sizeof(uint32_t) * output.size()};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, out_dram_config, mesh_device.get());

    tt::DataFormat cb_data_format = tt::DataFormat::UInt8;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = 2;

    // Circular buffer for matrix A tiles
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);


    // Circular buffer for matrix B tiles
    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);


    // Circular buffer for output tiles
    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size * 4, {{output_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(output_cb_index, single_tile_size * 4);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Create the data movement kernels and the compute kernel
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "/home/southbell/tt-example/src/data_type_test/kernels/reader.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "/home/southbell/tt-example/src/data_type_test/kernels/writer.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compile time arguments for the kernels
    // Note that these take effect at the kernel's compile time. Chaning these values will require recompilation of the
    // kernel. Having arguments at compile time allows the compiler to optimize the kernel for the specific use case.
    // Like applying loop unrolling, constant folding, etc.. resulting in a more efficient kernel.
    std::vector<uint32_t> compute_compile_time_args = {
        Nt   // Nt
    };
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "/home/southbell/tt-example/src/data_type_test/kernels/compute.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_compile_time_args});

    // Set kernel arguments
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, Nt});

    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Nt});

    // Upload the input data to the DRAM buffers, execute the kernels, wait for the result to be read into the output
    // buffer
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, W, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, a, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
}

///////////////////////////////////////

int main() {
    bool pass = true;

    // Open device
    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    // parameters for the matrix multiplication
    constexpr uint32_t N = 32;  

    static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");

    // input vectors with various ranges of values
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<std::uint8_t> dist(0, 255);

    std::vector<uint8_t> W_vec(N * N);
    std::vector<uint8_t> a_vec(N * TILE_WIDTH);

    for (uint8_t& v : W_vec) {
        v = dist(engine);
    }
    for (uint8_t& v : a_vec) {
        v = dist(engine);
    }

    // Golden Matmul running on CPU so we can compare later
    std::vector<uint32_t> golden_vec(N*N);
    golden_matmul(W_vec, a_vec, golden_vec, N);

    std::vector<uint8_t> A_bf(W_vec.size()), B_bf(a_vec.size());
    for (size_t i = 0; i < W_vec.size(); ++i) A_bf[i] = W_vec[i];  
    for (size_t i = 0; i < a_vec.size(); ++i) B_bf[i] = a_vec[i];
    A_bf = tilize_nfaces(A_bf, N, N);
    B_bf = tilize_nfaces(B_bf, N, TILE_WIDTH);

    std::vector<uint32_t> result_vec(N * TILE_WIDTH, 0);
    matmul_single_core(A_bf, B_bf, result_vec, N, mesh_device);
    result_vec = untilize_nfaces(result_vec, N, TILE_WIDTH);

    fmt::print("Output vector of size {}\n", result_vec.size());


    pass &= mesh_device->close();

    // 검증
    for(int i = 0; i < N * N; i++) {
        if(golden_vec.at(i) != result_vec.at(i)){
            fmt::print("golden_vec != result_vec at = {}, golden_vec = {}, result_vec = {}\n", i, golden_vec.at(i), result_vec.at(i));
        }
    }

    if (pass) {
        fmt::print("Test Passed!!!!! ---- data_type_test\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}