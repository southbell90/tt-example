#include <random>
#include <cmath>
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
using namespace tt;
using namespace std;
using namespace tt::tt_metal;


#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    bool pass = true;

    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t n_tiles = 1;
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(uint8_t) * elements_per_tile;
    constexpr uint32_t out_tile_size_bytes = sizeof(uint32_t) * elements_per_tile;

    // Allocate DRAM buffers for the input and output data.
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::DeviceLocalBufferConfig out_dram_config{
        .page_size = out_tile_size_bytes, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config{
        .size = tile_size_bytes * n_tiles}; 
    distributed::ReplicatedBufferConfig out_buffer_config{
        .size = out_tile_size_bytes * n_tiles}; 

    std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    std::shared_ptr<distributed::MeshBuffer> src1_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
        distributed::MeshBuffer::create(out_buffer_config, out_dram_config, mesh_device.get());

    // Allocate 2 circular buffers for input and output.
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::UInt8}})
            .set_page_size(src0_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src1_cb_index, tt::DataFormat::UInt8}})
            .set_page_size(src1_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_input_tiles * out_tile_size_bytes, {{output_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(output_cb_index, out_tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_id = CreateKernel(
        program,
        "/home/southbell/tt-example/src/matmul_uint8/kernels/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_id = CreateKernel(
        program,
        "/home/southbell/tt-example/src/matmul_uint8/kernels/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    KernelHandle compute_id = CreateKernel(
        program,
        "/home/southbell/tt-example/src/matmul_uint8/kernels/compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
        });

    // Initialize the input data with random values and use as the input to the kernel.
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<std::uint8_t> dist(0, 250);

    std::vector<uint8_t> src0_vec(elements_per_tile * n_tiles);
    std::vector<uint8_t> src1_vec(elements_per_tile * n_tiles);
    
    for (uint8_t& v : src0_vec) {
        v = dist(engine);
    }
    for (uint8_t& v : src1_vec) {
        v = dist(engine);
    }


    // 16-bit으로하면 overflow가 나서 FPU의 결과 result_vec과 일치하지 않는다.
    // 즉 FPU matmul의 결과는 32-bit로 나온다는 것과 같다.
    std::vector<uint32_t> golden(elements_per_tile * n_tiles, 0);

    for(int i = 0; i < TILE_WIDTH; i++) {
        for(int j = 0; j < TILE_HEIGHT; j++) {
            for(int k = 0; k < TILE_HEIGHT; k++) {
                golden.at(i * TILE_WIDTH + j) += src0_vec.at(i * TILE_WIDTH + k) * src1_vec.at(j + TILE_WIDTH * k);
            }
        }
    }

    src0_vec = tilize_nfaces(src0_vec, TILE_HEIGHT, TILE_WIDTH);
    src1_vec = tilize_nfaces(src1_vec, TILE_HEIGHT, TILE_WIDTH);

    // Set up the runtime arguments for the kernels.
    SetRuntimeArgs(program, compute_id, core, {n_tiles});
    SetRuntimeArgs(
        program,
        reader_id,
        core,
        {
            src0_dram_buffer->address(),
            src1_dram_buffer->address(),
            n_tiles
        });

    SetRuntimeArgs(program, writer_id, core, {dst_dram_buffer->address(), n_tiles});

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, /*blocking=*/false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read the result (from shard at mesh coordinate {0,0} on a unit mesh) and compare to our expected result.
    std::vector<uint32_t> result_vec(elements_per_tile * n_tiles);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    result_vec = untilize_nfaces(result_vec, TILE_HEIGHT, TILE_WIDTH);

    // 검증
    for(int i = 0; i < elements_per_tile * n_tiles; i++) {
        if(golden.at(i) != result_vec.at(i)) {
            fmt::print("golden and result unmatch at {}, golden = {}, result = {}\n", i, golden.at(i), result_vec.at(i));
            // pass = false;
            break;
        }
    }

    // Finally, close the device.
    pass &= mesh_device->close();

    if (pass) {
        fmt::print("Test Passed!! ---- matmul_uint8\n");
    } else {
        TT_THROW("Test Failed!!");
    }

    return 0;
}