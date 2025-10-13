#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "tt-metalium/base_types.hpp"

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int argc, char** argv) {
    bool pass = true;

    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    // data 다운로드/업로드와 프로그램 실행을 위해 mesh command queue를 만든다.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t n_tiles = 64;
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

    //3개의 DRAM buffer를 만든다. 2개는 data sources 그리고 1개는 output
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = n_tiles * tile_size_bytes
    };

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    constexpr float val_to_add = -1.0f;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
    std::vector<bfloat16> b_data(elements_per_tile * n_tiles, bfloat16(val_to_add));
    for(auto& val : a_data) {
        val = bfloat16(distribution(rng));
    }

    // Upload the data from host to the device.
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b_data, false);

    /*
        tensix core에 있는 SRAM은 circular buffers로 이루어져 있다.
        tensix에 있는 다른 커널들의 커뮤니케이션 채널이다.
        tensix에는 총 32개의 circular buffers가 있다.
        cb를 사용하기 위해서는 host program은 circular buffer를 할당하고 적절한 cb index를 활용해야 한다.
    */
   constexpr uint32_t tiles_per_cb = 2;
    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    CircularBufferConfig c0_cfg = CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes,
        /*data_format_spec=*/{{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, tile_size_bytes);
    CBHandle cb_src0 = CreateCircularBuffer(program, core, c0_cfg);

    tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
    CBHandle cb_src1 = CreateCircularBuffer(program, core, CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes,
        /*data_format_spec=*/{{src1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src1_cb_index, tile_size_bytes));
    tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
    CBHandle cb_dst = CreateCircularBuffer(program, core, CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes,
        /*data_format_spec=*/{{dst_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(dst_cb_index, tile_size_bytes));

    /*
        tensix core는 5 RISC-V cores를 가지고 있다.
        그 중 2개는 data movement cores이다. NoC과 연결되어 명령어를 발행하고 다른 chip resource에 접근이 가능하다. (DRAM 포함)
        다른 3개는 compute cores다. sing compute kernel을 3개의 cores가 실행한다. matrix and vector engines에 접근 가능하다.
        3개의 compute cores는 unpack, math, pack cores 이다.
        데이터를 L1에서 matrix or vector engines로 이동, computation 명령어 발행, 결과를 다시 L1으로 이동하는 역할을 한다.
    */
    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*src0_dram_buffer->get_backing_buffer()).append_to(reader_args);
    TensorAccessorArgs(*src1_dram_buffer->get_backing_buffer()).append_to(reader_args);
    auto reader = CreateKernel(
        program,
        "/home/southbell/tt-example/src/eltwise_binary/kernels/read_tiles.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_args});
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*dst_dram_buffer->get_backing_buffer()).append_to(writer_args);
    auto writer = CreateKernel(
        program,
        "/home/southbell/tt-example/src/eltwise_binary/kernels/write_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_args});
    auto compute = CreateKernel(
        program,
        "/home/southbell/tt-example/src/eltwise_binary/kernels/tiles_add.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});   // ComputeConfig 객체는 compute kernel이 만들어져야 한다고 알려준다.
                                                                // MathFidelitysms FPU에서 floating-point 연산을 얼마나 정확하게 컨트롤 한느지를 나타낸다.
                                                                // 다른 vector engine 같은 것들은 이 setting에 영향을 받지 않는다.
    
    // Set the runtime arguments for the kernels. This also registers
    // the kernels with the program.
    SetRuntimeArgs(program, reader, core, 
        {static_cast<uint32_t>(src0_dram_buffer->address()), static_cast<uint32_t>(src1_dram_buffer->address()), n_tiles}
    );
    SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(dst_dram_buffer->address()), n_tiles});
    SetRuntimeArgs(program, compute, core, {n_tiles});

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    
    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
    TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
    for (size_t i = 0; i < result_vec.size(); ++i) {
        const float expected = static_cast<float>(a_data[i]) + val_to_add;
        const float actual = static_cast<float>(result_vec[i]);

        if (std::abs(expected - actual) > eps) {
            pass = false;
            fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
        }
    }

    pass &= mesh_device->close();

    if (pass) {
        fmt::print("Test Passed!!!!\n");
    } else {
        TT_THROW("Test Failed!!!!");
    }

    return 0;
}