// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

// This example demonstrates a simple data copy from DRAM into L1(SRAM) and to another place in DRAM.
// The general flow is as follows:
// 1. Initialize the device
// 2. Create the data movement kernel (fancy word of specialized subroutines) on core {0, 0}
//    that will perform the copy
// 3. Create the buffer (both on DRAM And L1) and fill DRAM with data. Point the kernel to the buffers.
// 4. Execute the kernel
// 5. Read the data back from the buffer
// 6. Validate the data
// 7. Clean up the device. Exit

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();

    constexpr uint32_t num_tiles = 50;
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
    constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

    //allocation properties within a device
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::L1
    };

    //overall buffer size across all device in mesh
    distributed::ReplicatedBufferConfig l1_buffer_config{.size = tile_size_bytes};
    auto l1_buffer = distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());


    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    distributed::ReplicatedBufferConfig dram_buffer_config{.size = dram_buffer_size};
    auto input_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto output_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

    std::vector<bfloat16> input_vec(elements_per_tile * num_tiles);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
    for(auto& val : input_vec) {
        val = bfloat16(distribution(rng));
    }
    distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, false);

    constexpr CoreCoord core = {0,0};
    std::vector<uint32_t> dram_copy_compile_time_args;
    TensorAccessorArgs(*input_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
    TensorAccessorArgs(*output_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);

    KernelHandle dram_copy_kernel_id = CreateKernel(
        program,
        "/home/southbell/tt-example/src/loopback/kernels/loopback_dram_copy.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = dram_copy_compile_time_args
        }
    );

    const std::vector<uint32_t> runtime_args = {
        static_cast<uint32_t>(l1_buffer->address()),
        static_cast<uint32_t>(input_dram_buffer->address()),
        static_cast<uint32_t>(output_dram_buffer->address()),
        num_tiles
    };

    SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, output_dram_buffer, true);

    bool pass = true;
    for(int i = 0; i < input_vec.size(); i++) {
        if(input_vec[i] != result_vec[i]) {
            pass = false;
            break;
        }
    }

    pass &= mesh_device->close();

    if(pass) {
        fmt::print("My Test Passed!!!!!\n");
    } else {
        TT_THROW("My Test Failed!");
    }
    

    return 0;
}
