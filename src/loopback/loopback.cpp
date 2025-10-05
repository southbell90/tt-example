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

    /*
        mesh device를 만든다. 모든 연산들은 mesh abstraction을 사용한다.
        이 예제에서는 single device이기 때문에 1x1 mesh가 된다.
    */
    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    /*
        metalium에서의 연산들은 비동기적으로 거의 실행되고 연산의 순서는 command_queue에 의해 관리된다.
        command_queue는 FIFO 순서로 연산을 실행한다.
        command는 data upload/download 와 프로그램의 실행을 포함한다.
        mesh command queue는 mesh 전체에 걸친 연산들을 다룬다.
        Program 객체는 device에서 실행될 kernel들의 모음이다.
        Metailium은 다른 코어에 다른 kernel이 동시에 실행될 수 있다.
    */
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();


    /*
        3가지 종류의 버퍼가 있다.
        1. 코어 안에 있는 L1(SRAM) buffer
        2. input data가 될 DRAM buffer
        3. output data가 쓰일 DRAM buffer
        
        Tensix에서 일어나는 거의 모든 연산은 tile 크기로 aligned 돼있다.
        tile은 32x32 grid of values.
        아래 코드의 경우 tile의 개수는 50개이다.
        DRAM buffer에는 sizeof(tile) * num_tiles 크기 만큼의 buffer를 만든다.
        L1 buffer에는 한 번에 1개의 tile을 불러와서 계산한다.

        L1과 DRAM은 bank로 구성돼 있다.
        기본적인 buffer allocation 전략은 round-robin으로 page_size마다 모든 bank를 돌아가면서 접근한다.
        l1_config의 argument인 page_size는 위의 page_size를 의미한다.
        일반적으로 page_size는 tile size와 똑같이 맞춘다.
    */
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
    // 마지막 false는 non-blocking이라는 의미이다.(false = non-blocking, true = blocking)
    distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, false);


    constexpr CoreCoord core = {0,0};
    std::vector<uint32_t> dram_copy_compile_time_args;
    TensorAccessorArgs(*input_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
    TensorAccessorArgs(*output_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);

    /*
        DRAM에서 L1으로 데이터를 보내고 다시 받는 kernel을 만든다.

        loopback_dram_copy.cpp의 코드를 보면 주소를 카리키는 포인터가 아니라 uint32_t 타입을 사용하는 것을 알 수 있다.
        이것은 커널이 직접적으로 DRAM에 접근하지 않기 때문이다.
        대신 access request는 NoC로 보내지고 커널이 데이터에 접근하기 전에 buffer에 데이터가 들어가게 된다.
    */
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

    /*
        runtime arguments를 셋팅한다. 
        runtime에 kernel은 이 arguments에 접근할 수 있다.
    */
    const std::vector<uint32_t> runtime_args = {
        static_cast<uint32_t>(l1_buffer->address()),
        static_cast<uint32_t>(input_dram_buffer->address()),
        static_cast<uint32_t>(output_dram_buffer->address()),
        num_tiles
    };

    SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

    /*
        MeshWorkload는 mesh 전체에 걸쳐 실행될 프로그램들의 모음을 나타낸다.
        EnqueueMeshWorkload는 non-block으로 실행된다. (3번째 argument가 false)
        distributed::Finish는 command queue에 있는 모든 것이 완료될때 까지 기다리게 된다.
    */
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
