#include <cstdint>
#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include "tt-metalium/buffer.hpp"
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    /*
        Tensix Core의 RISCV processors 1,5는 data movement에 사용되지만 기본적인 컴퓨팅 능력은 가지고 있다.
        이 processors를 가지고 기본적인 integer 덧셈을 할 것이다.
    */

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    /*
        configure DRAM and L1 buffers
    */
    constexpr uint32_t buffer_size = sizeof(uint32_t);
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = buffer_size,
        .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_size,
        .buffer_type = BufferType::L1};
    //ReplicatedBufferConfig는 mesh 전체에 걸쳐 config 할 수 있게 해준다.
    distributed::ReplicatedBufferConfig buffer_config{
        .size = buffer_size,
    };

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src0_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
    auto src1_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
    auto dst_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::vector<uint32_t> src0_vec = {14};
    std::vector<uint32_t> src1_vec = {7};

    EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);
    EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, /*blocking=*/false);

    KernelHandle kernel_id = CreateKernel(
        program,
        "/home/southbell/tt-example/src/add_2_integers_in_riscv/kernels/reader_writer_in_riscv.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(
        program,
        kernel_id,
        core,
        {
            src0_dram_buffer->address(),
            src1_dram_buffer->address(),
            dst_dram_buffer->address(),
            src0_l1_buffer->address(),
            src1_l1_buffer->address(),
            dst_l1_buffer->address(),
        });

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0), true);

    std::cout << "Success: Result is " << result_vec[0] << std::endl;
    mesh_device->close();

    return 0;
}