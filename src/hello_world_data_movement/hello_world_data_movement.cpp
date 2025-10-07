#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main() {
    
    constexpr CoreCoord core = {0, 0};
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    
    /*
        Tensix에는 2개의 data movement core가 존재한다.
        이 예제에서는 2개의 똑같은 kernel을 RISCV_0, RISCV_1에 시작한다.
    */
    KernelHandle void_dataflow_kernel_noc0_id = CreateKernel(
        program,
        "/home/southbell/tt-example/src/hello_world_data_movement/kernel/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default });

    KernelHandle void_dataflow_kernel_noc1_id = CreateKernel(
        program,
        "/home/southbell/tt-example/src/hello_world_data_movement/kernel/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{ .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default });


    SetRuntimeArgs(program, void_dataflow_kernel_noc0_id, core, {});
    SetRuntimeArgs(program, void_dataflow_kernel_noc1_id, core, {});
    std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication." << std::endl;

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    return 0;
}