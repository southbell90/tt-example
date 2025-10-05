#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt-metalium/kernel_types.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main() {

    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    KernelHandle void_compute_kernel_id = CreateKernel(
        program,
        "/home/southbell/tt-example/src/hello_world_compute_kernel/kernel/void_compute_kernel.cpp",
        core,
        ComputeConfig{});

    SetRuntimeArgs(program, void_compute_kernel_id, core, {});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    fmt::print("Hello, Core (0, 0) on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

    distributed::Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    mesh_device->close();
    
    return 0;
}