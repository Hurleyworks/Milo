// optix_shocker_copybuffers.cu
// CUDA kernels for copying and visualizing buffers in the Shocker engine
// STUB VERSION - no implementation

#define PURE_CUDA
#include "../shocker_shared.h"

CUDA_DEVICE_KERNEL void copyToLinearBuffers(
    float4* linearColorBuffer,
    float4* linearAlbedoBuffer,
    float4* linearNormalBuffer,
    float2* linearMotionVectorBuffer) {
    // STUB
}

CUDA_DEVICE_KERNEL void visualizeToOutputBuffer(
    void* linearBuffer,
    uint32_t bufferTypeToDisplay,
    float motionVectorOffset, 
    float motionVectorScale,
    optixu::NativeBlockBuffer2D<float4> outputBuffer) {
    // STUB
}