// optix_shocker_copybuffers.cu
// CUDA kernels for copying and visualizing buffers in the Shocker engine
// STUB VERSION - no implementation

#define PURE_CUDA
#include "../shocker_shared.h"

using namespace shocker_shared;

CUDA_DEVICE_KERNEL void copyToLinearBuffers (
    float4* linearColorBuffer,
    float4* linearAlbedoBuffer,
    float4* linearNormalBuffer,
    float2* linearMotionVectorBuffer)
{
    const uint2 launchIndex = make_uint2 (
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= plp.s->imageSize.x ||
        launchIndex.y >= plp.s->imageSize.y)
        return;

    const uint32_t linearIndex = launchIndex.y * plp.s->imageSize.x + launchIndex.x;
    linearColorBuffer[linearIndex] = plp.s->beautyAccumBuffer.read (launchIndex);
    linearAlbedoBuffer[linearIndex] = plp.s->albedoAccumBuffer.read (launchIndex);
    Normal3D normal (getXYZ (plp.s->normalAccumBuffer.read (launchIndex)));
    if (normal.x != 0 || normal.y != 0 || normal.z != 0)
        normal.normalize();
    linearNormalBuffer[linearIndex] = make_float4 (normal.toNative(), 1.0f);
    const GBuffer1Elements gb1Elems = plp.s->GBuffer1[plp.f->bufferIndex].read (launchIndex);
    linearMotionVectorBuffer[linearIndex] = gb1Elems.motionVector.toNative();
}