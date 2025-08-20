#define PURE_CUDA
#include "../ripr_shared.h"

using namespace shared;
using namespace ripr_shared;



CUDA_DEVICE_KERNEL void copyToLinearBuffers(
    float4* linearColorBuffer,
    float4* linearAlbedoBuffer,
    float4* linearNormalBuffer,
    float2* linearMotionVectorBuffer) {
    const uint2 launchIndex = make_uint2(
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= ripr_plp.s->imageSize.x ||
        launchIndex.y >= ripr_plp.s->imageSize.y)
        return;

    const uint32_t linearIndex = launchIndex.y * ripr_plp.s->imageSize.x + launchIndex.x;
    linearColorBuffer[linearIndex] = ripr_plp.s->beautyAccumBuffer.read(launchIndex);
    linearAlbedoBuffer[linearIndex] = ripr_plp.s->albedoAccumBuffer.read(launchIndex);
    Normal3D normal(getXYZ(ripr_plp.s->normalAccumBuffer.read(launchIndex)));
    if (normal.x != 0 || normal.y != 0 || normal.z != 0)
        normal.normalize();
    linearNormalBuffer[linearIndex] = make_float4(normal.toNative(), 1.0f);
    const GBuffer1Elements gb1Elems = ripr_plp.s->GBuffer1[ripr_plp.f->bufferIndex].read(launchIndex);
    linearMotionVectorBuffer[linearIndex] = gb1Elems.motionVector.toNative();
}




