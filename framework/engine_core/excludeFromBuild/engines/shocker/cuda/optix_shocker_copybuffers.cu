// optix_shocker_copybuffers.cu
// CUDA kernels for copying and visualizing buffers in the Shocker engine
// Following Milo pattern - all parameters passed directly, no global plp

#include "../../../common/common_shared.h"

// Simple utility function
CUDA_DEVICE_FUNCTION CUDA_INLINE float3 getXYZ(const float4& v)
{
    return make_float3(v.x, v.y, v.z);
}

// Combined kernel that copies all Shocker surface buffers to linear memory in a single pass
// Takes all parameters directly like Milo, avoiding constant memory issues
CUDA_DEVICE_KERNEL void copySurfacesToLinear(
    CUsurfObject beautySurfObj,
    CUsurfObject albedoSurfObj,
    CUsurfObject normalSurfObj,
    CUsurfObject motionSurfObj,
    float4* linearColorBuffer,
    float4* linearAlbedoBuffer,
    float4* linearNormalBuffer,
    float2* linearMotionVectorBuffer,
    uint2 imageSize)
{
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= imageSize.x ||
        launchIndex.y >= imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;

    // Read beauty/color from surface object
    surf2Dread(&linearColorBuffer[linearIndex], beautySurfObj, launchIndex.x * sizeof(float4), launchIndex.y);
    
    // Read albedo from surface object
    surf2Dread(&linearAlbedoBuffer[linearIndex], albedoSurfObj, launchIndex.x * sizeof(float4), launchIndex.y);

    // Read normal and normalize if non-zero
    float4 normalData;
    surf2Dread(&normalData, normalSurfObj, launchIndex.x * sizeof(float4), launchIndex.y);
    float3 normal = getXYZ(normalData);
    if (normal.x != 0 || normal.y != 0 || normal.z != 0)
    {
        float len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        normal = make_float3(normal.x / len, normal.y / len, normal.z / len);
    }
    linearNormalBuffer[linearIndex] = make_float4(normal.x, normal.y, normal.z, 1.0f);

    // Read motion vector data for temporal denoising
    float4 motionData;
    surf2Dread(&motionData, motionSurfObj, launchIndex.x * sizeof(float4), launchIndex.y);
    // Only copy x,y components as motion vectors
    linearMotionVectorBuffer[linearIndex] = make_float2(motionData.x, motionData.y);
}