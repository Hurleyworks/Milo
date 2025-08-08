#include "../../../common/common_shared.h"

// Milo-specific buffer copy kernel
// Optimized for Milo's rendering pipeline which includes flow buffers for temporal denoising

// Simple utility function
CUDA_DEVICE_FUNCTION CUDA_INLINE float3 getXYZ(const float4& v)
{
    return make_float3(v.x, v.y, v.z);
}

// Combined kernel that copies all Milo surface buffers to linear memory in a single pass
// This is optimized for Milo's specific buffer layout including flow vectors
CUDA_DEVICE_KERNEL void copySurfacesToLinear(
    CUsurfObject colorSurfObj,
    CUsurfObject albedoSurfObj,
    CUsurfObject normalSurfObj,
    CUsurfObject flowSurfObj,
    float4* linearColorBuffer,
    float4* linearAlbedoBuffer,
    float4* linearNormalBuffer,
    float2* linearFlowBuffer,
    uint2 imageSize)
{
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= imageSize.x ||
        launchIndex.y >= imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;

    // Read color from surface object
    surf2Dread(&linearColorBuffer[linearIndex], colorSurfObj, launchIndex.x * sizeof(float4), launchIndex.y);
    
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

    // Read flow data (motion vectors for temporal denoising)
    float4 flowData;
    surf2Dread(&flowData, flowSurfObj, launchIndex.x * sizeof(float4), launchIndex.y);
    // Only copy x,y components as motion vectors
    linearFlowBuffer[linearIndex] = make_float2(flowData.x, flowData.y);
}