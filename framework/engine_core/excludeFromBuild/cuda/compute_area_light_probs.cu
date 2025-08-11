// compute_area_light_probs.cu
// GPU kernels for computing area light importance sampling distributions

#if 0

#define PURE_CUDA
#include "../common/deviceCommon.h"
#include "../common/common_shared.h"
#include "../material/DeviceDisneyMaterial.h"

using namespace shared;

// Compute importance for a single triangle based on its area and emittance
CUDA_DEVICE_FUNCTION CUDA_INLINE float computeTriangleImportance(
    const Vertex* vertexBuffer,
    const Triangle* triangleBuffer,
    uint32_t triIndex,
    const DisneyData* material) 
{
    const Triangle& tri = triangleBuffer[triIndex];
    const Vertex (&v)[3] = {
        vertexBuffer[tri.index0],
        vertexBuffer[tri.index1],
        vertexBuffer[tri.index2]
    };

    // Calculate triangle area
    Vector3D edge1 = v[1].position - v[0].position;
    Vector3D edge2 = v[2].position - v[0].position;
    Normal3D normal = cross(edge1, edge2);
    float area = 0.5f * length(normal);

    // Get emittance from material
    RGB emittance(0.0f, 0.0f, 0.0f);
    if (material) {
        // For now, use uniform emittance (no texture sampling)
        emittance = RGB(material->emittance);
        
        // TODO: Add texture sampling for spatially varying emission
        // emittance += RGB(getXYZ(tex2DLod<float4>(material->emittanceTexture, v[0].texCoord.x, v[0].texCoord.y, 0)));
        // emittance += RGB(getXYZ(tex2DLod<float4>(material->emittanceTexture, v[1].texCoord.x, v[1].texCoord.y, 0)));
        // emittance += RGB(getXYZ(tex2DLod<float4>(material->emittanceTexture, v[2].texCoord.x, v[2].texCoord.y, 0)));
        // emittance /= 3;
    }

    // Compute importance as luminance * area
    float luminance = 0.2126f * emittance.r + 0.7152f * emittance.g + 0.0722f * emittance.b;
    float importance = luminance * area;
    
    // Ensure valid value
    if (!isfinite(importance)) {
        importance = 0.0f;
    }
    
    return importance;
}

// Kernel to compute importance values for all triangles in a surface
CUDA_DEVICE_KERNEL void computeTriangleImportances(
    const Vertex* vertexBuffer,
    const Triangle* triangleBuffer,
    uint32_t numTriangles,
    const DisneyData* material,
    float* importanceBuffer) 
{
    uint32_t triIndex = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (triIndex < numTriangles) {
        float importance = computeTriangleImportance(
            vertexBuffer, triangleBuffer, triIndex, material);
        importanceBuffer[triIndex] = importance;
    }
}

// Build CDF from importance values
CUDA_DEVICE_KERNEL void buildCDF(
    const float* importances,
    uint32_t numValues,
    float* cdf,
    float* integral) 
{
    // Simple serial scan for now - can be optimized with parallel scan
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < numValues; ++i) {
            sum += importances[i];
            cdf[i] = sum;
        }
        
        // Normalize CDF
        if (sum > 0.0f) {
            float invSum = 1.0f / sum;
            for (uint32_t i = 0; i < numValues; ++i) {
                cdf[i] *= invSum;
            }
        }
        
        *integral = sum;
    }
}

// Build probability texture for 2D sampling
CUDA_DEVICE_KERNEL void buildProbabilityTexture2D(
    const float* importances,
    uint32_t numValues,
    uint32_t texWidth,
    uint32_t texHeight,
    float* texData) 
{
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (x < texWidth && y < texHeight) {
        uint32_t linearIdx = y * texWidth + x;
        float value = 0.0f;
        
        if (linearIdx < numValues) {
            value = importances[linearIdx];
        }
        
        texData[linearIdx] = value;
    }
}

// Compute geometry instance importance (sum of triangle importances)
CUDA_DEVICE_KERNEL void computeGeomInstImportances(
    const float* triangleIntegrals,
    const uint32_t* surfaceTriangleCounts,
    uint32_t numSurfaces,
    float* geomInstImportances) 
{
    uint32_t surfaceIdx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (surfaceIdx < numSurfaces) {
        // For each surface, sum up its triangle importances
        // This assumes triangleIntegrals contains the integral for each surface's distribution
        geomInstImportances[surfaceIdx] = triangleIntegrals[surfaceIdx];
    }
}

// Compute instance importance based on transform scale
CUDA_DEVICE_KERNEL void computeInstanceImportances(
    const float* geomInstIntegrals,
    const float* instanceScales,  // Scale factors from transform matrices
    uint32_t numInstances,
    float* instanceImportances) 
{
    uint32_t instIdx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (instIdx < numInstances) {
        // Instance importance = geometry importance * scale²
        float scale = instanceScales[instIdx];
        instanceImportances[instIdx] = geomInstIntegrals[instIdx] * scale * scale;
    }
}

// Export kernel function pointers for PTX generation
extern "C" {
    __global__ void computeTriangleImportancesKernel(
        const Vertex* vertexBuffer,
        const Triangle* triangleBuffer,
        uint32_t numTriangles,
        const DisneyData* material,
        float* importanceBuffer) 
    {
        computeTriangleImportances(vertexBuffer, triangleBuffer, numTriangles, material, importanceBuffer);
    }
    
    __global__ void buildCDFKernel(
        const float* importances,
        uint32_t numValues,
        float* cdf,
        float* integral) 
    {
        buildCDF(importances, numValues, cdf, integral);
    }
    
    __global__ void buildProbabilityTexture2DKernel(
        const float* importances,
        uint32_t numValues,
        uint32_t texWidth,
        uint32_t texHeight,
        float* texData) 
    {
        buildProbabilityTexture2D(importances, numValues, texWidth, texHeight, texData);
    }
    
    __global__ void computeGeomInstImportancesKernel(
        const float* triangleIntegrals,
        const uint32_t* surfaceTriangleCounts,
        uint32_t numSurfaces,
        float* geomInstImportances) 
    {
        computeGeomInstImportances(triangleIntegrals, surfaceTriangleCounts, numSurfaces, geomInstImportances);
    }
    
    __global__ void computeInstanceImportancesKernel(
        const float* geomInstIntegrals,
        const float* instanceScales,
        uint32_t numInstances,
        float* instanceImportances) 
    {
        computeInstanceImportances(geomInstIntegrals, instanceScales, numInstances, instanceImportances);
    }
}

#endif