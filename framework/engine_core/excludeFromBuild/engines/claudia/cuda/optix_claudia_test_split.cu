// optix_claudia_test_split.cu
// Test kernel to verify split parameter structure works correctly


#if 0

#include "principledDisney_claudia.h"
#include "../claudia_shared.h"

using namespace claudia_shared;

// The global declaration of claudia_plp_split is now in claudia_shared.h

// Simple test ray generation program using split parameters
CUDA_DEVICE_KERNEL void RT_RG_NAME(testSplit)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    
    // Test accessing static parameters through s pointer
    const int2 imageSize = claudia_plp_split.s->imageSize;
    
    // Test accessing per-frame parameters through f pointer
    const uint32_t numFrames = claudia_plp_split.f->numAccumFrames;
    const OptixTraversableHandle travHandle = claudia_plp_split.f->travHandle;
    
    // Test accessing camera through f pointer
    const PerspectiveCamera& camera = claudia_plp_split.f->camera;
    
    // Simple test: just write pixel coordinates to verify it works
    if (launchIndex.x == 0 && launchIndex.y == 0) {
        printf("ClaudiaSplit TEST: imageSize=(%d,%d), numFrames=%u, travHandle=%llu\n",
               imageSize.x, imageSize.y, numFrames, travHandle);
        printf("ClaudiaSplit TEST: camera pos=(%.2f,%.2f,%.2f)\n",
               camera.position.x, camera.position.y, camera.position.z);
    }
    
    // Test buffer access through s pointer
    PCG32RNG rng = claudia_plp_split.s->rngBuffer.read(launchIndex);
    // rng.next();  // Method doesn't exist - RNG uses getFloat0cTo1o() etc
    float random = rng.getFloat0cTo1o(); // Get a random value to advance state
    claudia_plp_split.s->rngBuffer.write(launchIndex, rng);
    
    // Test writing to accumulation buffer
    float4 testColor = make_float4(
        float(launchIndex.x) / float(imageSize.x),
        float(launchIndex.y) / float(imageSize.y),
        0.5f, 1.0f
    );
    claudia_plp_split.s->colorAccumBuffer.write(launchIndex, testColor);
}

// Simple closest hit for testing
CUDA_DEVICE_KERNEL void RT_CH_NAME(testSplit)() {
    // Just a stub for now
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    
    // Test accessing buffer index from per-frame params
    const uint32_t bufIdx = claudia_plp_split.f->bufferIndex;
    
    // Test accessing instance data through static params
    const auto instId = optixGetInstanceId();
    if (instId < 1) {  // Safety check
        const shared::InstanceData& inst = claudia_plp_split.s->instanceDataBufferArray[bufIdx][instId];
        // Just accessing it to verify compilation
    }
}

// Simple miss for testing  
CUDA_DEVICE_KERNEL void RT_MS_NAME(testSplit)() {
    // Test accessing environment settings from per-frame params
    if (claudia_plp_split.f->enableEnvLight) {
        const float rotation = claudia_plp_split.f->envLightRotation;
        const float powerCoeff = claudia_plp_split.f->envLightPowerCoeff;
        
        // Test accessing environment texture from static params
        if (claudia_plp_split.s->envLightTexture) {
            // Just checking it exists
        }
    }
}

#endif