#pragma once

// ShockerDeviceCore.h
// Device-side Shocker-specific structures that correspond to the host-side ShockerCore.h
// These structures use DisneyMaterial data directly instead of generic material data

#include "../common/common_shared.h"
#include "../material/DeviceDisneyMaterial.h"

namespace shocker {

// Shocker-specific geometry instance data with Disney material
struct ShockerGeometryInstanceData {
    union {
        struct {
            shared::ROBuffer<shared::Vertex> vertexBuffer;
            shared::ROBuffer<shared::Triangle> triangleBuffer;
        };
        struct {
            shared::ROBuffer<shared::CurveVertex> curveVertexBuffer;
            shared::ROBuffer<uint32_t> segmentIndexBuffer;
        };
    };
    shared::LightDistribution emitterPrimDist;
    uint32_t disneyMaterialSlot;  // Index into Disney material buffer
    uint32_t geomInstSlot;
};

// Pick info structure for mouse interaction
struct ShockerPickInfo {
    Point3D positionInWorld;
    Normal3D normalInWorld;
    Point3D rayOrigin;
    Vector3D rayDir;
    uint32_t instanceIndex;
    uint32_t matIndex;
    uint32_t primIndex;
    uint32_t hit : 1;
};

// Shocker-specific instance data
struct ShockerInstanceData {
    Matrix4x4 transform;
    Matrix4x4 curToPrevTransform;
    Matrix3x3 normalMatrix;
    float uniformScale;
    
    shared::ROBuffer<uint32_t> geomInstSlots;
    shared::LightDistribution lightGeomInstDist;
    
    // Area light support (matches shared::InstanceData)
    uint32_t isEmissive : 1;
    float emissiveScale;
};

// Launch parameters specific to Shocker pipelines
struct ShockerStaticPipelineLaunchParameters {
    int2 imageSize;
    
    // RNG buffer
    optixu::NativeBlockBuffer2D<shared::PCG32RNG> rngBuffer;
    
    // G-buffers (double buffered for temporal effects)
    struct GBuffer0Elements {
        uint32_t instSlot;
        uint32_t geomInstSlot; 
        uint32_t primIndex;
        uint16_t qbcB;  // Quantized barycentric B
        uint16_t qbcC;  // Quantized barycentric C
    };
    
    struct GBuffer1Elements {
        float2 motionVector;
        uint32_t materialID;
        float depth;
    };
    
    optixu::NativeBlockBuffer2D<GBuffer0Elements> GBuffer0[2];
    optixu::NativeBlockBuffer2D<GBuffer1Elements> GBuffer1[2];
    
    // Disney material and instance data
    shared::ROBuffer<shared::DisneyData> disneyMaterialBuffer;
    shared::ROBuffer<ShockerInstanceData> instanceDataBuffer;
    shared::ROBuffer<ShockerGeometryInstanceData> geometryInstanceDataBuffer;
    
    // Light data
    shared::LightDistribution lightInstDist;
    shared::RegularConstantContinuousDistribution2D envLightImportanceMap;
    CUtexObject envLightTexture;
    
    // Accumulation buffers for path tracing
    optixu::NativeBlockBuffer2D<float4> beautyAccumBuffer;
    optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
    optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
    optixu::NativeBlockBuffer2D<float2> motionAccumBuffer;
    
    // Pick info for mouse interaction
    ShockerPickInfo* pickInfos[2];
};

struct ShockerPerFramePipelineLaunchParameters {
    OptixTraversableHandle travHandle;
    uint32_t numAccumFrames;
    uint32_t frameIndex;
    
    // Camera data (Shocker-specific)
    struct ShockerPerspectiveCamera {
        float aspect;
        float fovY;
        Point3D position;
        Matrix3x3 orientation;
        float lensSize = 0.0f;
        float focusDistance = 5.0f;
    };
    
    ShockerPerspectiveCamera camera;
    ShockerPerspectiveCamera prevCamera;
    
    // Environment light
    float envLightPowerCoeff;
    float envLightRotation;
    
    // Interaction
    int2 mousePosition;
    
    // Render settings (bit flags)
    uint32_t maxPathLength : 4;      // Maximum bounces for path tracing
    uint32_t bufferIndex : 1;        // Double buffer index
    uint32_t enableJittering : 1;    // Anti-aliasing
    uint32_t enableEnvLight : 1;     // Environment lighting
    uint32_t enableDenoiser : 1;     // Denoising pass
    uint32_t renderMode : 3;          // 0=GBuffer, 1=PathTrace, 2+=Debug modes
    uint32_t enableMotionBlur : 1;   // Motion blur
    uint32_t enableDepthOfField : 1; // DOF
};

struct ShockerPipelineLaunchParameters {
    ShockerStaticPipelineLaunchParameters* s;
    ShockerPerFramePipelineLaunchParameters* f;
};

// Ray types for both pipelines
namespace GBufferRayType {
    enum Value {
        Primary = 0,
        NumTypes
    };
}

namespace PathTracingRayType {
    enum Value {
        Radiance = 0,
        Shadow = 1,
        NumTypes
    };
}

constexpr uint32_t maxNumRayTypes = 2;

// Hit point parameters
struct ShockerHitPointParams {
    RGB albedo;
    Point3D positionInWorld;
    Point3D prevPositionInWorld;
    Normal3D normalInWorld;
    Point2D texCoord;
    uint32_t instSlot;
    uint32_t geomInstSlot;
    uint32_t primIndex;
    uint16_t qbcB;
    uint16_t qbcC;
};

// Payload structures for OptiX ray tracing

// G-Buffer ray payload (simple, single bounce)
struct GBufferPayload {
    ShockerHitPointParams* hitPoint;
    ShockerPickInfo* pickInfo;
};

// Path tracing payload (complex, multi-bounce)
struct PathTracePayload {
    RGB throughput;
    RGB radiance;
    Point3D origin;
    Vector3D direction;
    shared::PCG32RNG rng;
    uint32_t pathLength : 4;
    uint32_t done : 1;
    uint32_t specularBounce : 1;
};

// Visibility/shadow ray payload
struct VisibilityPayload {
    float visibility;  // 0.0 = blocked, 1.0 = visible
};

// Payload signatures for OptiX
using ShockerGBufferRayPayloadSignature = 
    optixu::PayloadSignature<GBufferPayload>;
using ShockerPathTraceRayPayloadSignature = 
    optixu::PayloadSignature<PathTracePayload>;
using ShockerVisibilityRayPayloadSignature = 
    optixu::PayloadSignature<VisibilityPayload>;

// Constants
static constexpr float probToSampleEnvLight = 0.25f;

} // namespace shocker

// Global launch parameters for device code
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM shocker::ShockerPipelineLaunchParameters plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS shocker::ShockerPipelineLaunchParameters plp;
#endif

#endif // __CUDA_ARCH__