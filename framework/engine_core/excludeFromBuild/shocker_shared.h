#pragma once

// shocker_shared.h
// Shared definitions between host and device code for ShockerEngine
// Simplified version without callable programs, using Disney material directly

#include "common/common_shared.h"
#include "model/ShockerDeviceCore.h"

namespace shocker_shared {

// Import core types from shocker namespace
using namespace shocker;

// Shocker-specific structures
struct PerspectiveCamera {
    float aspect;
    float fovY;
    Point3D position;
    Matrix3x3 orientation;
    float lensSize = 0.0f;
    float focusDistance = 5.0f;
};

struct LightSample {
    RGB emittance;
    Point3D position;
    Normal3D normal;
    uint32_t atInfinity : 1;
};

struct PickInfo {
    Point3D positionInWorld;
    Normal3D normalInWorld;
    Point3D rayOrigin;
    Vector3D rayDir;
    uint32_t instanceIndex;
    uint32_t matIndex;
    uint32_t primIndex;
    uint32_t hit : 1;
};

struct HitPointParams {
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

// Buffer display modes for debugging
enum class BufferToDisplay {
    NoisyBeauty = 0,
    Albedo,
    Normal,
    Depth,
    Motion,
    DenoisedBeauty,
};

// Render modes
enum class RenderMode {
    GBufferPreview = 0,   // Fast preview using G-buffer pipeline
    PathTraceFinal,       // Full path tracing
    DebugNormals,        // Visualize normals from G-buffer
    DebugAlbedo,         // Visualize albedo from G-buffer
    DebugDepth,          // Visualize depth from G-buffer
    DebugMotion,         // Visualize motion vectors
};

} // namespace shocker_shared

// Device-side structures and helper functions
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#include "common/deviceCommon.h"

namespace shocker_shared {

// Hit group SBT record data
struct HitGroupSBTRecordData {
    uint32_t geomInstSlot;
    
    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData& get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};

// Hit point parameter helper
struct HitPointParameter {
    float bcB, bcC;
    int32_t primIndex;
    
    CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get() {
        HitPointParameter ret;
        const float2 bc = optixGetTriangleBarycentrics();
        ret.bcB = bc.x;
        ret.bcC = bc.y;
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

// Compute surface point from barycentric coordinates
CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint(
    const ShockerInstanceData& inst,
    const ShockerGeometryInstanceData& geomInst,
    uint32_t primIndex, float bcB, float bcC,
    Point3D* positionInWorld, 
    Normal3D* shadingNormalInWorld, 
    Vector3D* texCoord0DirInWorld,
    Normal3D* geometricNormalInWorld, 
    Point2D* texCoord) {
    
    const shared::Triangle& tri = geomInst.triangleBuffer[primIndex];
    const shared::Vertex& vA = geomInst.vertexBuffer[tri.index0];
    const shared::Vertex& vB = geomInst.vertexBuffer[tri.index1];
    const shared::Vertex& vC = geomInst.vertexBuffer[tri.index2];
    const float bcA = 1 - (bcB + bcC);
    
    // Compute position in world space
    const Point3D positionInObj = bcA * vA.position + bcB * vB.position + bcC * vC.position;
    *positionInWorld = inst.transform * positionInObj;
    
    // Compute geometric normal
    *geometricNormalInWorld = normalize(
        inst.normalMatrix * Normal3D(cross(vB.position - vA.position, vC.position - vA.position)));
    
    // Interpolate shading normal
    const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
    *shadingNormalInWorld = normalize(inst.normalMatrix * shadingNormalInObj);
    
    // Interpolate texture coordinates
    *texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;
    
    // Compute texture coordinate direction
    const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
    *texCoord0DirInWorld = inst.transform * texCoord0DirInObj;
    *texCoord0DirInWorld = normalize(
        *texCoord0DirInWorld - dot(*shadingNormalInWorld, *texCoord0DirInWorld) * *shadingNormalInWorld);
    
    // Handle degenerate cases
    if (!shadingNormalInWorld->allFinite()) {
        *geometricNormalInWorld = Normal3D(0, 0, 1);
        *shadingNormalInWorld = Normal3D(0, 0, 1);
        *texCoord0DirInWorld = Vector3D(1, 0, 0);
    }
    if (!texCoord0DirInWorld->allFinite()) {
        Vector3D bitangent;
        makeCoordinateSystem(*shadingNormalInWorld, texCoord0DirInWorld, &bitangent);
    }
}

// Check if current pixel is under cursor
CUDA_DEVICE_FUNCTION CUDA_INLINE bool isCursorPixel() {
    return plp.f->mousePosition == make_int2(optixGetLaunchIndex());
}

// Get debug print status
CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled() {
    return plp.f->renderMode >= static_cast<uint32_t>(RenderMode::DebugNormals);
}

} // namespace shocker_shared

#endif // __CUDA_ARCH__