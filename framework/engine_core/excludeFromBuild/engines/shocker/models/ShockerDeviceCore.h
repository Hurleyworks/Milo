#pragma once

// ShockerDeviceCore.h
// Device-side Shocker-specific structures that correspond to the host-side ShockerCore.h
// These are the device equivalents of ShockerSurface and ShockerNode

#include "../../../common/common_shared.h"
#include "../../../material/DeviceDisneyMaterial.h"

namespace shocker {

// Device-side equivalent of ShockerSurface (host-side geometry instance)
// This corresponds to GeometryInstanceData in common_shared.h but with Disney material
struct ShockerSurfaceData {
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
    uint32_t disneyMaterialSlot;  // Index into Disney material buffer (instead of generic materialSlot)
    uint32_t geomInstSlot;
};

// Device-side equivalent of ShockerNode (host-side instance)
// This corresponds to InstanceData in common_shared.h
struct ShockerNodeData {
    Matrix4x4 transform;
    Matrix4x4 curToPrevTransform;
    Matrix3x3 normalMatrix;
    float uniformScale;
    
    shared::ROBuffer<uint32_t> geomInstSlots;
    shared::LightDistribution lightGeomInstDist;
    
    // Area light support
    uint32_t isEmissive : 1;
    float emissiveScale;
};

} // namespace shocker