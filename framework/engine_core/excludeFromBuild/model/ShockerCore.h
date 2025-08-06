#pragma once

// ShockerCore.h
// Shocker-specific versions of geometry structures that use DisneyMaterial directly
// These are exact copies of the structures from common_host.h but with DisneyMaterial

#include "../../engine_core.h"
#include "../material/HostDisneyMaterial.h"
#include "../common/common_host.h"

namespace shocker {

// Exact copy of GeometryInstance but with DisneyMaterial instead of Material
struct ShockerSurface
{
    const ::DisneyMaterial* mat;

    uint32_t geomInstSlot;
    optixu::GeometryInstance optixGeomInst;
    std::variant<TriangleGeometry, CurveGeometry, TFDMGeometry, NRTDSMGeometry> geometry;
    AABB aabb;

    void finalize()
    {
        optixGeomInst.destroy();
    }
};

// Exact copy of GeometryGroup but referencing ShockerSurface
struct ShockerSurfaceGroup
{
    std::set<const ShockerSurface*> geomInsts;

    optixu::GeometryAccelerationStructure optixGas;
    cudau::Buffer optixGasMem;
    uint32_t numEmitterPrimitives;
    AABB aabb;
    uint32_t needsReallocation : 1;
    uint32_t needsRebuild : 1;
    uint32_t refittable : 1;
};

// Exact copy of Mesh but referencing ShockerSurfaceGroup
struct ShockerMesh
{
    struct ShockerSurfaceGroupInstance
    {
        const ShockerSurfaceGroup* geomGroup;
        Matrix4x4 transform;
    };
    std::vector<ShockerSurfaceGroupInstance> groupInsts;
};

// Exact copy of Instance but referencing ShockerMesh
struct ShockerNode
{
    ShockerMesh::ShockerSurfaceGroupInstance geomGroupInst;

    cudau::TypedBuffer<uint32_t> geomInstSlots;
    LightDistribution lightGeomInstDist;
    uint32_t instSlot;
    optixu::Instance optixInst;

    Matrix4x4 prevMatM2W;
    Matrix4x4 matM2W;
    Matrix3x3 nMatM2W;
};

} // namespace shocker