#pragma once

// RiPRCore.h
// RiPR-specific versions of geometry structures that use DisneyMaterial directly
// These are exact copies of the structures from common_host.h but with DisneyMaterial

#include "../../../../engine_core.h"
#include "../../../material/HostDisneyMaterial.h"
#include "../../../common/common_host.h"

namespace ripr {

// Exact copy of GeometryInstance but with DisneyMaterial instead of Material
struct RiPRSurface
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

// Exact copy of GeometryGroup but referencing RiPRSurface
struct RiPRSurfaceGroup
{
    std::set<const RiPRSurface*> geomInsts;

    optixu::GeometryAccelerationStructure optixGas;
    cudau::Buffer optixGasMem;
    uint32_t numEmitterPrimitives;
    AABB aabb;
    uint32_t needsReallocation : 1;
    uint32_t needsRebuild : 1;
    uint32_t refittable : 1;
};

// Exact copy of Mesh but referencing RiPRSurfaceGroup
struct RiPRMesh
{
    struct RiPRSurfaceGroupInstance
    {
        const RiPRSurfaceGroup* geomGroup;
        Matrix4x4 transform;
    };
    std::vector<RiPRSurfaceGroupInstance> groupInsts;
};

// Exact copy of Instance but referencing RiPRMesh
struct RiPRNode
{
    RiPRMesh::RiPRSurfaceGroupInstance geomGroupInst;

    cudau::TypedBuffer<uint32_t> geomInstSlots;
    LightDistribution lightGeomInstDist;
    uint32_t instSlot;
    optixu::Instance optixInst;

    Matrix4x4 prevMatM2W;
    Matrix4x4 matM2W;
    Matrix3x3 nMatM2W;
};

} // namespace ripr