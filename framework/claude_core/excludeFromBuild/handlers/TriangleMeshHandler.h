#pragma once

#include "../RenderContext.h"
#include "../common/common_host.h"
#include "../material/HostDisneyMaterial.h"
#include <sabi_core/sabi_core.h>

// Disney material-specific triangle mesh structures for OptiX rendering
struct OptiXTriMeshSurface
{
    uint32_t geomInstSlot;
    // For now we only support TriangleGeometry in TriangleMeshHandler
    TriangleGeometry triGeometry;
    AABB aabb;
    optixu::GeometryInstance optixGeomInst;
    DisneyMaterial* disneyMaterial;  // Direct pointer to Disney material
    
    void finalize()
    {
        triGeometry.vertexBuffer.finalize();
        triGeometry.triangleBuffer.finalize();
        if (triGeometry.emitterPrimDist.isInitialized())
            triGeometry.emitterPrimDist.finalize();
        optixGeomInst.destroy();
    }
};

struct OptiXTriMesh
{
    std::set<OptiXTriMeshSurface*> surfaces;
    AABB aabb;
    uint32_t numEmitterPrimitives = 0;
    
    optixu::GeometryAccelerationStructure optixGas;
    cudau::Buffer optixGasMem;
    
    bool needsReallocation = true;
    bool needsRebuild = true;
    bool refittable = false;
};

class TriangleMeshHandler;
using TriangleMeshHandlerPtr = std::shared_ptr<TriangleMeshHandler>;

class TriangleMeshHandler
{
public:
    static TriangleMeshHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<TriangleMeshHandler>(ctx);
    }

    explicit TriangleMeshHandler(RenderContextPtr ctx);
    ~TriangleMeshHandler();

    bool initialize();
    void finalize();

    // Create an OptiXTriMeshSurface from CgModel surface
    OptiXTriMeshSurface* createTriMeshSurface(
        const sabi::CgModel& model,
        uint32_t surfaceIndex,
        DisneyMaterial* disneyMaterial);

    // Create and build an OptiXTriMesh with GAS
    OptiXTriMesh* createTriMesh(
        const std::set<OptiXTriMeshSurface*>& surfaces);

    // Create an OptiXTriMesh from a complete CgModel with Disney materials
    OptiXTriMesh* createTriMeshFromModel(
        sabi::CgModelPtr model,
        const std::vector<DisneyMaterial*>& disneyMaterials = {});
    
    // Create an OptiXTriMesh from a CgModel, automatically creating Disney materials
    OptiXTriMesh* createTriMeshFromModelWithMaterials(
        sabi::CgModelPtr model,
        const std::filesystem::path& materialFolder = {});

    // Build or rebuild GAS
    void buildGAS(OptiXTriMesh* mesh, CUstream stream = 0);
    OptixTraversableHandle getTraversableHandle(OptiXTriMesh* mesh) const;

    // Access geometry instance data
    shared::GeometryInstanceData* getGeometryInstanceData(uint32_t slot);
    cudau::TypedBuffer<shared::GeometryInstanceData>& getGeometryDataBuffer() 
    { 
        return geomInstDataBuffer_; 
    }

    // Cleanup
    void destroyTriMeshSurface(OptiXTriMeshSurface* surface);
    void destroyTriMesh(OptiXTriMesh* mesh);

private:
    RenderContextPtr renderContext_;
    CUcontext cudaContext_;
    
    // Geometry instance data buffer (mapped for host/device access)
    static constexpr uint32_t maxNumGeometryInstances = 1024;
    cudau::TypedBuffer<shared::GeometryInstanceData> geomInstDataBuffer_;
    
    // Slot management
    SlotFinder geomInstSlotFinder_;
    
    // Track created resources for cleanup
    std::vector<OptiXTriMeshSurface*> triMeshSurfaces_;
    std::vector<OptiXTriMesh*> triMeshes_;
};