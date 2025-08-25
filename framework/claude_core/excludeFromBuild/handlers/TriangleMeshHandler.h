#pragma once

#include "../RenderContext.h"
#include "../common/common_host.h"
#include <sabi_core/sabi_core.h>

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

    // Create a GeometryInstance from CgModel surface
    GeometryInstance* createGeometryInstance(
        const sabi::CgModel& model,
        uint32_t surfaceIndex,
        uint32_t materialSlot);

    // Create and build a GeometryGroup with GAS
    GeometryGroup* createGeometryGroup(
        const std::set<GeometryInstance*>& instances);

    // Build or rebuild GAS
    void buildGAS(GeometryGroup* group, CUstream stream = 0);
    OptixTraversableHandle getTraversableHandle(GeometryGroup* group) const;

    // Access geometry instance data
    shared::GeometryInstanceData* getGeometryInstanceData(uint32_t slot);
    cudau::TypedBuffer<shared::GeometryInstanceData>& getGeometryDataBuffer() 
    { 
        return geomInstDataBuffer_; 
    }

    // Cleanup
    void destroyGeometryInstance(GeometryInstance* instance);
    void destroyGeometryGroup(GeometryGroup* group);

private:
    RenderContextPtr renderContext_;
    CUcontext cudaContext_;
    
    // Geometry instance data buffer (mapped for host/device access)
    static constexpr uint32_t maxNumGeometryInstances = 1024;
    cudau::TypedBuffer<shared::GeometryInstanceData> geomInstDataBuffer_;
    
    // Slot management
    SlotFinder geomInstSlotFinder_;
    
    // Track created resources for cleanup
    std::vector<GeometryInstance*> geometryInstances_;
    std::vector<GeometryGroup*> geometryGroups_;
};