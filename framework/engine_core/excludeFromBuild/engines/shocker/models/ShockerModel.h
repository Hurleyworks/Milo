#pragma once

// ShockerModel.h
// Base class and hierarchy for Shocker-specific model management
// Preserves the OO design of OptiXModel while adapting to the new architecture

#include "../../../RenderContext.h"
#include "../../../common/common_host.h"
#include "../shocker_shared.h"

using Eigen::AlignedBox3f;
using sabi::RenderableNode;
using sabi::SpaceTime;

using ShockerModelPtr = std::shared_ptr<class ShockerModel>;

namespace ShockerConstants {
    constexpr uint32_t MATERIAL_SETS = 3;
}

// Base class for all Shocker models
class ShockerModel
{
public:
    virtual ~ShockerModel()
    {
        instance.destroy();
    }

    virtual void createGeometry(RenderContextPtr ctx, RenderableNode& node, optixu::Scene* scene) {}

    virtual void extractVertexPositions(MatrixXf& V) {}
    virtual void extractTriangleIndices(MatrixXu& F) {}

    // Create a Geometry Acceleration Structure, not all derived classes have geometry
    virtual void createGAS(RenderContextPtr ctx, optixu::Scene* scene, uint32_t numRayTypes) {}
    virtual GAS* getGAS() = 0;
    virtual optixu::GeometryInstance* getGeometryInstance() = 0;

    optixu::Instance& getOptiXInstance() { return instance; }
    void setOptiXInstance(const optixu::Instance& inst) { instance = inst; }

    void setVisibility(uint32_t mask)
    {
        instance.setVisibilityMask(mask);
    }

    uint32_t getVisibility() const
    {
        return instance.getVisibilityMask();
    }
    
    // Get the geometry instance slot
    uint32_t getGeomInstSlot() const { return geomInstSlot_; }
    
    // Set the geometry instance slot (called by handler during allocation)
    void setGeomInstSlot(uint32_t slot) { geomInstSlot_ = slot; }
    
    // Virtual method to populate geometry instance data in the global buffer
    // Derived classes should implement this based on their geometry type
    virtual void populateGeometryInstanceData(shared::GeometryInstanceData* geomInstData) {}

protected:
    optixu::Instance instance;
    
    // Geometry instance slot in the global buffer
    uint32_t geomInstSlot_ = SlotFinder::InvalidSlotIndex;
};

// High-performance Shocker-based triangle mesh model supporting static and dynamic geometry,
// material properties, and efficient acceleration structure updates
class ShockerTriangleModel : public ShockerModel
{
public:
    static ShockerModelPtr create()
    {
        return std::make_shared<ShockerTriangleModel>();
    }

    ~ShockerTriangleModel()
    {
        gasData.gas.destroy();
        gasData.gasMem.finalize();
        geomInst.destroy();
        materialIndexBuffer.finalize();
        triangleBuffer.finalize();
        vertexBuffer.finalize();
        if (originalVertexBuffer.isInitialized())
            originalVertexBuffer.finalize();
    }

    // Creates and sets up geometry from provided node data
    void createGeometry(RenderContextPtr ctx, RenderableNode& node, optixu::Scene* scene) override;

    // Creates acceleration structure for this geometry
    void createGAS(RenderContextPtr ctx, optixu::Scene* scene, uint32_t numRayTypes) override;

    // Extracts vertex positions into an Eigen matrix
    void extractVertexPositions(MatrixXf& V) override;

    // Extracts triangle indices into an Eigen matrix
    void extractTriangleIndices(MatrixXu& F) override;

    // Gets the GAS data for this model
    GAS* getGAS() override { return &gasData; }

    // Gets the geometry instance
    optixu::GeometryInstance* getGeometryInstance() override { return &geomInst; }

    // Creates a copy of the original vertex buffer for deformation
    void enableDeformation(RenderContextPtr ctx)
    {
        if (vertexBuffer.numElements() == 0)
            return;

        LOG(DBUG) << _FN_;
        originalVertexBuffer = vertexBuffer.copy();
    }

    // Returns true if this model has deformation enabled
    bool hasDeformation() const
    {
        return originalVertexBuffer.isInitialized();
    }

    // Gets the original undeformed vertex buffer
    const cudau::TypedBuffer<shared::Vertex>& getOriginalVertexBuffer() const
    {
        return originalVertexBuffer;
    }

    // Gets the current vertex buffer which may be deformed
    const cudau::TypedBuffer<shared::Vertex>& getCurrentVertexBuffer() const
    {
        return vertexBuffer;
    }

    // Gets the triangle buffer
    const cudau::TypedBuffer<shared::Triangle>& getTriangleBuffer() const
    {
        return triangleBuffer;
    }

    // Updates vertex positions and normals after deformation
    void updateDeformedGeometry(RenderContextPtr ctx)
    {
        if (!hasDeformation())
            return;

        // Rebuild GAS after vertex modifications
        GAS* gas = getGAS();
        if (gas)
            gas->gas.rebuild(ctx->getCudaStream(), gas->gasMem, ctx->getASBuildScratchMem());
    }

    optixu::Material getMaterialAt(uint32_t matSetIdx, uint32_t matIdx)
    {
        return geomInst.getMaterial(matSetIdx, matIdx);
    }

    LightDistribution* getTriLightImportance() 
    {
        return &emitterPrimDist;
    }
    
    // Get emitter primitive distribution for area light support
    const LightDistribution& getEmitterPrimDistribution() const
    {
        return emitterPrimDist;
    }
    
    // Populate geometry instance data for triangle mesh
    void populateGeometryInstanceData(shared::GeometryInstanceData* geomInstData) override
    {
        if (!geomInstData) return;
        
        // Populate vertex and triangle buffers
        geomInstData->vertexBuffer = vertexBuffer.getROBuffer<shared::enableBufferOobCheck>();
        geomInstData->triangleBuffer = triangleBuffer.getROBuffer<shared::enableBufferOobCheck>();
        
        // Convert light distribution to device format
        emitterPrimDist.getDeviceType(&geomInstData->emitterPrimDist);
        
        // Material slot will be set separately when materials are assigned
        // For now, use invalid slot as we don't have the material yet
        geomInstData->materialSlot = SlotFinder::InvalidSlotIndex;
        
        // Store geometry instance slot
        geomInstData->geomInstSlot = geomInstSlot_;
    }
    
    // Update material slot in geometry instance data
    void updateMaterialSlot(uint32_t slot, shared::GeometryInstanceData* geomInstData)
    {
        if (geomInstData)
        {
            geomInstData->materialSlot = slot;
        }
    }

private:
    GAS gasData;
    optixu::GeometryInstance geomInst;
    cudau::TypedBuffer<uint8_t> materialIndexBuffer;
    cudau::TypedBuffer<shared::Vertex> vertexBuffer;
    cudau::TypedBuffer<shared::Vertex> originalVertexBuffer;
    cudau::TypedBuffer<shared::Triangle> triangleBuffer;
    LightDistribution emitterPrimDist;
};

// Flyweight is used for geometry instancing. It has no geometry of its own
class ShockerFlyweightModel : public ShockerModel
{
public:
    static ShockerModelPtr create()
    {
        return std::make_shared<ShockerFlyweightModel>();
    }

    GAS* getGAS() override { return nullptr; }
    optixu::GeometryInstance* getGeometryInstance() override { return nullptr; }
};

// Phantom is used for collision-free instance painting
class ShockerPhantomModel : public ShockerModel
{
public:
    static ShockerModelPtr create()
    {
        return std::make_shared<ShockerPhantomModel>();
    }

    GAS* getGAS() override { return nullptr; }
    optixu::GeometryInstance* getGeometryInstance() override { return nullptr; }
};