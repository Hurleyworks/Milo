// OptiXModel2 - Experimental version using TriangleMesh and TriangleMeshSurface structures
// This demonstrates how the production OptiXModel pattern could work with the 
// TriangleMeshHandler's data structures for better integration.
//
// Key differences from original OptiXModel:
// - Uses TriangleMesh structure which contains GAS and surfaces
// - Uses TriangleMeshSurface for per-surface data
// - Separates geometry ownership from instancing more clearly
// - Shows how flyweight pattern works with the handler structures
#pragma once

#include "../RenderContext.h"
#include "../handlers/TriangleMeshHandler.h"

using Eigen::AlignedBox3f;
using sabi::RenderableNode;
using sabi::SpaceTime;

using OptiXModel2Ptr = std::shared_ptr<class OptiXModel2>;

constexpr uint32_t MATERIAL_SETS_V2 = 3;

// Base class for all OptiX models
class OptiXModel2
{
public:
    virtual ~OptiXModel2()
    {
        instance.destroy();
    }

    virtual void createGeometryAndMaterials(RenderContextPtr ctx, RenderableNode& node) {}
    virtual void extractVertexPositions(MatrixXf& V) {}
    virtual void extractTriangleIndices(MatrixXu& F) {}
    
    // Get the triangle mesh (may be null for flyweights)
    virtual TriangleMesh* getTriangleMesh() = 0;
    
    // Check if this model owns geometry
    virtual bool ownsGeometry() const = 0;
    
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

protected:
    optixu::Instance instance;
};

// OptiXTriangleModel2 - Owns actual geometry using TriangleMesh structure
class OptiXTriangleModel2 : public OptiXModel2
{
public:
    static OptiXModel2Ptr create()
    {
        return std::make_shared<OptiXTriangleModel2>();
    }
    
    ~OptiXTriangleModel2()
    {
        // TriangleMesh destructor handles cleanup
    }
    
    // Creates geometry and materials from node data
    void createGeometryAndMaterials(RenderContextPtr ctx, RenderableNode& node) override;
    
    // Extract vertex positions into Eigen matrix
    void extractVertexPositions(MatrixXf& V) override;
    
    // Extract triangle indices into Eigen matrix
    void extractTriangleIndices(MatrixXu& F) override;
    
    // Get the owned triangle mesh
    TriangleMesh* getTriangleMesh() override { return &triangleMesh; }
    
    // This model owns its geometry
    bool ownsGeometry() const override { return true; }
    
    // Enable deformation support
    void enableDeformation(RenderContextPtr ctx);
    
    // Check if deformation is enabled
    bool hasDeformation() const { return originalVertexBuffer.isInitialized(); }
    
    // Get original undeformed vertices
    const cudau::TypedBuffer<shared::Vertex>& getOriginalVertexBuffer() const
    {
        return originalVertexBuffer;
    }
    
    // Get current (possibly deformed) vertices
    const cudau::TypedBuffer<shared::Vertex>& getCurrentVertexBuffer() const
    {
        return triangleMesh.vertex_buffer;
    }
    
    // Update geometry after deformation
    void updateDeformedGeometry(RenderContextPtr ctx);
    
    // Get material at specific index
    optixu::Material getMaterialAt(uint32_t surfaceIdx, uint32_t matSetIdx = 0);
    
    // Get light distribution for emissive triangles
    LightDistribution* getTriLightImportance() 
    {
        return &emitterPrimDist;
    }

private:
    // Build the GAS from surfaces
    void buildGAS(RenderContextPtr ctx, uint32_t numRayTypes);
    
private:
    TriangleMesh triangleMesh;  // Contains GAS, vertices, and surfaces
    cudau::TypedBuffer<shared::Vertex> originalVertexBuffer;  // For deformation
    cudau::TypedBuffer<uint8_t> materialIndexBuffer;  // Per-triangle material IDs
    LightDistribution emitterPrimDist;
};

// OptiXFlyweightModel2 - References shared geometry, doesn't own it
class OptiXFlyweightModel2 : public OptiXModel2
{
public:
    static OptiXModel2Ptr create(TriangleMesh* sharedMesh = nullptr)
    {
        auto model = std::make_shared<OptiXFlyweightModel2>();
        model->setSharedMesh(sharedMesh);
        return model;
    }
    
    // Set the shared mesh this flyweight references
    void setSharedMesh(TriangleMesh* mesh)
    {
        sharedTriangleMesh = mesh;
    }
    
    // Get the referenced triangle mesh (not owned)
    TriangleMesh* getTriangleMesh() override { return sharedTriangleMesh; }
    
    // Flyweights don't own geometry
    bool ownsGeometry() const override { return false; }
    
private:
    TriangleMesh* sharedTriangleMesh = nullptr;  // Pointer to shared geometry
};

// OptiXPhantomModel2 - For collision-free instance painting
class OptiXPhantomModel2 : public OptiXModel2
{
public:
    static OptiXModel2Ptr create()
    {
        return std::make_shared<OptiXPhantomModel2>();
    }
    
    TriangleMesh* getTriangleMesh() override { return nullptr; }
    bool ownsGeometry() const override { return false; }
};

// Helper class to manage geometry sharing
class GeometryInstanceManager
{
public:
    // Create a geometry-owning model
    OptiXModel2Ptr createGeometryOwner(RenderContextPtr ctx, RenderableNode& node)
    {
        auto model = OptiXTriangleModel2::create();
        model->createGeometryAndMaterials(ctx, node);
        
        // Store for potential sharing
        auto* mesh = model->getTriangleMesh();
        if (mesh)
        {
            geometryOwners[node->getClientID()] = model;
        }
        
        return model;
    }
    
    // Create a flyweight that shares existing geometry
    OptiXModel2Ptr createFlyweight(uint64_t geometryOwnerId)
    {
        auto it = geometryOwners.find(geometryOwnerId);
        if (it != geometryOwners.end())
        {
            auto* mesh = it->second->getTriangleMesh();
            return OptiXFlyweightModel2::create(mesh);
        }
        
        // Fallback to phantom if geometry not found
        return OptiXPhantomModel2::create();
    }
    
    // Check if geometry exists
    bool hasGeometry(uint64_t id) const
    {
        return geometryOwners.find(id) != geometryOwners.end();
    }
    
private:
    std::unordered_map<uint64_t, OptiXModel2Ptr> geometryOwners;
};