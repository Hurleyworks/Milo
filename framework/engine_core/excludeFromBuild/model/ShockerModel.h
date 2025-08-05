#pragma once

// ShockerModel.h
// Redesigned to follow the correct pattern from common_host.h:
// - Creates multiple GeometryInstances (one per surface)
// - Creates a GeometryGroup containing all GeometryInstances
// - Does NOT own Instance - Instances are created separately

#include "../RenderContext.h"
#include "../common/common_host.h"
#include "../milo_shared.h"

#include <sabi_core/sabi_core.h>

using sabi::RenderableNode;
using sabi::SpaceTime;
using sabi::CgModel;
using sabi::CgModelPtr;

// Forward declarations
class ShockerModelHandler;
class ShockerMaterialHandler;

// Shared pointer type for ShockerModel
using ShockerModelPtr = std::shared_ptr<class ShockerModel>;

// Geometry type enumeration
enum class ShockerGeometryType
{
    Triangle,
    Curve,
    TFDM,
    NRTDSM,
    Flyweight,
    Phantom
};

// Base class for all Shocker models
// Creates GeometryInstances for each surface and groups them
// Each GeometryInstance will have one DisneyMaterial (not shared::Material)
class ShockerModel
{
public:
    ShockerModel() = default;
    virtual ~ShockerModel() = default;

    // Core interface - must be implemented by derived classes
    virtual ShockerGeometryType getGeometryType() const = 0;
    
    // Create all geometry from the RenderableNode
    virtual void createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder) = 0;
    
    // Get the GeometryGroup for this model
    GeometryGroup* getGeometryGroup() { return geometryGroup_.get(); }
    const GeometryGroup* getGeometryGroup() const { return geometryGroup_.get(); }
    
    // Get all GeometryInstances
    const std::vector<std::unique_ptr<GeometryInstance>>& getGeometryInstances() const { 
        return geometryInstances_; 
    }
    
    // Get all geometry instance slots
    std::vector<uint32_t> getGeomInstSlots() const;
    
    // Transform utilities
    static Matrix4x4 convertSpaceTimeToMatrix(const SpaceTime& st);
    static Matrix3x3 calculateNormalMatrix(const Matrix4x4& transform);
    
    // Get combined AABB
    const AABB& getAABB() const { return combinedAABB_; }
    
    // Get source node
    sabi::Renderable* getSourceNode() const { return sourceNode_; }
    
protected:
    // Helper to calculate combined AABB from all geometry instances
    void calculateCombinedAABB();
    
protected:
    // Geometry instances (one per surface/material)
    std::vector<std::unique_ptr<GeometryInstance>> geometryInstances_;
    
    // Geometry group containing all instances
    std::unique_ptr<GeometryGroup> geometryGroup_;
    
    // Combined bounding box
    AABB combinedAABB_;
    
    // Reference to the source node (not owned)
    sabi::Renderable* sourceNode_ = nullptr;
};

// Model that can contain mixed geometry types
// Named TriangleModel for compatibility but can handle all geometry types
class ShockerTriangleModel : public ShockerModel
{
public:
    static ShockerModelPtr create()
    {
        return std::make_shared<ShockerTriangleModel>();
    }
    
    ShockerTriangleModel() = default;
    ~ShockerTriangleModel() override = default;
    
    // ShockerModel interface implementation
    // Returns the primary geometry type (but can contain mixed types)
    ShockerGeometryType getGeometryType() const override { 
        return ShockerGeometryType::Triangle; 
    }
    
    void createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder) override;
    
private:
    // Helper to create appropriate geometry based on surface properties
    void createGeometryForSurface(
        const CgModelPtr& model,
        size_t surfaceIndex,
        GeometryInstance* geomInst);
    
    // Helper to extract triangle geometry for a specific surface
    void extractTriangleGeometry(
        const CgModelPtr& model,
        size_t surfaceIndex,
        std::vector<shared::Vertex>& vertices,
        std::vector<shared::Triangle>& triangles,
        std::vector<uint32_t>& materialIndices);
    
    // Helper to extract curve geometry for a specific surface
    void extractCurveGeometry(
        const CgModelPtr& model,
        size_t surfaceIndex,
        GeometryInstance* geomInst);
    
    // Helper to calculate AABB for vertices
    AABB calculateAABBForVertices(const std::vector<shared::Vertex>& vertices);
    
    // Helper to determine geometry type for a surface
    bool shouldUseCurveGeometry(const CgModelPtr& model, size_t surfaceIndex) const;
    bool shouldUseDisplacementGeometry(const CgModelPtr& model, size_t surfaceIndex) const;
};

// Flyweight model for instancing (references another model's geometry)
class ShockerFlyweightModel : public ShockerModel
{
public:
    static ShockerModelPtr create()
    {
        return std::make_shared<ShockerFlyweightModel>();
    }
    
    ShockerGeometryType getGeometryType() const override { 
        return ShockerGeometryType::Flyweight; 
    }
    
    void createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder) override;
    
    // Set the source model this flyweight references
    void setSourceModel(ShockerModel* sourceModel) { sourceModel_ = sourceModel; }
    ShockerModel* getSourceModel() const { return sourceModel_; }
    
private:
    // Reference to the source model (not owned)
    ShockerModel* sourceModel_ = nullptr;
};

// Phantom model for collision-free instances
class ShockerPhantomModel : public ShockerModel
{
public:
    static ShockerModelPtr create()
    {
        return std::make_shared<ShockerPhantomModel>();
    }
    
    ShockerGeometryType getGeometryType() const override { 
        return ShockerGeometryType::Phantom; 
    }
    
    void createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder) override;
};