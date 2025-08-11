#pragma once

// RiPRModel.h
// Uses RiPR-specific types that work directly with DisneyMaterial
// - Creates multiple RiPRSurfaces (one per surface)
// - Creates a RiPRSurfaceGroup containing all RiPRSurfaces
// - Does NOT own RiPRNode - Nodes are created separately

#include "../../../RenderContext.h"
#include "../../../common/common_host.h"
#include "../../milo/milo_shared.h"
#include "RiPRCore.h"

#include <sabi_core/sabi_core.h>

using sabi::RenderableNode;
using sabi::SpaceTime;
using sabi::CgModel;
using sabi::CgModelPtr;

// Forward declarations
class RiPRModelHandler;
class RiPRMaterialHandler;

// Shared pointer type for RiPRModel
using RiPRModelPtr = std::shared_ptr<class RiPRModel>;

// Geometry type enumeration
enum class RiPRGeometryType
{
    Triangle,
    Curve,
    TFDM,
    NRTDSM,
    Flyweight,
    Phantom
};

// Base class for all RiPR models
// Creates RiPRSurfaces for each surface and groups them
// Each RiPRSurface will have one DisneyMaterial directly
class RiPRModel
{
public:
    RiPRModel() = default;
    virtual ~RiPRModel() = default;

    // Core interface - must be implemented by derived classes
    virtual RiPRGeometryType getGeometryType() const = 0;
    
    // Create all geometry from the RenderableNode
    virtual void createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder, RenderContext* renderContext = nullptr) = 0;
    
    // Get the RiPRSurfaceGroup for this model
    ripr::RiPRSurfaceGroup* getSurfaceGroup() { return surfaceGroup_.get(); }
    const ripr::RiPRSurfaceGroup* getSurfaceGroup() const { return surfaceGroup_.get(); }
    
    // Get all RiPRSurfaces
    const std::vector<std::unique_ptr<ripr::RiPRSurface>>& getSurfaces() const { 
        return surfaces_; 
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
    // Helper to calculate combined AABB from all surfaces
    void calculateCombinedAABB();
    
protected:
    // RiPR surfaces (one per surface/material)
    std::vector<std::unique_ptr<ripr::RiPRSurface>> surfaces_;
    
    // Surface group containing all surfaces
    std::unique_ptr<ripr::RiPRSurfaceGroup> surfaceGroup_;
    
    // Combined bounding box
    AABB combinedAABB_;
    
    // Reference to the source node (not owned)
    sabi::Renderable* sourceNode_ = nullptr;
};

// Model that can contain mixed geometry types
// Named TriangleModel for compatibility but can handle all geometry types
class RiPRTriangleModel : public RiPRModel
{
public:
    static RiPRModelPtr create()
    {
        return std::make_shared<RiPRTriangleModel>();
    }
    
    RiPRTriangleModel() = default;
    ~RiPRTriangleModel() override = default;
    
    // RiPRModel interface implementation
    // Returns the primary geometry type (but can contain mixed types)
    RiPRGeometryType getGeometryType() const override { 
        return RiPRGeometryType::Triangle; 
    }
    
    void createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder, RenderContext* renderContext = nullptr) override;
    
private:
    // Helper to create appropriate geometry based on surface properties
    void createGeometryForSurface(
        const CgModelPtr& model,
        size_t surfaceIndex,
        ripr::RiPRSurface* surface,
        RenderContext* renderContext = nullptr);
    
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
        ripr::RiPRSurface* surface);
    
    // Helper to calculate AABB for vertices
    AABB calculateAABBForVertices(const std::vector<shared::Vertex>& vertices);
    
    // Helper to determine geometry type for a surface
    bool shouldUseCurveGeometry(const CgModelPtr& model, size_t surfaceIndex) const;
    bool shouldUseDisplacementGeometry(const CgModelPtr& model, size_t surfaceIndex) const;
};

// Flyweight model for instancing (references another model's geometry)
class RiPRFlyweightModel : public RiPRModel
{
public:
    static RiPRModelPtr create()
    {
        return std::make_shared<RiPRFlyweightModel>();
    }
    
    RiPRGeometryType getGeometryType() const override { 
        return RiPRGeometryType::Flyweight; 
    }
    
    void createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder, RenderContext* renderContext = nullptr) override;
    
    // Set the source model this flyweight references
    void setSourceModel(RiPRModel* sourceModel) { sourceModel_ = sourceModel; }
    RiPRModel* getSourceModel() const { return sourceModel_; }
    
private:
    // Reference to the source model (not owned)
    RiPRModel* sourceModel_ = nullptr;
};

// Phantom model for collision-free instances
class RiPRPhantomModel : public RiPRModel
{
public:
    static RiPRModelPtr create()
    {
        return std::make_shared<RiPRPhantomModel>();
    }
    
    RiPRGeometryType getGeometryType() const override { 
        return RiPRGeometryType::Phantom; 
    }
    
    void createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder, RenderContext* renderContext = nullptr) override;
};