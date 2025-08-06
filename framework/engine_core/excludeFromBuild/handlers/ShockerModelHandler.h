#pragma once

// ShockerModelHandler.h
// Manages ShockerModel lifecycle, geometry instances, and groups
// Part of the ShockerEngine handler architecture

#include "../model/ShockerModel.h"
#include "../model/ShockerCore.h"
#include "../common/common_host.h"
#include "../milo_shared.h"
#include <unordered_map>
#include <memory>
#include <vector>

// Forward declarations
class ShockerMaterialHandler;
class ShockerTextureHandler;
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

class ShockerModelHandler
{
public:
    ShockerModelHandler() = default;
    ~ShockerModelHandler() = default;
    
    // Initialize the handler with dependencies
    void initialize(RenderContextPtr context);
    
    // Main conversion pipeline - creates model from RenderableNode
    ShockerModelPtr processRenderableNode(const sabi::RenderableNode& node);
    
    // Create a model based on geometry type
    ShockerModelPtr createModelByType(const sabi::CgModelPtr& cgModel);
    
    // Surface management (replaces geometry instance)
    shocker::ShockerSurface* createShockerSurface(ShockerModel* model);
    
    // Surface group management (replaces geometry group)
    shocker::ShockerSurfaceGroup* createShockerSurfaceGroup(const std::vector<shocker::ShockerSurface*>& surfaces);
    
    // Node creation with transforms (replaces instance)
    shocker::ShockerNode* createShockerNode(ShockerModel* model, const sabi::SpaceTime& spacetime);
    
    // Get model by name
    ShockerModelPtr getModel(const std::string& name) const;
    
    // Check if model exists
    bool hasModel(const std::string& name) const;
    
    // Get all models
    const std::unordered_map<std::string, ShockerModelPtr>& getAllModels() const { return models_; }
    
    // Get count of all surfaces across all models
    size_t getShockerSurfaceCount() const;
    
    // Get all surface groups  
    const std::vector<std::unique_ptr<shocker::ShockerSurfaceGroup>>& getShockerSurfaceGroups() const { return surfaceGroups_; }
    
    // Clear all data
    void clear();
    
    // Set material handler (for future integration)
    void setMaterialHandler(ShockerMaterialHandler* handler) { materialHandler_ = handler; }
    
    // Set texture handler (for future integration)
    void setTextureHandler(ShockerTextureHandler* handler) { textureHandler_ = handler; }
    
private:
    // Helper to determine geometry type from CgModel
    ShockerGeometryType determineGeometryType(const sabi::CgModelPtr& model) const;
    
    // Helper to calculate combined AABB
    AABB calculateCombinedAABB(const std::vector<shocker::ShockerSurface*>& surfaces) const;
    
    // Slot management helpers
    uint32_t allocateGeometryInstanceSlot()
    {
        uint32_t slot = geomInstSlotFinder_.getFirstAvailableSlot();
        if (slot != SlotFinder::InvalidSlotIndex) {
            geomInstSlotFinder_.setInUse(slot);
        }
        return slot;
    }
    
    void releaseGeometryInstanceSlot(uint32_t slot)
    {
        if (slot != SlotFinder::InvalidSlotIndex) {
            geomInstSlotFinder_.setNotInUse(slot);
        }
    }
    
    uint32_t allocateInstanceSlot()
    {
        uint32_t slot = instanceSlotFinder_.getFirstAvailableSlot();
        if (slot != SlotFinder::InvalidSlotIndex) {
            instanceSlotFinder_.setInUse(slot);
        }
        return slot;
    }
    
    void releaseInstanceSlot(uint32_t slot)
    {
        if (slot != SlotFinder::InvalidSlotIndex) {
            instanceSlotFinder_.setNotInUse(slot);
        }
    }
    
private:
    // Model registry - stores all created models by name
    std::unordered_map<std::string, ShockerModelPtr> models_;
    
    // Surfaces - owns all created surfaces (replaces geometry instances)
    std::vector<std::unique_ptr<shocker::ShockerSurface>> surfaces_;
    
    // Surface groups - owns all created surface groups (replaces geometry groups)
    std::vector<std::unique_ptr<shocker::ShockerSurfaceGroup>> surfaceGroups_;
    
    // Nodes - owns all created nodes (replaces instances)
    std::vector<std::unique_ptr<shocker::ShockerNode>> nodes_;
    
    // Slot management
    SlotFinder geomInstSlotFinder_;
    SlotFinder instanceSlotFinder_;
    
    // Dependencies
    RenderContextPtr renderContext_;
    ShockerMaterialHandler* materialHandler_ = nullptr;
    ShockerTextureHandler* textureHandler_ = nullptr;
    
    // Statistics
    size_t totalTriangles_ = 0;
    size_t totalVertices_ = 0;
};