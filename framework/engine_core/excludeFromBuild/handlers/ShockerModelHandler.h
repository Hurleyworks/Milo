#pragma once

// ShockerModelHandler.h
// Manages ShockerModel lifecycle, geometry instances, and groups
// Part of the ShockerEngine handler architecture

#include "../model/ShockerModel.h"
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
    
    // Geometry instance management
    GeometryInstance* createGeometryInstance(ShockerModel* model);
    
    // Geometry group management
    GeometryGroup* createGeometryGroup(const std::vector<GeometryInstance*>& instances);
    
    // Instance creation with transforms
    Instance* createInstance(ShockerModel* model, const sabi::SpaceTime& spacetime);
    
    // Get model by name
    ShockerModelPtr getModel(const std::string& name) const;
    
    // Check if model exists
    bool hasModel(const std::string& name) const;
    
    // Get all models
    const std::unordered_map<std::string, ShockerModelPtr>& getAllModels() const { return models_; }
    
    // Get all geometry instances
    const std::vector<std::unique_ptr<GeometryInstance>>& getGeometryInstances() const { return geometryInstances_; }
    
    // Get all geometry groups  
    const std::vector<std::unique_ptr<GeometryGroup>>& getGeometryGroups() const { return geometryGroups_; }
    
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
    AABB calculateCombinedAABB(const std::vector<GeometryInstance*>& instances) const;
    
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
    
    // Geometry instances - owns all created geometry instances
    std::vector<std::unique_ptr<GeometryInstance>> geometryInstances_;
    
    // Geometry groups - owns all created geometry groups
    std::vector<std::unique_ptr<GeometryGroup>> geometryGroups_;
    
    // Instances - owns all created instances
    std::vector<std::unique_ptr<Instance>> instances_;
    
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