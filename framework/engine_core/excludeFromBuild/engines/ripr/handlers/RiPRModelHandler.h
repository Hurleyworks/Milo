#pragma once

// RiPRModelHandler.h
// Manages RiPRModel lifecycle, geometry instances, and groups
// Part of the RiPREngine handler architecture

#include "../models/RiPRModel.h"
#include "../models/RiPRCore.h"
#include "../models/RiPRDeviceCore.h"
#include "../../../common/common_host.h"
#include "../../milo/milo_shared.h"
#include <unordered_map>
#include <memory>
#include <vector>

// Forward declarations
class RiPRMaterialHandler;
class RiPRTextureHandler;
class AreaLightHandler;
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
using AreaLightHandlerPtr = std::shared_ptr<AreaLightHandler>;

class RiPRModelHandler
{
public:
    RiPRModelHandler() = default;
    ~RiPRModelHandler() = default;
    
    // Initialize the handler with dependencies
    void initialize(RenderContextPtr context);
    
    // Main conversion pipeline - creates model from RenderableNode
    RiPRModelPtr processRenderableNode(const sabi::RenderableNode& node);
    
    // Create a model based on geometry type
    RiPRModelPtr createModelByType(const sabi::CgModelPtr& cgModel);
    
    // Surface management (replaces geometry instance)
    ripr::RiPRSurface* createRiPRSurface(RiPRModel* model);
    
    // Surface group management (replaces geometry group)
    ripr::RiPRSurfaceGroup* createRiPRSurfaceGroup(const std::vector<ripr::RiPRSurface*>& surfaces);
    
    // Node creation with transforms (replaces instance)
    ripr::RiPRNode* createRiPRNode(RiPRModel* model, const sabi::SpaceTime& spacetime);
    
    // Get model by name
    RiPRModelPtr getModel(const std::string& name) const;
    
    // Check if model exists
    bool hasModel(const std::string& name) const;
    
    // Get all models
    const std::unordered_map<std::string, RiPRModelPtr>& getAllModels() const { return models_; }
    
    // Get count of all surfaces across all models
    size_t getRiPRSurfaceCount() const;
    
    // Get all surface groups  
    const std::vector<std::unique_ptr<ripr::RiPRSurfaceGroup>>& getRiPRSurfaceGroups() const { return surfaceGroups_; }
    
    // Clear all data
    void clear();
    
    // Set material handler (for future integration)
    void setMaterialHandler(RiPRMaterialHandler* handler) { materialHandler_ = handler; }
    
    // Set texture handler (for future integration)
    void setTextureHandler(RiPRTextureHandler* handler) { textureHandler_ = handler; }
    
    // Set area light handler
    void setAreaLightHandler(AreaLightHandlerPtr handler) { areaLightHandler_ = handler; }
    
    // Get the geometry instance data buffer for GPU upload
    cudau::TypedBuffer<ripr::RiPRSurfaceData>* getGeometryInstanceDataBuffer()
    {
        return &geometryInstanceDataBuffer_;
    }
    
    // Update the geometry instance data buffer with all surface data
    void updateGeometryInstanceDataBuffer();
    
private:
    // Helper to determine geometry type from CgModel
    RiPRGeometryType determineGeometryType(const sabi::CgModelPtr& model) const;
    
    // Helper to calculate combined AABB
    AABB calculateCombinedAABB(const std::vector<ripr::RiPRSurface*>& surfaces) const;
    
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
    std::unordered_map<std::string, RiPRModelPtr> models_;
    
    // Surfaces - owns all created surfaces (replaces geometry instances)
    std::vector<std::unique_ptr<ripr::RiPRSurface>> surfaces_;
    
    // Surface groups - owns all created surface groups (replaces geometry groups)
    std::vector<std::unique_ptr<ripr::RiPRSurfaceGroup>> surfaceGroups_;
    
    // Nodes - owns all created nodes (replaces instances)
    std::vector<std::unique_ptr<ripr::RiPRNode>> nodes_;
    
    // Slot management
    SlotFinder geomInstSlotFinder_;
    SlotFinder instanceSlotFinder_;
    
    // GPU buffers
    cudau::TypedBuffer<ripr::RiPRSurfaceData> geometryInstanceDataBuffer_;
    
    // Dependencies
    RenderContextPtr renderContext_;
    RiPRMaterialHandler* materialHandler_ = nullptr;
    RiPRTextureHandler* textureHandler_ = nullptr;
    AreaLightHandlerPtr areaLightHandler_ = nullptr;
    
    // Statistics
    size_t totalTriangles_ = 0;
    size_t totalVertices_ = 0;
};