#pragma once

// ShockerSceneHandler.h
// Manages the scene graph for the Shocker rendering system
// Coordinates between ShockerModelHandler and ShockerMaterialHandler

#include "../../../RenderContext.h"
#include "../../../common/common_host.h"
#include "../models/ShockerCore.h"

using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::WeakRenderableList;

// Forward declarations
class ShockerModelHandler;
class ShockerMaterialHandler;
class AreaLightHandler;
using ShockerSceneHandlerPtr = std::shared_ptr<class ShockerSceneHandler>;
using ShockerModelHandlerPtr = std::shared_ptr<ShockerModelHandler>;
using ShockerMaterialHandlerPtr = std::shared_ptr<ShockerMaterialHandler>;
using AreaLightHandlerPtr = std::shared_ptr<AreaLightHandler>;

class ShockerSceneHandler
{
public:
    // Factory function for creating ShockerSceneHandler objects
    static ShockerSceneHandlerPtr create(RenderContextPtr ctx) 
    { 
        return std::make_shared<ShockerSceneHandler>(ctx); 
    }

    // Map type for tracking scene nodes by instance index
    using NodeMap = std::unordered_map<uint32_t, RenderableWeakRef>;

public:
    // Constructor initializes the scene handler with a render context
    ShockerSceneHandler(RenderContextPtr ctx);
    
    // Destructor handles cleanup of scene resources
    ~ShockerSceneHandler();

    // Initialize the scene handler
    void initialize();

    // Set the model handler (must be called before using model operations)
    void setModelHandler(ShockerModelHandlerPtr modelHandler) { modelHandler_ = modelHandler; }
    
    // Set the material handler (must be called before using material operations)  
    void setMaterialHandler(ShockerMaterialHandlerPtr materialHandler) { materialHandler_ = materialHandler; }
    
    // Set the area light handler
    void setAreaLightHandler(AreaLightHandlerPtr areaLightHandler) { areaLightHandler_ = areaLightHandler; }

    // Get handlers
    ShockerModelHandlerPtr getModelHandler() const { return modelHandler_; }
    ShockerMaterialHandlerPtr getMaterialHandler() const { return materialHandler_; }
    AreaLightHandlerPtr getAreaLightHandler() const { return areaLightHandler_; }

    // Creates a node from a renderable node (replaces createInstance)
    shocker::ShockerNode* createShockerNode(RenderableWeakRef& weakNode);

    // Creates multiple nodes from a list of renderable nodes
    void createNodeList(const WeakRenderableList& weakNodeList);

    // Process a renderable node (creates model and node)
    void processRenderableNode(RenderableNode& node);

    // Clear all nodes and models
    void clear();

    // Get all nodes
    const std::vector<shocker::ShockerNode*>& getShockerNodes() const { return nodes_; }

    // Get node by index
    shocker::ShockerNode* getShockerNode(uint32_t index) const;

    // Get renderable node by node index
    RenderableWeakRef getRenderableNode(uint32_t nodeIndex) const;

    // Build acceleration structures
    void buildAccelerationStructures();

    // Update acceleration structures
    void updateAccelerationStructures();
    
    // Find node that contains a given surface
    shocker::ShockerNode* findNodeForSurface(shocker::ShockerSurface* surface) const;

    // Get statistics
    size_t getNodeCount() const { return nodes_.size(); }
    size_t getSurfaceCount() const;
    size_t getMaterialCount() const;
    
    // Get the traversable handle for the scene
    OptixTraversableHandle getTraversableHandle() const { return travHandle_; }
    
    // Set the scene (must be called before building acceleration structures)
    void setScene(optixu::Scene* scene) { scene_ = scene; }

private:
    // Render context
    RenderContextPtr ctx_;
    
    // Handlers
    ShockerModelHandlerPtr modelHandler_;
    ShockerMaterialHandlerPtr materialHandler_;
    AreaLightHandlerPtr areaLightHandler_;
    
    // Scene data
    std::vector<shocker::ShockerNode*> nodes_;
    NodeMap nodeMap_;
    
    // Instance slot management
    SlotFinder instanceSlotFinder_;
    
    // Initialization flag
    bool isInitialized_ = false;
    
    // Maximum number of instances
    static constexpr size_t MaxNumInstances = 100000;
    
    // OptiX scene and IAS
    optixu::Scene* scene_ = nullptr;  // Not owned, set by engine
    optixu::InstanceAccelerationStructure ias_;
    cudau::Buffer iasMem_;
    cudau::TypedBuffer<OptixInstance> instanceBuffer_;
    OptixTraversableHandle travHandle_ = 0;  // 0 is valid for empty scene
};