#pragma once

// ShockerSceneHandler.h
// Manages the scene graph for the Shocker rendering system
// Coordinates between ShockerModelHandler and ShockerMaterialHandler

#include "../RenderContext.h"
#include "../common/common_host.h"

using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::WeakRenderableList;

// Forward declarations
class ShockerModelHandler;
class ShockerMaterialHandler;
using ShockerSceneHandlerPtr = std::shared_ptr<class ShockerSceneHandler>;
using ShockerModelHandlerPtr = std::shared_ptr<ShockerModelHandler>;
using ShockerMaterialHandlerPtr = std::shared_ptr<ShockerMaterialHandler>;

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

    // Get handlers
    ShockerModelHandlerPtr getModelHandler() const { return modelHandler_; }
    ShockerMaterialHandlerPtr getMaterialHandler() const { return materialHandler_; }

    // Creates an instance from a renderable node
    Instance* createInstance(RenderableWeakRef& weakNode);

    // Creates multiple instances from a list of renderable nodes
    void createInstanceList(const WeakRenderableList& weakNodeList);

    // Process a renderable node (creates model and instance)
    void processRenderableNode(RenderableNode& node);

    // Clear all instances and models
    void clear();

    // Get all instances
    const std::vector<Instance*>& getInstances() const { return instances_; }

    // Get instance by index
    Instance* getInstance(uint32_t index) const;

    // Get node by instance index
    RenderableWeakRef getNode(uint32_t instanceIndex) const;

    // Build acceleration structures
    void buildAccelerationStructures();

    // Update acceleration structures
    void updateAccelerationStructures();

    // Get statistics
    size_t getInstanceCount() const { return instances_.size(); }
    size_t getGeometryInstanceCount() const;
    size_t getMaterialCount() const;

private:
    // Render context
    RenderContextPtr ctx_;
    
    // Handlers
    ShockerModelHandlerPtr modelHandler_;
    ShockerMaterialHandlerPtr materialHandler_;
    
    // Scene data
    std::vector<Instance*> instances_;
    NodeMap nodeMap_;
    
    // Instance slot management
    SlotFinder instanceSlotFinder_;
    
    // Initialization flag
    bool isInitialized_ = false;
    
    // Maximum number of instances
    static constexpr size_t MaxNumInstances = 100000;
};