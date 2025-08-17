#pragma once

// ShockerModelHandler.h
// Responsible for managing Shocker 3D models in the rendering system, providing
// model creation, retrieval, and visibility control functionality

#include "../models/ShockerModel.h"

using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::WeakRenderableList;

using ShockerModelHandlerPtr = std::shared_ptr<class ShockerModelHandler>;
class ShockerMaterialHandler;
class ShockerSceneHandler;
class ShockerEngine;

class ShockerModelHandler
{
    // Internal class that manages the storage and lookup of Shocker models
    class ShockerModelManager
    {
    public:
        // Stores a Shocker model with the specified key identifier
        // Returns the key for future reference
        ItemID storeModel(ShockerModelPtr model, ItemID key)
        {
            models[key] = model;
            return key;
        }

        ShockerModelPtr retrieveModel(ItemID key)
        {
            auto it = models.find(key);
            if (it != models.end())
            {
                return it->second;
            }
            LOG(DBUG) << "Could not find model with key " << key;
            return nullptr;
        }

        void removeModel(ItemID itemID)
        {
            auto it = models.find(itemID);
            if (it != models.end())
            {
                models.erase(it);
            }
        }

        // Sets visibility mask for all managed models
        // Used to control which models are visible in different ray types
        void setAllVisibility(uint32_t mask)
        {
            for (auto& it : models)
            {
                it.second->getOptiXInstance().setVisibilityMask(mask);
            }
        }
        
        // Get all models for updating materials
        const std::map<ItemID, ShockerModelPtr>& getModels() const { return models; }
        
        // Clear all models
        void clear()
        {
            models.clear();
        }

    private:
        std::map<ItemID, ShockerModelPtr> models;  // Storage map for models indexed by ID
    };

public:
    // Factory method to create a new ShockerModelHandler instance
    static ShockerModelHandlerPtr create(RenderContextPtr ctx) 
    { 
        return std::make_shared<ShockerModelHandler>(ctx); 
    }

public:
    // Constructor that initializes the handler with a render context
    ShockerModelHandler(RenderContextPtr ctx);
    
    // Destructor
    ~ShockerModelHandler();

    // Initialize the model handler with buffer allocation for geometry instances
    void initialize();

    // Finalize and clean up all resources
    void finalize();

    // Set the material and scene handlers (must be called before adding models)
    void setHandlers(std::shared_ptr<ShockerMaterialHandler> materialHandler,
                     std::shared_ptr<ShockerSceneHandler> sceneHandler)
    {
        materialHandler_ = materialHandler;
        sceneHandler_ = sceneHandler;
    }
    
    // Set the engine pointer for accessing compute kernels
    void setEngine(ShockerEngine* engine) { engine_ = engine; }

    // Adds a single model from a renderable node to the Shocker scene
    void addCgModel(RenderableWeakRef weakNode);
    
    // Adds multiple models from a list of renderable nodes
    void addCgModelList(const WeakRenderableList& weakNodeList);
    
    // Sets visibility mask for all managed models
    void setAllModelsVisibility(uint32_t mask)
    {
        modelMgr.setAllVisibility(mask);
    }

    // Get all models for updating materials
    const std::map<ItemID, ShockerModelPtr>& getModels() const 
    { 
        return modelMgr.getModels(); 
    }
    
    // Retrieves a Shocker model by its ID
    ShockerModelPtr getShockerModel(ItemID key)
    {
        return modelMgr.retrieveModel(key);
    }
    
    // Retrieves a Shocker model by renderable node
    ShockerModelPtr getModel(const RenderableNode& node)
    {
        if (!node) return nullptr;
        return getShockerModel(node->getID());
    }

    void removeModel(ItemID itemID);

    // Get the geometry instance data buffer
    cudau::TypedBuffer<shared::GeometryInstanceData>* getGeometryInstanceDataBuffer()
    {
        return &geometryInstanceDataBuffer_;
    }

    // Allocate a geometry instance slot
    uint32_t allocateGeometryInstanceSlot()
    {
        uint32_t slot = geomInstSlotFinder_.getFirstAvailableSlot();
        if (slot != SlotFinder::InvalidSlotIndex)
        {
            geomInstSlotFinder_.setInUse(slot);
        }
        return slot;
    }

    // Release a geometry instance slot
    void releaseGeometryInstanceSlot(uint32_t slot)
    {
        if (slot != SlotFinder::InvalidSlotIndex)
        {
            geomInstSlotFinder_.setNotInUse(slot);
        }
    }
    
    // Compute light probabilities for emissive geometry
    void computeLightProbabilities(ShockerTriangleModel* model, uint32_t geomInstSlot);

private:
    RenderContextPtr ctx_ = nullptr;  // Render context for OptiX operations
    ShockerModelManager modelMgr;        // Model manager that stores and retrieves models
    
    // Handler references (not owned)
    std::shared_ptr<ShockerMaterialHandler> materialHandler_;
    std::shared_ptr<ShockerSceneHandler> sceneHandler_;
    ShockerEngine* engine_ = nullptr;  // Engine for accessing compute kernels

    // Slot management for geometry instances
    SlotFinder geomInstSlotFinder_;

    // Geometry instance data buffer
    cudau::TypedBuffer<shared::GeometryInstanceData> geometryInstanceDataBuffer_;
    
    // Track initialization state
    bool isInitialized_ = false;
    
    // Maximum number of geometry instances
    static constexpr uint32_t maxNumGeometryInstances = 65536;
};