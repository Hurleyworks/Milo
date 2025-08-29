#pragma once

// RiPRModelHandler.h
// Responsible for managing RiPR 3D models in the rendering system, providing
// model creation, retrieval, and visibility control functionality

#include "../models/RiPRModel.h"

using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::WeakRenderableList;

using RiPRModelHandlerPtr = std::shared_ptr<class RiPRModelHandler>;
class RiPRSceneHandler;
class RiPREngine;

class RiPRModelHandler
{
    // Internal class that manages the storage and lookup of RiPR models
    class RiPRModelManager
    {
    public:
        // Stores a RiPR model with the specified key identifier
        // Returns the key for future reference
        ItemID storeModel(RiPRModelPtr model, ItemID key)
        {
            models[key] = model;
            return key;
        }

        RiPRModelPtr retrieveModel(ItemID key)
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
        const std::map<ItemID, RiPRModelPtr>& getModels() const { return models; }
        
        // Clear all models
        void clear()
        {
            models.clear();
        }

    private:
        std::map<ItemID, RiPRModelPtr> models;  // Storage map for models indexed by ID
    };

public:
    // Factory method to create a new RiPRModelHandler instance
    static RiPRModelHandlerPtr create(RenderContextPtr ctx) 
    { 
        return std::make_shared<RiPRModelHandler>(ctx); 
    }

public:
    // Constructor that initializes the handler with a render context
    RiPRModelHandler(RenderContextPtr ctx);
    
    // Destructor
    ~RiPRModelHandler();

    // Initialize the model handler with buffer allocation for geometry instances
    void initialize();

    // Finalize and clean up all resources
    void finalize();

    // Set the scene handler (must be called before adding models)
    void setSceneHandler(std::shared_ptr<RiPRSceneHandler> sceneHandler)
    {
        sceneHandler_ = sceneHandler;
    }
    
    // Set the engine pointer for accessing compute kernels
    void setEngine(RiPREngine* engine) { engine_ = engine; }
    
    // Set the area light handler
    void setAreaLightHandler(std::shared_ptr<class RiPRAreaLightHandler> areaLightHandler);
    std::shared_ptr<class RiPRAreaLightHandler> getAreaLightHandler() { return areaLightHandler_; }

    // Adds a single model from a renderable node to the RiPR scene
    void addCgModel(RenderableWeakRef weakNode);
    
    // Adds multiple models from a list of renderable nodes
    void addCgModelList(const WeakRenderableList& weakNodeList);
    
    // Sets visibility mask for all managed models
    void setAllModelsVisibility(uint32_t mask)
    {
        modelMgr.setAllVisibility(mask);
    }
    
    // Compute light probabilities for a triangle model with emissive material
    void computeLightProbabilities(RiPRTriangleModel* model, uint32_t geomInstSlot, uint32_t materialSlot);

    // Get all models for updating materials
    const std::map<ItemID, RiPRModelPtr>& getModels() const 
    { 
        return modelMgr.getModels(); 
    }
    
    // Retrieves a RiPR model by its ID
    RiPRModelPtr getRiPRModel(ItemID key)
    {
        return modelMgr.retrieveModel(key);
    }
    
    // Retrieves a RiPR model by renderable node
    RiPRModelPtr getModel(const RenderableNode& node)
    {
        if (!node) return nullptr;
        return getRiPRModel(node->getID());
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
    void computeLightProbabilities(RiPRTriangleModel* model, uint32_t geomInstSlot);

private:
    RenderContextPtr ctx_ = nullptr;  // Render context for OptiX operations
    RiPRModelManager modelMgr;        // Model manager that stores and retrieves models
    
    // Handler references (not owned)
    std::shared_ptr<RiPRSceneHandler> sceneHandler_;
    std::shared_ptr<class RiPRAreaLightHandler> areaLightHandler_;
    RiPREngine* engine_ = nullptr;  // Engine for accessing compute kernels

    // Slot management for geometry instances
    SlotFinder geomInstSlotFinder_;

    // Geometry instance data buffer
    cudau::TypedBuffer<shared::GeometryInstanceData> geometryInstanceDataBuffer_;
    
    // Track initialization state
    bool isInitialized_ = false;
    
    // Maximum number of geometry instances
    static constexpr uint32_t maxNumGeometryInstances = 65536;
};