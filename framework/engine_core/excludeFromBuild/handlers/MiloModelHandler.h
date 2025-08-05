#pragma once

// MiloModelHandler.h
// Responsible for managing Milo 3D models in the rendering system, providing
// model creation, retrieval, and visibility control functionality

#include "../model/MiloModel.h"

using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::WeakRenderableList;

using MiloModelHandlerPtr = std::shared_ptr<class MiloModelHandler>;
class MiloMaterialHandler;
class MiloSceneHandler;

class MiloModelHandler
{
    // Internal class that manages the storage and lookup of Milo models
    class MiloModelManager
    {
    public:
        // Stores a Milo model with the specified key identifier
        // Returns the key for future reference
        ItemID storeModel(MiloModelPtr model, ItemID key)
        {
            models[key] = model;
            return key;
        }

        MiloModelPtr retrieveModel(ItemID key)
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
        const std::map<ItemID, MiloModelPtr>& getModels() const { return models; }
        
        // Clear all models
        void clear()
        {
            models.clear();
        }

    private:
        std::map<ItemID, MiloModelPtr> models;  // Storage map for models indexed by ID
    };

public:
    // Factory method to create a new MiloModelHandler instance
    static MiloModelHandlerPtr create(RenderContextPtr ctx) 
    { 
        return std::make_shared<MiloModelHandler>(ctx); 
    }

public:
    // Constructor that initializes the handler with a render context
    MiloModelHandler(RenderContextPtr ctx);
    
    // Destructor
    ~MiloModelHandler();

    // Initialize the model handler with buffer allocation for geometry instances
    void initialize();

    // Finalize and clean up all resources
    void finalize();

    // Set the material and scene handlers (must be called before adding models)
    void setHandlers(std::shared_ptr<MiloMaterialHandler> materialHandler,
                     std::shared_ptr<MiloSceneHandler> sceneHandler)
    {
        materialHandler_ = materialHandler;
        sceneHandler_ = sceneHandler;
    }

    // Adds a single model from a renderable node to the Milo scene
    void addCgModel(RenderableWeakRef weakNode);
    
    // Adds multiple models from a list of renderable nodes
    void addCgModelList(const WeakRenderableList& weakNodeList);
    
    // Sets visibility mask for all managed models
    void setAllModelsVisibility(uint32_t mask)
    {
        modelMgr.setAllVisibility(mask);
    }

    // Get all models for updating materials
    const std::map<ItemID, MiloModelPtr>& getModels() const 
    { 
        return modelMgr.getModels(); 
    }
    
    // Retrieves a Milo model by its ID
    MiloModelPtr getMiloModel(ItemID key)
    {
        return modelMgr.retrieveModel(key);
    }
    
    // Retrieves a Milo model by renderable node
    MiloModelPtr getModel(const RenderableNode& node)
    {
        if (!node) return nullptr;
        return getMiloModel(node->getID());
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

private:
    RenderContextPtr ctx_ = nullptr;  // Render context for OptiX operations
    MiloModelManager modelMgr;        // Model manager that stores and retrieves models
    
    // Handler references (not owned)
    std::shared_ptr<MiloMaterialHandler> materialHandler_;
    std::shared_ptr<MiloSceneHandler> sceneHandler_;

    // Slot management for geometry instances
    SlotFinder geomInstSlotFinder_;

    // Geometry instance data buffer
    cudau::TypedBuffer<shared::GeometryInstanceData> geometryInstanceDataBuffer_;
    
    // Track initialization state
    bool isInitialized_ = false;
    
    // Maximum number of geometry instances
    static constexpr uint32_t maxNumGeometryInstances = 65536;
};