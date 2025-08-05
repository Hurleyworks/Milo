#include "MiloModelHandler.h"
#include "MiloMaterialHandler.h"
#include "MiloSceneHandler.h"
#include "../model/MiloModel.h"

// Constructor initializes the model handler with the render context
MiloModelHandler::MiloModelHandler(RenderContextPtr ctx) :
    ctx_(ctx)
{
    LOG(DBUG) << _FN_;
}

// Destructor handles cleanup of model resources
MiloModelHandler::~MiloModelHandler()
{
    finalize();
}

// Initialize the model handler with buffer allocation for geometry instances
void MiloModelHandler::initialize()
{
    if (isInitialized_)
    {
        LOG(WARNING) << "MiloModelHandler already initialized";
        return;
    }
    
    LOG(INFO) << "Initializing MiloModelHandler";
    
    if (!ctx_)
    {
        LOG(WARNING) << "RenderContext is null, cannot initialize";
        return;
    }
    
    CUcontext cuContext = ctx_->getCudaContext();
    if (!cuContext)
    {
        LOG(WARNING) << "CUDA context not available";
        return;
    }
    
    // Initialize slot finder for geometry instances
    geomInstSlotFinder_.initialize(maxNumGeometryInstances);
    LOG(INFO) << "Initialized geometry instance slot finder with capacity: " << maxNumGeometryInstances;
    
    // Initialize geometry instance data buffer
    // Using Device buffer type for better performance (same as Shocker)
    geometryInstanceDataBuffer_.initialize(cuContext, cudau::BufferType::Device, maxNumGeometryInstances);
    LOG(INFO) << "Initialized geometry instance data buffer";
    
    isInitialized_ = true;
    LOG(INFO) << "MiloModelHandler initialized successfully";
}

// Finalize and clean up all resources
void MiloModelHandler::finalize()
{
    if (!isInitialized_)
        return;
        
    LOG(INFO) << "Finalizing MiloModelHandler";
    
    try
    {
        // First sync CUDA operations if context is available
        if (ctx_ && ctx_->getCudaStream())
        {
            CUDADRV_CHECK(cuStreamSynchronize(ctx_->getCudaStream()));
        }

        // Clear all models - this will trigger cleanup of their resources
        modelMgr.clear();
        
        // Finalize the geometry instance buffer
        geometryInstanceDataBuffer_.finalize();
        
        // Reset the slot finder
        geomInstSlotFinder_.reset();
        
        LOG(INFO) << "MiloModelHandler finalized successfully";
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Exception during MiloModelHandler finalization: " << e.what();
    }
    
    isInitialized_ = false;
}

// Adds a single renderable node to the Milo scene
// Creates appropriate model type based on node characteristics
void MiloModelHandler::addCgModel(RenderableWeakRef weakNode)
{
    // Skip if the node reference is no longer valid
    if (weakNode.expired()) return;

    // Get a strong reference to the node
    RenderableNode node = weakNode.lock();

    LOG(DBUG) << "Processing " << node->getName()
               << ", isInstance: " << (node->isInstance() ? "true" : "false")
               << ", clientID: " << node->getClientID()
               << ", instancedFrom: " << (node->getInstancedFrom() ? node->getInstancedFrom()->getName() : "none")
               << ", instancedFrom clientID: " << (node->getInstancedFrom() ? node->getInstancedFrom()->getClientID() : 0);

    // Verify handlers are set
    if (!materialHandler_ || !sceneHandler_)
    {
        LOG(WARNING) << "Material or scene handler not set - cannot process model";
        return;
    }

    // Handle instance nodes (references to other models)
    if (node->isInstance())
    {
        // Before creating an instance, verify the source model exists
        RenderableNode instancedFrom = node->getInstancedFrom();
        if (!instancedFrom)
        {
            LOG(WARNING) << "Cannot process instance - no source model specified for " << node->getName();
            return;
        }

        MiloModelPtr sourceModel = getMiloModel(instancedFrom->getClientID());
        if (!sourceModel)
        {
            LOG(WARNING) << "Source model not yet available for instance " << node->getName()
                         << " - source: " << instancedFrom->getName()
                         << " clientID: " << instancedFrom->getClientID();
            // Store this in a pending list to try again later
            return;
        }

        // Create a flyweight model that references another model's geometry
        MiloModelPtr miloModel = MiloFlyweightModel::create();
        if (!miloModel)
        {
            LOG(WARNING) << "Failed to create MiloModelPtr";
            return;
        }

        // Flyweight models share the geometry instance slot with their source model
        miloModel->setGeomInstSlot(sourceModel->getGeomInstSlot());
        LOG(DBUG) << "Flyweight model " << node->getName() 
                  << " sharing geometry instance slot " << sourceModel->getGeomInstSlot()
                  << " with source model " << instancedFrom->getName();

        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(miloModel, node->getClientID());

        // Create a geometry instance in the scene
        sceneHandler_->createGeometryInstance(weakNode);
    }
    // Handle phantom nodes (used for physics/collision)
    else if (node->isPhantom())
    {
        // Create a phantom model for collision detection
        MiloModelPtr miloModel = MiloPhantomModel::create();
        
        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(miloModel, node->getClientID());
        
        // Create a physics phantom in the scene
        sceneHandler_->createPhysicsPhantom(weakNode);
    }
    // Handle regular geometry nodes
    else
    {
        // Create a triangle model for regular geometry
        MiloModelPtr miloModel = MiloTriangleModel::create();

        // Allocate a geometry instance slot for this model
        uint32_t geomInstSlot = allocateGeometryInstanceSlot();
        if (geomInstSlot == SlotFinder::InvalidSlotIndex)
        {
            LOG(WARNING) << "Failed to allocate geometry instance slot for " << node->getName();
            return;
        }
        miloModel->setGeomInstSlot(geomInstSlot);
        LOG(DBUG) << "Allocated geometry instance slot " << geomInstSlot << " for " << node->getName();

        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(miloModel, node->getClientID());

        // Build the geometry
        optixu::Scene* scene = sceneHandler_->getScene();
        if (!scene)
        {
            LOG(WARNING) << "Scene not available from scene handler";
            return;
        }
        miloModel->createGeometry(ctx_, node, scene);

        // Now create materials for each surface
        MiloTriangleModel* triangleModel = dynamic_cast<MiloTriangleModel*>(miloModel.get());
        if (triangleModel)
        {
            sabi::CgModelPtr model = node->getModel();
            fs::path contentFolder = ctx_->getPropertyService().renderProps->getVal<std::string>(RenderKey::ContentFolder);
            
            uint32_t materialCount = model->S.size();
            if (materialCount > 1)
            {
                for (int i = 0; i < materialCount; ++i)
                {
                    optixu::Material mat = materialHandler_->createDisneyMaterial(
                        model->S[i].cgMaterial, contentFolder, model);
                    triangleModel->getGeometryInstance()->setMaterial(0, i, mat);
                }
            }
            else
            {
                optixu::Material mat = materialHandler_->createDisneyMaterial(
                    model->S[0].cgMaterial, contentFolder, model);
                triangleModel->getGeometryInstance()->setMaterial(0, 0, mat);
            }
        }

        // Create a Geometry Acceleration Structure (GAS) for ray tracing
        miloModel->createGAS(ctx_, scene, milo_shared::NumRayTypes);
        
        // Populate geometry instance data in the global buffer
        if (geomInstSlot != SlotFinder::InvalidSlotIndex)
        {
            shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
            miloModel->populateGeometryInstanceData(&geomInstDataOnHost[geomInstSlot]);
            geometryInstanceDataBuffer_.unmap();
            LOG(DBUG) << "Populated geometry instance data for slot " << geomInstSlot;
        }

        // Create a regular instance in the scene
        sceneHandler_->createInstance(weakNode);
    }
}

// Adds multiple renderable nodes to the Milo scene efficiently
// Optimizes the process by building models first, then creating instances
void MiloModelHandler::addCgModelList(const WeakRenderableList& weakNodeList)
{
    // Start a scoped timer to measure processing time
    ScopedStopWatch sw("Processing model list in Milo");

    // Verify handlers are set
    if (!materialHandler_ || !sceneHandler_)
    {
        LOG(WARNING) << "Material or scene handler not set - cannot process models";
        return;
    }

    // Track the number of instance nodes for logging
    uint32_t instanceCount = 0;
    
    // Process each node in the list
    for (const auto& weakNode : weakNodeList)
    {
        // Skip invalid nodes
        if (weakNode.expired()) continue;
        RenderableNode node = weakNode.lock();

        // Handle instance nodes separately (flyweight pattern)
        if (node->isInstance()) 
        {
            // Get the source model to share its geometry instance slot
            RenderableNode instancedFrom = node->getInstancedFrom();
            if (!instancedFrom)
            {
                LOG(WARNING) << "Cannot process instance - no source model specified for " << node->getName();
                continue;
            }
            
            MiloModelPtr sourceModel = getMiloModel(instancedFrom->getClientID());
            if (!sourceModel)
            {
                LOG(WARNING) << "Source model not yet available for instance " << node->getName();
                continue;
            }
            
            // Create flyweight model that references another model's geometry
            MiloModelPtr miloModel = MiloFlyweightModel::create();
            
            // Flyweight models share the geometry instance slot with their source model
            miloModel->setGeomInstSlot(sourceModel->getGeomInstSlot());
            
            // Store the model with the client ID as key
            ItemID key = modelMgr.storeModel(miloModel, node->getClientID());
            
            // Count instances for logging
            ++instanceCount;
        }
        // Handle regular geometry nodes
        else
        {
            // Create a triangle model for regular geometry
            MiloModelPtr miloModel = MiloTriangleModel::create();
            
            // Allocate a geometry instance slot for this model
            uint32_t geomInstSlot = allocateGeometryInstanceSlot();
            if (geomInstSlot == SlotFinder::InvalidSlotIndex)
            {
                LOG(WARNING) << "Failed to allocate geometry instance slot for " << node->getName();
                continue;
            }
            miloModel->setGeomInstSlot(geomInstSlot);

            // Store the model with the client ID as key
            ItemID key = modelMgr.storeModel(miloModel, node->getClientID());
            
            // Build the geometry
            optixu::Scene* scene = sceneHandler_->getScene();
            if (!scene)
            {
                LOG(WARNING) << "Scene not available from scene handler";
                continue;
            }
            miloModel->createGeometry(ctx_, node, scene);

            // Now create materials for each surface
            MiloTriangleModel* triangleModel = dynamic_cast<MiloTriangleModel*>(miloModel.get());
            if (triangleModel)
            {
                sabi::CgModelPtr model = node->getModel();
                fs::path contentFolder = ctx_->getPropertyService().renderProps->getVal<std::string>(RenderKey::ContentFolder);
                
                uint32_t materialCount = model->S.size();
                if (materialCount > 1)
                {
                    for (int i = 0; i < materialCount; ++i)
                    {
                        optixu::Material mat = materialHandler_->createDisneyMaterial(
                            model->S[i].cgMaterial, contentFolder, model);
                        triangleModel->getGeometryInstance()->setMaterial(0, i, mat);
                    }
                }
                else
                {
                    optixu::Material mat = materialHandler_->createDisneyMaterial(
                        model->S[0].cgMaterial, contentFolder, model);
                    triangleModel->getGeometryInstance()->setMaterial(0, 0, mat);
                }
            }

            // Create a Geometry Acceleration Structure (GAS) for ray tracing
            optixu::Scene* sceneForGAS = sceneHandler_->getScene();
            miloModel->createGAS(ctx_, sceneForGAS, milo_shared::NumRayTypes);
            
            // Populate geometry instance data in the global buffer
            if (geomInstSlot != SlotFinder::InvalidSlotIndex)
            {
                shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
                miloModel->populateGeometryInstanceData(&geomInstDataOnHost[geomInstSlot]);
                geometryInstanceDataBuffer_.unmap();
            }
        }
    }

    // Create all instances in the scene at once for better performance
    sceneHandler_->createInstanceList(weakNodeList);

    // Log the results of the batch processing
    LOG(DBUG) << "Processed " << weakNodeList.size() << " models of which " << instanceCount << " are instances";
}

void MiloModelHandler::removeModel(ItemID itemID)
{
    // Get the model before removing it so we can release its slot
    MiloModelPtr model = modelMgr.retrieveModel(itemID);
    if (model)
    {
        uint32_t slot = model->getGeomInstSlot();
        if (slot != SlotFinder::InvalidSlotIndex)
        {
            // Only release slot for non-flyweight models
            // Flyweight models share slots with their source models
            MiloFlyweightModel* flyweight = dynamic_cast<MiloFlyweightModel*>(model.get());
            if (!flyweight)
            {
                releaseGeometryInstanceSlot(slot);
                LOG(DBUG) << "Released geometry instance slot " << slot << " for model ID " << itemID;
            }
        }
    }
    
    modelMgr.removeModel(itemID);
}