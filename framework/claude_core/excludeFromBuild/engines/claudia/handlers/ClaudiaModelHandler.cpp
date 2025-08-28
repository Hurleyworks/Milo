#include "ClaudiaModelHandler.h"
#include "ClaudiaSceneHandler.h"
#include "ClaudiaAreaLightHandler.h"
#include "../ClaudiaEngine.h"
#include "../models/ClaudiaModel.h"
#include "../../../handlers/Handlers.h"

// Constructor initializes the model handler with the render context
ClaudiaModelHandler::ClaudiaModelHandler(RenderContextPtr ctx) :
    ctx_(ctx)
{
    LOG(DBUG) << _FN_;
}

// Destructor handles cleanup of model resources
ClaudiaModelHandler::~ClaudiaModelHandler()
{
    finalize();
}

// Initialize the model handler with buffer allocation for geometry instances
void ClaudiaModelHandler::initialize()
{
    if (isInitialized_)
    {
        LOG(WARNING) << "ClaudiaModelHandler already initialized";
        return;
    }
    
    LOG(INFO) << "Initializing ClaudiaModelHandler";
    
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
    // Using Device buffer type for better performance (same as Claudia)
    geometryInstanceDataBuffer_.initialize(cuContext, cudau::BufferType::Device, maxNumGeometryInstances);
    LOG(INFO) << "Initialized geometry instance data buffer";
    
    isInitialized_ = true;
    LOG(INFO) << "ClaudiaModelHandler initialized successfully";
}

// Finalize and clean up all resources
void ClaudiaModelHandler::finalize()
{
    if (!isInitialized_)
        return;
        
    LOG(INFO) << "Finalizing ClaudiaModelHandler";
    
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
        
        LOG(INFO) << "ClaudiaModelHandler finalized successfully";
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Exception during ClaudiaModelHandler finalization: " << e.what();
    }
    
    isInitialized_ = false;
}

// Adds a single renderable node to the Claudia scene
// Creates appropriate model type based on node characteristics
void ClaudiaModelHandler::addCgModel(RenderableWeakRef weakNode)
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
    if (!ctx_->getHandlers().disneyMaterialHandler || !sceneHandler_)
    {
        LOG(WARNING) << "Material or scene handler not set - cannot process model";
        return;
    }

    cudau::Buffer& scratchMem  = sceneHandler_->getASBuildScratchMem();

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

        ClaudiaModelPtr sourceModel = getClaudiaModel(instancedFrom->getClientID());
        if (!sourceModel)
        {
            LOG(WARNING) << "Source model not yet available for instance " << node->getName()
                         << " - source: " << instancedFrom->getName()
                         << " clientID: " << instancedFrom->getClientID();
            // Store this in a pending list to try again later
            return;
        }

        // Create a flyweight model that references another model's geometry
        ClaudiaModelPtr claudiaModel = ClaudiaFlyweightModel::create();
        if (!claudiaModel)
        {
            LOG(WARNING) << "Failed to create ClaudiaModelPtr";
            return;
        }

        // Flyweight models share the geometry instance slot with their source model
        claudiaModel->setGeomInstSlot(sourceModel->getGeomInstSlot());
        LOG(DBUG) << "Flyweight model " << node->getName() 
                  << " sharing geometry instance slot " << sourceModel->getGeomInstSlot()
                  << " with source model " << instancedFrom->getName();

        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(claudiaModel, node->getClientID());

        // Create a geometry instance in the scene
        sceneHandler_->createGeometryInstance(weakNode);
    }
    // Handle phantom nodes (used for physics/collision)
    else if (node->isPhantom())
    {
        // Create a phantom model for collision detection
        ClaudiaModelPtr claudiaModel = ClaudiaPhantomModel::create();
        
        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(claudiaModel, node->getClientID());
        
        // Create a physics phantom in the scene
        sceneHandler_->createPhysicsPhantom(weakNode);
    }
    // Handle regular geometry nodes
    else
    {
        // Create a triangle model for regular geometry
        ClaudiaModelPtr claudiaModel = ClaudiaTriangleModel::create();

        // Allocate a geometry instance slot for this model
        uint32_t geomInstSlot = allocateGeometryInstanceSlot();
        if (geomInstSlot == SlotFinder::InvalidSlotIndex)
        {
            LOG(WARNING) << "Failed to allocate geometry instance slot for " << node->getName();
            return;
        }
        claudiaModel->setGeomInstSlot(geomInstSlot);
        LOG(DBUG) << "Allocated geometry instance slot " << geomInstSlot << " for " << node->getName();

        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(claudiaModel, node->getClientID());

        // Build the geometry
        optixu::Scene scene = sceneHandler_->getScene();
        if (!scene)
        {
            LOG(WARNING) << "Scene not available from scene handler";
            return;
        }
        claudiaModel->createGeometry(ctx_, node, scene);

        // Now create materials for each surface
        ClaudiaTriangleModel* triangleModel = dynamic_cast<ClaudiaTriangleModel*>(claudiaModel.get());
        if (triangleModel)
        {
            sabi::CgModelPtr model = node->getModel();
            fs::path contentFolder = ctx_->getPropertyService().renderProps->getVal<std::string>(RenderKey::ContentFolder);
            
            uint32_t materialCount = model->S.size();
            bool hasEmissive = false;
            if (materialCount > 1)
            {
                for (int i = 0; i < materialCount; ++i)
                {
                    auto [mat, slot] = ctx_->getHandlers().disneyMaterialHandler->createDisneyMaterial(
                        model->S[i].cgMaterial, contentFolder, model);
                    triangleModel->getGeometryInstance()->setMaterial(0, i, mat);
                    
                    // Check if material is emissive - both luminous and luminousColor must be non-zero
                    float luminousSum = model->S[i].cgMaterial.emission.luminousColor.x() +
                                        model->S[i].cgMaterial.emission.luminousColor.y() +
                                        model->S[i].cgMaterial.emission.luminousColor.z();
                    if (model->S[i].cgMaterial.emission.luminous > 0.0f && luminousSum > 0.0f)
                    {
                        hasEmissive = true;
                    }
                }
            }
            else
            {
                auto [mat, slot] = ctx_->getHandlers().disneyMaterialHandler->createDisneyMaterial(
                    model->S[0].cgMaterial, contentFolder, model);
                triangleModel->getGeometryInstance()->setMaterial(0, 0, mat);
                
                // Check if material is emissive - both luminous and luminousColor must be non-zero
                float luminousSum = model->S[0].cgMaterial.emission.luminousColor.x() +
                                    model->S[0].cgMaterial.emission.luminousColor.y() +
                                    model->S[0].cgMaterial.emission.luminousColor.z();
                if (model->S[0].cgMaterial.emission.luminous > 0.0f && luminousSum > 0.0f)
                {
                    hasEmissive = true;
                }
            }
            
            // Mark the model as having emissive materials
            triangleModel->setHasEmissiveMaterials(hasEmissive);
            
            // If model has emissive materials, compute light probabilities
            if (hasEmissive)
            {
                computeLightProbabilities(triangleModel, geomInstSlot);
            }
        }

        // Create a Geometry Acceleration Structure (GAS) for ray tracing
        claudiaModel->createGAS (ctx_, scene, claudia_shared::maxNumRayTypes, scratchMem);
        
        // Populate geometry instance data in the global buffer
        if (geomInstSlot != SlotFinder::InvalidSlotIndex)
        {
            shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
            claudiaModel->populateGeometryInstanceData(&geomInstDataOnHost[geomInstSlot]);
            geometryInstanceDataBuffer_.unmap();
            LOG(DBUG) << "Populated geometry instance data for slot " << geomInstSlot;
        }

        // Create a regular instance in the scene
        sceneHandler_->createInstance(weakNode);
    }
}

// Adds multiple renderable nodes to the Claudia scene efficiently
// Optimizes the process by building models first, then creating instances
void ClaudiaModelHandler::addCgModelList(const WeakRenderableList& weakNodeList)
{
    // Start a scoped timer to measure processing time
    ScopedStopWatch sw("Processing model list in Claudia");

    // Verify handlers are set
    if (!ctx_->getHandlers().disneyMaterialHandler || !sceneHandler_)
    {
        LOG(WARNING) << "Material or scene handler not set - cannot process models";
        return;
    }

    cudau::Buffer& scratchMem = sceneHandler_->getASBuildScratchMem();

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
            
            ClaudiaModelPtr sourceModel = getClaudiaModel(instancedFrom->getClientID());
            if (!sourceModel)
            {
                LOG(WARNING) << "Source model not yet available for instance " << node->getName();
                continue;
            }
            
            // Create flyweight model that references another model's geometry
            ClaudiaModelPtr claudiaModel = ClaudiaFlyweightModel::create();
            
            // Flyweight models share the geometry instance slot with their source model
            claudiaModel->setGeomInstSlot(sourceModel->getGeomInstSlot());
            
            // Store the model with the client ID as key
            ItemID key = modelMgr.storeModel(claudiaModel, node->getClientID());
            
            // Count instances for logging
            ++instanceCount;
        }
        // Handle regular geometry nodes
        else
        {
            // Create a triangle model for regular geometry
            ClaudiaModelPtr claudiaModel = ClaudiaTriangleModel::create();
            
            // Allocate a geometry instance slot for this model
            uint32_t geomInstSlot = allocateGeometryInstanceSlot();
            if (geomInstSlot == SlotFinder::InvalidSlotIndex)
            {
                LOG(WARNING) << "Failed to allocate geometry instance slot for " << node->getName();
                continue;
            }
            claudiaModel->setGeomInstSlot(geomInstSlot);

            // Store the model with the client ID as key
            ItemID key = modelMgr.storeModel(claudiaModel, node->getClientID());
            
            // Build the geometry
            optixu::Scene scene = sceneHandler_->getScene();
            if (!scene)
            {
                LOG(WARNING) << "Scene not available from scene handler";
                continue;
            }
            claudiaModel->createGeometry(ctx_, node, scene);

            // Now create materials for each surface
            ClaudiaTriangleModel* triangleModel = dynamic_cast<ClaudiaTriangleModel*>(claudiaModel.get());
            if (triangleModel)
            {
                sabi::CgModelPtr model = node->getModel();
                fs::path contentFolder = ctx_->getPropertyService().renderProps->getVal<std::string>(RenderKey::ContentFolder);
                
                uint32_t materialCount = model->S.size();
                bool hasEmissive = false;
                if (materialCount > 1)
                {
                    for (int i = 0; i < materialCount; ++i)
                    {
                        auto [mat, slot] = ctx_->getHandlers().disneyMaterialHandler->createDisneyMaterial(
                            model->S[i].cgMaterial, contentFolder, model);
                        triangleModel->getGeometryInstance()->setMaterial(0, i, mat);
                        
                        // Check if material is emissive - both luminous and luminousColor must be non-zero
                        float luminousSum = model->S[i].cgMaterial.emission.luminousColor.x() +
                                            model->S[i].cgMaterial.emission.luminousColor.y() +
                                            model->S[i].cgMaterial.emission.luminousColor.z();
                        if (model->S[i].cgMaterial.emission.luminous > 0.0f && luminousSum > 0.0f)
                        {
                            hasEmissive = true;
                        }
                    }
                }
                else
                {
                    auto [mat, slot] = ctx_->getHandlers().disneyMaterialHandler->createDisneyMaterial(
                        model->S[0].cgMaterial, contentFolder, model);
                    triangleModel->getGeometryInstance()->setMaterial(0, 0, mat);
                    
                    // Check if material is emissive - both luminous and luminousColor must be non-zero
                    float luminousSum = model->S[0].cgMaterial.emission.luminousColor.x() +
                                        model->S[0].cgMaterial.emission.luminousColor.y() +
                                        model->S[0].cgMaterial.emission.luminousColor.z();
                    if (model->S[0].cgMaterial.emission.luminous > 0.0f && luminousSum > 0.0f)
                    {
                        hasEmissive = true;
                    }
                }
                
                // Mark the model as having emissive materials
                triangleModel->setHasEmissiveMaterials(hasEmissive);
                
                // If model has emissive materials, compute light probabilities
                if (hasEmissive)
                {
                    computeLightProbabilities(triangleModel, geomInstSlot);
                }
            }

            // Create a Geometry Acceleration Structure (GAS) for ray tracing
            optixu::Scene sceneForGAS = sceneHandler_->getScene();
            claudiaModel->createGAS (ctx_, sceneForGAS, claudia_shared::maxNumRayTypes, scratchMem);
            
            // Populate geometry instance data in the global buffer
            if (geomInstSlot != SlotFinder::InvalidSlotIndex)
            {
                shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
                claudiaModel->populateGeometryInstanceData(&geomInstDataOnHost[geomInstSlot]);
                geometryInstanceDataBuffer_.unmap();
            }
        }
    }

    // Create all instances in the scene at once for better performance
    sceneHandler_->createInstanceList(weakNodeList);

    // Log the results of the batch processing
    LOG(DBUG) << "Processed " << weakNodeList.size() << " models of which " << instanceCount << " are instances";
}

void ClaudiaModelHandler::removeModel(ItemID itemID)
{
    // Get the model before removing it so we can release its slot
    ClaudiaModelPtr model = modelMgr.retrieveModel(itemID);
    if (model)
    {
        uint32_t slot = model->getGeomInstSlot();
        if (slot != SlotFinder::InvalidSlotIndex)
        {
            // Only release slot for non-flyweight models
            // Flyweight models share slots with their source models
            ClaudiaFlyweightModel* flyweight = dynamic_cast<ClaudiaFlyweightModel*>(model.get());
            if (!flyweight)
            {
                releaseGeometryInstanceSlot(slot);
                LOG(DBUG) << "Released geometry instance slot " << slot << " for model ID " << itemID;
            }
        }
    }
    
    modelMgr.removeModel(itemID);
}

void ClaudiaModelHandler::setAreaLightHandler(std::shared_ptr<ClaudiaAreaLightHandler> areaLightHandler)
{
    areaLightHandler_ = areaLightHandler;
}

void ClaudiaModelHandler::computeLightProbabilities(ClaudiaTriangleModel* model, uint32_t geomInstSlot)
{
    if (!model || !engine_ || geomInstSlot == SlotFinder::InvalidSlotIndex)
    {
        LOG(WARNING) << "Cannot compute light probabilities: invalid parameters";
        return;
    }
    
    // Use area light handler if available
    if (areaLightHandler_)
    {
        // Prepare the geometry light distribution
        uint32_t numTriangles = model->getTriangleBuffer().numElements();
        areaLightHandler_->prepareGeometryLightDistribution(model, numTriangles);
        
        // Mark it as dirty so it will be updated in the next frame
        areaLightHandler_->markGeometryDirty(model);
        
        LOG(DBUG) << "Prepared light distribution for emissive geometry at slot " << geomInstSlot 
                  << " with " << numTriangles << " triangles";
    }
    else
    {
        LOG(WARNING) << "AreaLightHandler not set, cannot compute light probabilities";
    }
}