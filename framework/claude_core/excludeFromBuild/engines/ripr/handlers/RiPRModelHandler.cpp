#include "RiPRModelHandler.h"
#include "RiPRSceneHandler.h"
#include "RiPRAreaLightHandler.h"
#include "../RiPREngine.h"
#include "../models/RiPRModel.h"
#include "../../../handlers/Handlers.h"


// Constructor initializes the model handler with the render context
RiPRModelHandler::RiPRModelHandler(RenderContextPtr ctx) :
    ctx_(ctx)
{
    LOG(DBUG) << _FN_;
}

// Destructor handles cleanup of model resources
RiPRModelHandler::~RiPRModelHandler()
{
    finalize();
}

// Initialize the model handler with buffer allocation for geometry instances
void RiPRModelHandler::initialize()
{
    if (isInitialized_)
    {
        LOG(WARNING) << "RiPRModelHandler already initialized";
        return;
    }
    
    LOG(INFO) << "Initializing RiPRModelHandler";
    
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
    // Using Device buffer type for better performance (same as RiPR)
    geometryInstanceDataBuffer_.initialize(cuContext, cudau::BufferType::Device, maxNumGeometryInstances);
    LOG(INFO) << "Initialized geometry instance data buffer";
    
    isInitialized_ = true;
    LOG(INFO) << "RiPRModelHandler initialized successfully";
}

// Finalize and clean up all resources
void RiPRModelHandler::finalize()
{
    if (!isInitialized_)
        return;
        
    LOG(INFO) << "Finalizing RiPRModelHandler";
    
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
        
        LOG(INFO) << "RiPRModelHandler finalized successfully";
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Exception during RiPRModelHandler finalization: " << e.what();
    }
    
    isInitialized_ = false;
}

// Adds a single renderable node to the RiPR scene
// Creates appropriate model type based on node characteristics
void RiPRModelHandler::addCgModel(RenderableWeakRef weakNode)
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

        RiPRModelPtr sourceModel = getRiPRModel(instancedFrom->getClientID());
        if (!sourceModel)
        {
            LOG(WARNING) << "Source model not yet available for instance " << node->getName()
                         << " - source: " << instancedFrom->getName()
                         << " clientID: " << instancedFrom->getClientID();
            // Store this in a pending list to try again later
            return;
        }

        // Create a flyweight model that references another model's geometry
        RiPRModelPtr riprModel = RiPRFlyweightModel::create();
        if (!riprModel)
        {
            LOG(WARNING) << "Failed to create RiPRModelPtr";
            return;
        }

        // Flyweight models share the geometry instance slot with their source model
        riprModel->setGeomInstSlot(sourceModel->getGeomInstSlot());
        LOG(DBUG) << "Flyweight model " << node->getName() 
                  << " sharing geometry instance slot " << sourceModel->getGeomInstSlot()
                  << " with source model " << instancedFrom->getName();

        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(riprModel, node->getClientID());

        // Create a geometry instance in the scene
        sceneHandler_->createGeometryInstance(weakNode);
    }
    // Handle phantom nodes (used for physics/collision)
    else if (node->isPhantom())
    {
        // Create a phantom model for collision detection
        RiPRModelPtr riprModel = RiPRPhantomModel::create();
        
        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(riprModel, node->getClientID());
        
        // Create a physics phantom in the scene
        sceneHandler_->createPhysicsPhantom(weakNode);
    }
    // Handle regular geometry nodes
    else
    {
        // Create a triangle model for regular geometry
        RiPRModelPtr riprModel = RiPRTriangleModel::create();

        // Allocate a geometry instance slot for this model
        uint32_t geomInstSlot = allocateGeometryInstanceSlot();
        if (geomInstSlot == SlotFinder::InvalidSlotIndex)
        {
            LOG(WARNING) << "Failed to allocate geometry instance slot for " << node->getName();
            return;
        }
        riprModel->setGeomInstSlot(geomInstSlot);
        LOG(DBUG) << "Allocated geometry instance slot " << geomInstSlot << " for " << node->getName();

        // Store the model with the client ID as key
        ItemID key = modelMgr.storeModel(riprModel, node->getClientID());

        // Build the geometry
        optixu::Scene scene = sceneHandler_->getScene();
        if (!scene)
        {
            LOG(WARNING) << "Scene not available from scene handler";
            return;
        }
        riprModel->createGeometry(ctx_, node, scene);

        // Now create materials for each surface
        RiPRTriangleModel* triangleModel = dynamic_cast<RiPRTriangleModel*>(riprModel.get());
        std::vector<uint32_t> materialSlots;
        std::vector<bool> isEmissiveMat;
        
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
                    materialSlots.push_back(slot);
                    
                    // Check if material is emissive - both luminous and luminousColor must be non-zero
                    float luminousSum = model->S[i].cgMaterial.emission.luminousColor.x() +
                                        model->S[i].cgMaterial.emission.luminousColor.y() +
                                        model->S[i].cgMaterial.emission.luminousColor.z();
                    bool matEmissive = (model->S[i].cgMaterial.emission.luminous > 0.0f && luminousSum > 0.0f);
                    isEmissiveMat.push_back(matEmissive);
                    if (matEmissive)
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
                materialSlots.push_back(slot);
                
                // Check if material is emissive - both luminous and luminousColor must be non-zero
                float luminousSum = model->S[0].cgMaterial.emission.luminousColor.x() +
                                    model->S[0].cgMaterial.emission.luminousColor.y() +
                                    model->S[0].cgMaterial.emission.luminousColor.z();
                bool matEmissive = (model->S[0].cgMaterial.emission.luminous > 0.0f && luminousSum > 0.0f);
                isEmissiveMat.push_back(matEmissive);
                if (matEmissive)
                {
                    hasEmissive = true;
                }
            }
            
            // Mark the model as having emissive materials
            triangleModel->setHasEmissiveMaterials(hasEmissive);
        }

        // Create a Geometry Acceleration Structure (GAS) for ray tracing
        riprModel->createGAS (ctx_, scene, ripr_shared::maxNumRayTypes, scratchMem);
        
        // Populate geometry instance data in the global buffer BEFORE computing light probabilities
        if (geomInstSlot != SlotFinder::InvalidSlotIndex)
        {
            shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
            riprModel->populateGeometryInstanceData(&geomInstDataOnHost[geomInstSlot]);
            geometryInstanceDataBuffer_.unmap();
            LOG(DBUG) << "Populated geometry instance data for slot " << geomInstSlot;
        }
        
        // Now that buffers are populated, compute light probabilities if model has emissive materials
        if (triangleModel && triangleModel->hasEmissiveMaterials())
        {
            // Find first emissive material and compute probabilities for it
            // TODO: Handle multiple emissive materials properly
            for (size_t i = 0; i < materialSlots.size(); ++i)
            {
                if (isEmissiveMat[i])
                {
                    computeLightProbabilities(triangleModel, geomInstSlot, materialSlots[i]);
                    break; // TODO: Handle multiple emissive materials
                }
            }
        }

        // Create a regular instance in the scene
        sceneHandler_->createInstance(weakNode);
    }
}

// Adds multiple renderable nodes to the RiPR scene efficiently
// Optimizes the process by building models first, then creating instances
void RiPRModelHandler::addCgModelList(const WeakRenderableList& weakNodeList)
{
    // Start a scoped timer to measure processing time
    ScopedStopWatch sw("Processing model list in RiPR");

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
            
            RiPRModelPtr sourceModel = getRiPRModel(instancedFrom->getClientID());
            if (!sourceModel)
            {
                LOG(WARNING) << "Source model not yet available for instance " << node->getName();
                continue;
            }
            
            // Create flyweight model that references another model's geometry
            RiPRModelPtr riprModel = RiPRFlyweightModel::create();
            
            // Flyweight models share the geometry instance slot with their source model
            riprModel->setGeomInstSlot(sourceModel->getGeomInstSlot());
            
            // Store the model with the client ID as key
            ItemID key = modelMgr.storeModel(riprModel, node->getClientID());
            
            // Count instances for logging
            ++instanceCount;
        }
        // Handle regular geometry nodes
        else
        {
            // Create a triangle model for regular geometry
            RiPRModelPtr riprModel = RiPRTriangleModel::create();
            
            // Allocate a geometry instance slot for this model
            uint32_t geomInstSlot = allocateGeometryInstanceSlot();
            if (geomInstSlot == SlotFinder::InvalidSlotIndex)
            {
                LOG(WARNING) << "Failed to allocate geometry instance slot for " << node->getName();
                continue;
            }
            riprModel->setGeomInstSlot(geomInstSlot);

            // Store the model with the client ID as key
            ItemID key = modelMgr.storeModel(riprModel, node->getClientID());
            
            // Build the geometry
            optixu::Scene scene = sceneHandler_->getScene();
            if (!scene)
            {
                LOG(WARNING) << "Scene not available from scene handler";
                continue;
            }
            riprModel->createGeometry(ctx_, node, scene);

            // Now create materials for each surface
            RiPRTriangleModel* triangleModel = dynamic_cast<RiPRTriangleModel*>(riprModel.get());
            std::vector<uint32_t> materialSlots;
            std::vector<bool> isEmissiveMat;
            
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
                        materialSlots.push_back(slot);
                        
                        // Check if material is emissive - both luminous and luminousColor must be non-zero
                        float luminousSum = model->S[i].cgMaterial.emission.luminousColor.x() +
                                            model->S[i].cgMaterial.emission.luminousColor.y() +
                                            model->S[i].cgMaterial.emission.luminousColor.z();
                        bool matEmissive = (model->S[i].cgMaterial.emission.luminous > 0.0f && luminousSum > 0.0f);
                        isEmissiveMat.push_back(matEmissive);
                        if (matEmissive)
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
                    materialSlots.push_back(slot);
                    
                    // Check if material is emissive - both luminous and luminousColor must be non-zero
                    float luminousSum = model->S[0].cgMaterial.emission.luminousColor.x() +
                                        model->S[0].cgMaterial.emission.luminousColor.y() +
                                        model->S[0].cgMaterial.emission.luminousColor.z();
                    bool matEmissive = (model->S[0].cgMaterial.emission.luminous > 0.0f && luminousSum > 0.0f);
                    isEmissiveMat.push_back(matEmissive);
                    if (matEmissive)
                    {
                        hasEmissive = true;
                    }
                }
                
                // Mark the model as having emissive materials
                triangleModel->setHasEmissiveMaterials(hasEmissive);
                
            }

            // Create a Geometry Acceleration Structure (GAS) for ray tracing
            optixu::Scene sceneForGAS = sceneHandler_->getScene();
            riprModel->createGAS (ctx_, sceneForGAS, ripr_shared::maxNumRayTypes, scratchMem);
            
            // Populate geometry instance data in the global buffer BEFORE computing light probabilities
            if (geomInstSlot != SlotFinder::InvalidSlotIndex)
            {
                shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
                riprModel->populateGeometryInstanceData(&geomInstDataOnHost[geomInstSlot]);
                geometryInstanceDataBuffer_.unmap();
            }
            
            // Now that buffers are populated, compute light probabilities if model has emissive materials
            if (triangleModel && triangleModel->hasEmissiveMaterials())
            {
                // Find first emissive material and compute probabilities for it
                // TODO: Handle multiple emissive materials properly
                for (size_t i = 0; i < materialSlots.size(); ++i)
                {
                    if (isEmissiveMat[i])
                    {
                        computeLightProbabilities(triangleModel, geomInstSlot, materialSlots[i]);
                        break; // TODO: Handle multiple emissive materials
                    }
                }
            }
        }
    }

    // Create all instances in the scene at once for better performance
    sceneHandler_->createInstanceList(weakNodeList);

    // Log the results of the batch processing
    LOG(DBUG) << "Processed " << weakNodeList.size() << " models of which " << instanceCount << " are instances";
}

void RiPRModelHandler::removeModel(ItemID itemID)
{
    // Get the model before removing it so we can release its slot
    RiPRModelPtr model = modelMgr.retrieveModel(itemID);
    if (model)
    {
        uint32_t slot = model->getGeomInstSlot();
        if (slot != SlotFinder::InvalidSlotIndex)
        {
            // Only release slot for non-flyweight models
            // Flyweight models share slots with their source models
            RiPRFlyweightModel* flyweight = dynamic_cast<RiPRFlyweightModel*>(model.get());
            if (!flyweight)
            {
                releaseGeometryInstanceSlot(slot);
                LOG(DBUG) << "Released geometry instance slot " << slot << " for model ID " << itemID;
            }
        }
    }
    
    modelMgr.removeModel(itemID);
}

void RiPRModelHandler::setAreaLightHandler(std::shared_ptr<RiPRAreaLightHandler> areaLightHandler)
{
    areaLightHandler_ = areaLightHandler;
}

// Compute light probabilities for a triangle model with emissive material
// Similar to RiPRModelHandler::computeLightProbabilities
void RiPRModelHandler::computeLightProbabilities(RiPRTriangleModel* model, uint32_t geomInstSlot, uint32_t materialSlot)
{
    if (!model || !engine_ || geomInstSlot == SlotFinder::InvalidSlotIndex)
    {
        LOG(WARNING) << "Cannot compute light probabilities: invalid parameters";
        return;
    }
    
    // Get area light handler from engine
    if (!areaLightHandler_)
    {
        LOG(WARNING) << "AreaLightHandler not available";
        return;
    }
    
    // Get number of triangles from the model's triangle buffer
    uint32_t numTriangles = model->getTriangleBuffer().numElements();
    if (numTriangles == 0)
    {
        LOG(WARNING) << "No triangles in model";
        return;
    }
    
    LOG(INFO) << "Computing light probabilities for " << numTriangles << " triangles with material slot " << materialSlot;
    
    // Initialize the emitter distribution if not already initialized
    if (!model->getEmitterPrimDistribution().isInitialized())
    {
        // Get non-const reference through the model's public method
        LightDistribution* emitterDist = model->getTriLightImportance();
        emitterDist->initialize(
            ctx_->getCudaContext(), cudau::BufferType::Device, nullptr, numTriangles);
        LOG(DBUG) << "Initialized emitter distribution for " << numTriangles << " triangles";
        
        // Update the device buffer with the newly initialized distribution
        shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
        
        // Debug: Check the distribution before copying
        LOG(INFO) << "Before getDeviceType:";
        LOG(INFO) << "  emitterDist isInitialized: " << emitterDist->isInitialized();
        
        emitterDist->getDeviceType(&geomInstDataOnHost[geomInstSlot].emitterPrimDist);
        
        // The device struct members are private, so we can't directly access them from host
        LOG(INFO) << "Called getDeviceType to update device struct";
        
        geometryInstanceDataBuffer_.unmap();
        LOG(DBUG) << "Updated geometry instance data with emitter distribution";
    }
    
    // Prepare geometry light distribution if not already done
    areaLightHandler_->prepareGeometryLightDistribution(model, numTriangles);
    
    // Update geometry light distribution with the material slot
    CUstream stream = ctx_->getCudaStream();
    areaLightHandler_->updateGeometryLightDistribution(stream, model, geomInstSlot, materialSlot);
    
    // Finalize the emitter distribution (compute CDF from weights)
    // This is critical for proper light sampling in the path tracer
    LightDistribution* emitterDist = model->getTriLightImportance();
    
#if USE_PROBABILITY_TEXTURE
    // For probability texture, generate mip levels
    uint2 dims = shared::computeProbabilityTextureDimentions(numTriangles);
    uint32_t numMipLevels = shared::nextPowOf2Exponent(dims.x) + 1;
    
    // Generate mip levels for hierarchical sampling
    for (int dstMipLevel = 1; dstMipLevel < numMipLevels; ++dstMipLevel) {
        dims = (dims + uint2(1, 1)) / 2;
        // TODO: Call computeMip kernel when available
        // areaLightHandler_->computeMip(stream, emitterDist, dstMipLevel, dims);
    }
#else
    // For discrete distribution, finalize to compute the integral
    // The CDF computation will be done by the finalizeEmitterDistribution method
    // Model is already a raw pointer to RiPRTriangleModel, we need to wrap it
    // Since we don't own the model, use a non-owning shared_ptr
    RiPRModelPtr modelPtr(model, [](RiPRModel*){});  // Empty deleter since we don't own it
    areaLightHandler_->finalizeEmitterDistribution(stream, modelPtr, geomInstSlot);
#endif
    
    LOG(DBUG) << "Completed light probability computation for geometry instance";
}