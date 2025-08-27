#include "RiPRModelHandler.h"
#include "RiPRSceneHandler.h"
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

        // Track material slots and emissive flags outside the triangle model block
        std::vector<uint32_t> materialSlots;
        std::vector<bool> isEmissive;
        
        // Now create materials for each surface
        RiPRTriangleModel* triangleModel = dynamic_cast<RiPRTriangleModel*>(riprModel.get());
        if (triangleModel)
        {
            sabi::CgModelPtr model = node->getModel();
            fs::path contentFolder = ctx_->getPropertyService().renderProps->getVal<std::string>(RenderKey::ContentFolder);
            
            uint32_t materialCount = model->S.size();
            
            if (materialCount > 1)
            {
                for (int i = 0; i < materialCount; ++i)
                {
                    auto [mat, slot] = ctx_->getHandlers().disneyMaterialHandler->createDisneyMaterial(
                        model->S[i].cgMaterial, contentFolder, model);
                    triangleModel->getGeometryInstance()->setMaterial(0, i, mat);
                    
                    materialSlots.push_back(slot);
                    isEmissive.push_back(model->S[i].cgMaterial.emission.luminous > 0.0f);
                }
            }
            else
            {
                auto [mat, slot] = ctx_->getHandlers().disneyMaterialHandler->createDisneyMaterial(
                    model->S[0].cgMaterial, contentFolder, model);
                triangleModel->getGeometryInstance()->setMaterial(0, 0, mat);
                
                materialSlots.push_back(slot);
                isEmissive.push_back(model->S[0].cgMaterial.emission.luminous > 0.0f);
            }
            
            // Check if any surface is emissive
            bool hasEmissive = std::any_of(isEmissive.begin(), isEmissive.end(), [](bool e) { return e; });
            
            // Mark the model as having emissive materials
            triangleModel->setHasEmissiveMaterials(hasEmissive);
        }

        // Create a Geometry Acceleration Structure (GAS) for ray tracing
        riprModel->createGAS (ctx_, scene, ripr_shared::maxNumRayTypes, scratchMem);
        
        // Populate geometry instance data in the global buffer - MUST be done before computing light probabilities
        if (geomInstSlot != SlotFinder::InvalidSlotIndex)
        {
            shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
            riprModel->populateGeometryInstanceData(&geomInstDataOnHost[geomInstSlot]);
            geometryInstanceDataBuffer_.unmap();
            LOG(DBUG) << "Populated geometry instance data for slot " << geomInstSlot;
            
            // If model has emissive materials, compute light probabilities AFTER geometry data is populated
            if (triangleModel && triangleModel->hasEmissiveMaterials())
            {
                // Pass the material slots and emissive flags to properly compute light probabilities
                // For now, just use the first emissive material found
                bool hasComputedEmissive = false;
                for (size_t i = 0; i < materialSlots.size(); ++i)
                {
                    if (isEmissive[i])
                    {
                        computeLightProbabilities(triangleModel, geomInstSlot, materialSlots[i]);
                        hasComputedEmissive = true;
                        break;  // TODO: Handle multiple emissive materials properly
                    }
                }
                
                // After computing light probabilities, trigger distribution building
                if (hasComputedEmissive && engine_)
                {
                    engine_->buildLightDistributions();
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
            if (triangleModel)
            {
                sabi::CgModelPtr model = node->getModel();
                fs::path contentFolder = ctx_->getPropertyService().renderProps->getVal<std::string>(RenderKey::ContentFolder);
                
                uint32_t materialCount = model->S.size();
                if (materialCount > 1)
                {
                    for (int i = 0; i < materialCount; ++i)
                    {
                        auto [mat, slot] = ctx_->getHandlers().disneyMaterialHandler->createDisneyMaterial(
                            model->S[i].cgMaterial, contentFolder, model);
                        triangleModel->getGeometryInstance()->setMaterial(0, i, mat);
                    }
                }
                else
                {
                    auto [mat, slot] = ctx_->getHandlers().disneyMaterialHandler->createDisneyMaterial(
                        model->S[0].cgMaterial, contentFolder, model);
                    triangleModel->getGeometryInstance()->setMaterial(0, 0, mat);
                }
            }

            // Create a Geometry Acceleration Structure (GAS) for ray tracing
            optixu::Scene sceneForGAS = sceneHandler_->getScene();
            riprModel->createGAS (ctx_, sceneForGAS, ripr_shared::maxNumRayTypes, scratchMem);
            
            // Populate geometry instance data in the global buffer
            if (geomInstSlot != SlotFinder::InvalidSlotIndex)
            {
                shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
                riprModel->populateGeometryInstanceData(&geomInstDataOnHost[geomInstSlot]);
                geometryInstanceDataBuffer_.unmap();
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

// computeLightProbabilities - Prepares a 3D model to act as a light source in the scene
//
// PURPOSE:
// In path tracing (a realistic rendering technique), any object with an emissive material can act as a light source.
// This includes things like lamp bulbs, TV screens, neon signs, or glowing hot metal. However, when we want to 
// efficiently sample these light sources during rendering, we need to know which triangles are more important
// (emit more light) so we can sample them more frequently. This function builds that importance distribution.
//
// WHAT IT DOES:
// 1. Takes a 3D model made of triangles and checks each triangle to see if it emits light
// 2. For each triangle, calculates an "importance" value based on:
//    - How much light it emits (brightness/color from its material's emission texture)
//    - How large the triangle is (bigger triangles emit more total light)
// 3. Creates a probability distribution that says "when picking a random light sample, prefer brighter/larger triangles"
// 4. This distribution is stored on the GPU for fast access during rendering
//
// HOW IT WORKS:
// Imagine you have a room with 100 light bulbs of different sizes and brightness:
// - Small, dim bulbs might get importance value of 0.1
// - Large, bright bulbs might get importance value of 10.0
// When the renderer needs to sample a light, it uses these weights to pick bulbs probabilistically:
// - The bright bulb is 100x more likely to be chosen than the dim one
// - This reduces noise in the final image because we sample important lights more often
//
// TECHNICAL DETAILS:
// - Runs a CUDA kernel (GPU program) that processes all triangles in parallel
// - Each triangle's importance = emission_color_brightness × triangle_area
// - Builds a CDF (Cumulative Distribution Function) for efficient random sampling
// - The CDF allows O(log n) sampling using binary search during rendering
//
// PARAMETERS:
// - model: The 3D model whose triangles we're analyzing for light emission
// - geomInstSlot: Index in the global geometry buffer where this model's data is stored
//
// EXAMPLE:
// A mesh representing a neon sign would have high importance values for the glowing tube triangles
// and zero importance for the non-emissive mounting bracket triangles. During rendering, the path
// tracer would almost always sample points on the glowing tubes when looking for light from this object.
void RiPRModelHandler::computeLightProbabilities(RiPRTriangleModel* model, uint32_t geomInstSlot, uint32_t materialSlot)
{
    if (!model || !engine_ || geomInstSlot == SlotFinder::InvalidSlotIndex)
    {
        LOG(WARNING) << "Cannot compute light probabilities: invalid parameters";
        return;
    }
    
    // Get geometry instance buffer device pointer
    shared::GeometryInstanceData* geomInstData = geometryInstanceDataBuffer_.getDevicePointer();
    if (!geomInstData)
    {
        LOG(WARNING) << "Geometry instance buffer not available";
        return;
    }
    
    // Get material buffer from DisneyMaterialHandler
    auto materialHandler = ctx_->getHandlers().disneyMaterialHandler;
    if (!materialHandler)
    {
        LOG(WARNING) << "DisneyMaterialHandler not available";
        return;
    }
    
    cudau::TypedBuffer<shared::DisneyData>* materialBuffer = materialHandler->getMaterialDataBuffer();
    if (!materialBuffer)
    {
        LOG(WARNING) << "Material buffer not available";
        return;
    }
    
    shared::DisneyData* materialData = materialBuffer->getDevicePointer();
    if (!materialData)
    {
        LOG(WARNING) << "Material buffer device pointer not available";
        return;
    }
    
    // Get number of triangles from the model's triangle buffer
    uint32_t numTriangles = model->getTriangleBuffer().numElements();
    if (numTriangles == 0)
    {
        LOG(WARNING) << "No triangles in model";
        return;
    }
    
    // Initialize the emitter distribution if not already initialized
    if (!model->getEmitterPrimDist().isInitialized())
    {
        model->getEmitterPrimDist().initialize(
            ctx_->getCudaContext(), cudau::BufferType::Device, nullptr, numTriangles);
        LOG(DBUG) << "Initialized emitter distribution for " << numTriangles << " triangles";
        
        // Update the device buffer with the newly initialized distribution
        shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
        model->getEmitterPrimDist().getDeviceType(&geomInstDataOnHost[geomInstSlot].emitterPrimDist);
        geometryInstanceDataBuffer_.unmap();
    }
    
    // Get RiPREngine to access kernel handles
    RiPREngine* riprEngine = dynamic_cast<RiPREngine*>(engine_);
    if (!riprEngine)
    {
        LOG(WARNING) << "Engine is not RiPREngine";
        return;
    }
    
    const RiPREngine::ComputeProbTex& kernels = riprEngine->getComputeProbTex();
    CUstream stream = ctx_->getCudaStream();
    
    // Get pointer to the specific material's data
    shared::DisneyData* specificMaterial = &materialData[materialSlot];
    
    // Launch kernel to compute triangle probabilities
    kernels.computeTriangleProbBuffer(
        stream,
        kernels.computeTriangleProbBuffer.calcGridDim(numTriangles),
        &geomInstData[geomInstSlot],
        numTriangles,
        specificMaterial  // Pass specific material instead of whole buffer
    );
    
    // Compute CDF from weights using CUB scan
    // This is required before calling finalize!
    size_t scratchMemSize = 0;
    CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
        nullptr, scratchMemSize,
        model->getEmitterPrimDist().weightsOnDevice(),
        model->getEmitterPrimDist().cdfOnDevice(),
        numTriangles, stream));
    
    cudau::Buffer scanScratchMem;
    scanScratchMem.initialize(ctx_->getCudaContext(), cudau::BufferType::Device, scratchMemSize, 1u);
    
    CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
        scanScratchMem.getDevicePointer(), scratchMemSize,
        model->getEmitterPrimDist().weightsOnDevice(),
        model->getEmitterPrimDist().cdfOnDevice(),
        numTriangles, stream));
    
    scanScratchMem.finalize();
    
    // Finalize the distribution (build CDF)
    kernels.finalizeDiscreteDistribution1D(
        stream,
        kernels.finalizeDiscreteDistribution1D.calcGridDim(1),
        &geomInstData[geomInstSlot].emitterPrimDist
    );
    
    // Synchronize to ensure completion
    CUDADRV_CHECK(cuStreamSynchronize(stream));
    
    // Optional: Check if any triangles were marked as emissive
    // Read back just the integral value from the distribution
    shared::LightDistribution lightDistOnHost;
    CUDADRV_CHECK(cuMemcpyDtoH(&lightDistOnHost, 
                               reinterpret_cast<CUdeviceptr>(&geomInstData[geomInstSlot].emitterPrimDist),
                               sizeof(lightDistOnHost)));
    
    float integral = lightDistOnHost.integral();
    if (integral > 0.0f)
    {
        LOG(INFO) << "Computed light probabilities for " << numTriangles 
                  << " triangles at slot " << geomInstSlot 
                  << " (integral: " << integral << ")";
        
        // Mark the model as having emissive materials
        if (auto triModel = dynamic_cast<RiPRTriangleModel*>(model))
        {
            triModel->setHasEmissiveMaterials(true);
        }
        
        // Update the host-side geometry instance data with the computed distribution
        // This ensures the distribution is available for future calls to populateGeometryInstanceData
        shared::GeometryInstanceData* geomInstDataOnHost = geometryInstanceDataBuffer_.map();
        model->getEmitterPrimDist().getDeviceType(&geomInstDataOnHost[geomInstSlot].emitterPrimDist);
        geometryInstanceDataBuffer_.unmap();
    }
    else
    {
        LOG(DBUG) << "No emissive triangles found in " << numTriangles 
                  << " triangles at slot " << geomInstSlot;
        
        // IMPORTANT: Clear the emissive flag since no actual emissive triangles were found
        // This handles cases where the material has emission.luminous > 0 but the actual
        // computed emission is zero (e.g., due to other material properties)
        if (auto triModel = dynamic_cast<RiPRTriangleModel*>(model))
        {
            triModel->setHasEmissiveMaterials(false);
        }
    }
}