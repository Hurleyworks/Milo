#include "ClaudiaAreaLightHandler.h"
#include "../ClaudiaEngine.h"
#include "../models/ClaudiaModel.h"
#include "ClaudiaModelHandler.h"
#include "ClaudiaSceneHandler.h"
#include "../../../handlers/Handlers.h"
#include "../../../handlers/DisneyMaterialHandler.h"
#include "../../../common/common_shared.h"

ClaudiaAreaLightHandler::ClaudiaAreaLightHandler(RenderContextPtr ctx)
    : ctx_(ctx)
{
    LOG(DBUG) << _FN_;
}

ClaudiaAreaLightHandler::~ClaudiaAreaLightHandler()
{
    finalize();
}

void ClaudiaAreaLightHandler::initialize()
{
    if (isInitialized_)
    {
        LOG(WARNING) << "ClaudiaAreaLightHandler already initialized";
        return;
    }
    
    LOG(INFO) << "Initializing ClaudiaAreaLightHandler";
    
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
    
    // Initialize scratch memory for scan operations
    size_t scanScratchSize;
    constexpr int32_t maxScanSize = std::max<int32_t>({
        maxNumMaterials,
        maxNumGeometryInstances,
        maxNumInstances
    });
    
    // Use CUB device scan for computing exclusive sum (CDF)
    CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
        nullptr, scanScratchSize,
        static_cast<float*>(nullptr), static_cast<float*>(nullptr), maxScanSize));
    
    scanScratchMem_.initialize(cuContext, cudau::BufferType::Device, scanScratchSize, 1u);
    scanScratchSize_ = scanScratchSize;
    LOG(DBUG) << "Allocated scan scratch memory: " << scanScratchSize << " bytes";
    
    // Initialize scene-level light distribution
#if USE_PROBABILITY_TEXTURE
    sceneLightDist_.initialize(cuContext, maxNumInstances);
#else
    // For discrete distribution, we need to provide initial values
    std::vector<float> initialWeights(maxNumInstances, 0.0f);
    sceneLightDist_.initialize(cuContext, cudau::BufferType::Device, 
                              initialWeights.data(), maxNumInstances);
#endif
    LOG(DBUG) << "Initialized scene light distribution for " << maxNumInstances << " instances";
    
    maxInstances_ = maxNumInstances;
    maxGeomInstances_ = maxNumGeometryInstances;
    
    // Initialize light computation kernels
    initializeLightKernels();
    
    isInitialized_ = true;
    LOG(INFO) << "ClaudiaAreaLightHandler initialized successfully";
}

void ClaudiaAreaLightHandler::finalize()
{
    if (!isInitialized_)
        return;
        
    LOG(INFO) << "Finalizing ClaudiaAreaLightHandler";
    
    try
    {
        // First sync CUDA operations if context is available
        if (ctx_ && ctx_->getCudaStream())
        {
            CUDADRV_CHECK(cuStreamSynchronize(ctx_->getCudaStream()));
        }
        
        // Clean up light kernels
        cleanupLightKernels();
        
        // Clean up light distributions
        sceneLightDist_.finalize();
        
        // Clear instance distributions
        for (auto& pair : instanceLightDists_)
        {
            pair.second.finalize();
        }
        instanceLightDists_.clear();
        
        // We don't manage geometry distributions anymore - they're owned by models
        
        // Clean up scratch memory
        if (scanScratchMem_.isInitialized())
        {
            scanScratchMem_.finalize();
        }
        
        LOG(INFO) << "ClaudiaAreaLightHandler finalized successfully";
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Exception during ClaudiaAreaLightHandler finalization: " << e.what();
    }
    
    isInitialized_ = false;
}

void ClaudiaAreaLightHandler::prepareSceneLightDistribution(uint32_t maxInstances)
{
    if (maxInstances > maxInstances_)
    {
        LOG(WARNING) << "Requested instances (" << maxInstances 
                     << ") exceeds maximum (" << maxInstances_ << ")";
        return;
    }
    
    // Only log on first preparation or when count changes
    if (!sceneDistributionPrepared_)
    {
        LOG(DBUG) << "Scene light distribution prepared with capacity for up to " << maxInstances << " instances";
        sceneDistributionPrepared_ = true;
    }
}

void ClaudiaAreaLightHandler::updateSceneLightDistribution(CUstream stream, uint32_t bufferIndex)
{
    if (!sceneHandler_)
    {
        LOG(WARNING) << "Scene handler not set - cannot update scene distribution";
        return;
    }
    
    // Get the current number of instances
    uint32_t numInstances = sceneHandler_->getInstanceCount();
    if (numInstances == 0)
    {
        LOG(DBUG) << "No instances in scene, skipping light distribution update";
        return;
    }
    
    // Only update if the number of instances has changed or we have dirty distributions
    if (numInstances != lastUpdateNumInstances_ || hasDirtyDistributions())
    {
        // Update the scene light distribution with instance probabilities
        computeInstProbabilities(stream, numInstances);
        lastUpdateNumInstances_ = numInstances;
    }
}

void ClaudiaAreaLightHandler::setupSceneLightSampling(CUstream stream, CUdeviceptr lightInstDistAddr, uint32_t bufferIndex)
{
    // Copy the scene light distribution address to the specified device location
    if (lightInstDistAddr)
    {
#if USE_PROBABILITY_TEXTURE
        shared::ProbabilityTexture deviceDist;
        sceneLightDist_.getDeviceType(&deviceDist);
#else
        shared::DiscreteDistribution1D deviceDist;
        sceneLightDist_.getDeviceType(&deviceDist);
#endif
        
        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            lightInstDistAddr,
            &deviceDist,
            sizeof(deviceDist),
            stream
        ));
    }
}

void ClaudiaAreaLightHandler::prepareInstanceLightDistribution(ClaudiaModel* model)
{
    if (!model)
        return;
        
    // Check if we already have a distribution for this instance
    auto it = instanceLightDists_.find(model);
    if (it == instanceLightDists_.end())
    {
        // Create new distribution for this instance
        CUcontext cuContext = ctx_->getCudaContext();
        LightDistribution dist;
        
#if USE_PROBABILITY_TEXTURE
        dist.initialize(cuContext, maxGeomInstances_);
#else
        std::vector<float> initialWeights(maxGeomInstances_, 0.0f);
        dist.initialize(cuContext, cudau::BufferType::Device,
                       initialWeights.data(), maxGeomInstances_);
#endif
        
        instanceLightDists_[model] = std::move(dist);
        LOG(DBUG) << "Created instance light distribution for model";
    }
    
    dirtyInstances_.insert(model);
}

void ClaudiaAreaLightHandler::updateInstanceLightDistribution(CUstream stream, ClaudiaModel* model, uint32_t instSlot)
{
    if (!model)
        return;
        
    auto it = instanceLightDists_.find(model);
    if (it == instanceLightDists_.end())
    {
        LOG(WARNING) << "No instance light distribution found for model";
        return;
    }
    
    // Mark as dirty for batch update
    dirtyInstances_.insert(model);
}

void ClaudiaAreaLightHandler::markInstanceDirty(ClaudiaModel* model)
{
    if (model)
    {
        dirtyInstances_.insert(model);
    }
}

void ClaudiaAreaLightHandler::prepareGeometryLightDistribution(ClaudiaTriangleModel* triModel, uint32_t numTriangles)
{
    if (!triModel || numTriangles == 0)
    {
        LOG(WARNING) << "Invalid model or triangle count for geometry light distribution";
        return;
    }
    
    // We don't create a new distribution here - the model already has one
    // initialized in ClaudiaModelHandler::computeLightProbabilities
    // Just mark it as dirty so we know to compute probabilities for it
    LOG(DBUG) << "Marking geometry for probability computation: " << numTriangles << " triangles";
    
    dirtyGeometries_.insert(triModel);
}

void ClaudiaAreaLightHandler::updateGeometryLightDistribution(CUstream stream, ClaudiaTriangleModel* triModel, uint32_t geomInstSlot, uint32_t materialSlot)
{
    if (!triModel)
        return;
        
    // Get the model's emitter distribution (not creating a new one)
    LightDistribution* modelDist = triModel->getTriLightImportance();
    if (!modelDist || !modelDist->isInitialized())
    {
        LOG(WARNING) << "Model's emitter distribution not initialized";
        return;
    }
    
    // Get triangle count
    uint32_t numTriangles = triModel->getTriangleBuffer().numElements();
    
    // Compute triangle probabilities with material slot using the model's distribution
    computeTriangleProbabilities(stream, geomInstSlot, numTriangles, materialSlot, *modelDist);
    
    // Finalize the distribution (compute CDF)
    finalizeLightDistribution(stream, *modelDist, numTriangles);
}

void ClaudiaAreaLightHandler::markGeometryDirty(ClaudiaTriangleModel* triModel)
{
    if (triModel)
    {
        dirtyGeometries_.insert(triModel);
    }
}

void ClaudiaAreaLightHandler::rebuildAllDistributions(CUstream stream)
{
    LOG(INFO) << "Rebuilding all light distributions";
    
    // Mark all instance distributions as dirty
    for (auto& pair : instanceLightDists_)
    {
        dirtyInstances_.insert(pair.first);
    }
    
    // Note: Geometry distributions are handled by models themselves now
    // We just need to mark them dirty if we're tracking them
    
    // Update all dirty distributions
    updateDirtyDistributions(stream);
    
    needsFullRebuild_ = false;
}

void ClaudiaAreaLightHandler::updateDirtyDistributions(CUstream stream, uint32_t bufferIndex)
{
    // Process dirty geometries first (lowest level)
    // Since geometry distributions are now owned by models and computed
    // via updateGeometryLightDistribution, we just clear the dirty set
    if (!dirtyGeometries_.empty())
    {
        LOG(DBUG) << "Clearing " << dirtyGeometries_.size() << " dirty geometry flags";
        dirtyGeometries_.clear();
    }
    
    // Process dirty instances (middle level)
    for (auto* model : dirtyInstances_)
    {
        auto it = instanceLightDists_.find(model);
        if (it != instanceLightDists_.end())
        {
            // For instances, we need to aggregate geometry instance probabilities
            // This would require knowing which geometry instances belong to this instance
            // For now, we'll skip the detailed implementation
            LOG(DBUG) << "Updating instance light distribution";
        }
    }
    dirtyInstances_.clear();
    
    // Update scene-level distribution if needed
    updateSceneLightDistribution(stream, bufferIndex);
}

float ClaudiaAreaLightHandler::getTotalLightPower() const
{
    // TODO: Calculate total light power from all distributions
    return 1.0f;
}

void ClaudiaAreaLightHandler::resizeScratchMemory(size_t requiredSize)
{
    if (requiredSize > scanScratchSize_)
    {
        CUcontext cuContext = ctx_->getCudaContext();
        
        if (scanScratchMem_.isInitialized())
        {
            scanScratchMem_.finalize();
        }
        
        scanScratchMem_.initialize(cuContext, cudau::BufferType::Device, requiredSize, 1u);
        scanScratchSize_ = requiredSize;
        
        LOG(DBUG) << "Resized scratch memory to " << requiredSize << " bytes";
    }
}

// Private helper methods

void ClaudiaAreaLightHandler::computeTriangleProbabilities(
    CUstream stream, 
    uint32_t geomInstSlot,
    uint32_t numTriangles,
    uint32_t materialSlot,
    LightDistribution& dist)
{
    if (!modelHandler_)
    {
        LOG(WARNING) << "ModelHandler not available";
        return;
    }
    
    // Get DisneyMaterialHandler from RenderContext
    auto materialHandler = ctx_->getHandlers().disneyMaterialHandler;
    if (!materialHandler)
    {
        LOG(WARNING) << "DisneyMaterialHandler not available";
        return;
    }
    
    // Get material buffer
    auto* materialBuffer = materialHandler->getMaterialDataBuffer();
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
    
    // Get specific material pointer
    shared::DisneyData* specificMaterial = &materialData[materialSlot];
    
    // Get geometry instance data from model handler
    auto* geomInstBuffer = modelHandler_->getGeometryInstanceDataBuffer();
    shared::GeometryInstanceData* geomInstData = geomInstBuffer->getDevicePointer();
    if (!geomInstData)
    {
        LOG(WARNING) << "Geometry instance buffer device pointer not available";
        return;
    }
    
#if USE_PROBABILITY_TEXTURE
    // Implementation for texture-based computation
    LOG(DBUG) << "Computing triangle probabilities for " << numTriangles << " triangles with material slot " << materialSlot;
    
    // Call the kernel with the material pointer and the output buffer
    // Note: cudau::Kernel doesn't have isValid(), we check if the module is loaded
    if (computeProbTex_.cudaModule)
    {
        // Get the mip buffer where probabilities will be written
        float* mipBuffer = dist.getMipBuffer(0);  // First mip level
        if (!mipBuffer)
        {
            LOG(WARNING) << "Mip buffer not available";
            return;
        }
        
        computeProbTex_.computeTriangleProbTexture(
            stream,
            computeProbTex_.computeTriangleProbTexture.calcGridDim(numTriangles),
            &geomInstData[geomInstSlot],
            numTriangles,
            specificMaterial,
            mipBuffer
        );
    }
    else
    {
        LOG(WARNING) << "compute_light_probs module not loaded";
    }
#else
    // Implementation for buffer-based computation
    LOG(INFO) << "Computing triangle probabilities for " << numTriangles << " triangles with material slot " << materialSlot;
    LOG(INFO) << "  GeomInstData ptr: " << &geomInstData[geomInstSlot] 
              << ", Material ptr: " << specificMaterial;
    
    // Call the kernel with the material pointer AND the distribution buffer
    // Note: cudau::Kernel doesn't have isValid(), we check if the module is loaded
    if (computeProbTex_.cudaModule)
    {
        LOG(INFO) << "Launching computeTriangleProbBuffer kernel...";
        
        // Get the distribution's weight buffer where probabilities will be written
        float* probBuffer = dist.weightsOnDevice();
        if (!probBuffer)
        {
            LOG(WARNING) << "Distribution buffer not available";
            return;
        }
        
        computeProbTex_.computeTriangleProbBuffer(
            stream,
            computeProbTex_.computeTriangleProbBuffer.calcGridDim(numTriangles),
            &geomInstData[geomInstSlot],
            numTriangles,
            specificMaterial
        );
        // Sync to see debug output immediately
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        LOG(INFO) << "computeTriangleProbBuffer kernel completed";
    }
    else
    {
        LOG(WARNING) << "compute_light_probs module not loaded";
    }
#endif
}

void ClaudiaAreaLightHandler::computeGeomInstProbabilities(
    CUstream stream,
    uint32_t instSlot,
    uint32_t numGeomInsts,
    LightDistribution& dist)
{
    #if 0
    if (!modelHandler_ || !sceneHandler_)
        return;
    
    // Get the instance data buffer
    auto instDataBuffer = sceneHandler_->getInstanceDataBuffer();
    if (!instDataBuffer)
    {
        LOG(WARNING) << "Instance data buffer not available";
        return;
    }
    
    // Get the geometry instance data buffer
    auto geomInstDataBuffer = modelHandler_->getGeometryInstanceDataBuffer();
    if (!geomInstDataBuffer)
    {
        LOG(WARNING) << "Geometry instance data buffer not available";
        return;
    }
    
    shared::InstanceData* instData = instDataBuffer->getDevicePointerAt(instSlot);
    
#if USE_PROBABILITY_TEXTURE
    // Implementation for texture-based computation
    if (computeProbTex_.cudaModule && computeProbTex_.computeGeomInstProbTexture.isInitialized())
    {
        uint2 dims = shared::computeProbabilityTextureDimentions(numGeomInsts);
        computeProbTex_.computeGeomInstProbTexture(
            stream,
            computeProbTex_.computeGeomInstProbTexture.calcGridDim(dims.x * dims.y),
            instData,
            instSlot,
            numGeomInsts,
            geomInstDataBuffer->getDevicePointer(),
            dist.getSurfaceObject(0)
        );
        LOG(DBUG) << "Computed geometry instance probabilities for " << numGeomInsts << " instances";
    }
#else
    // Implementation for buffer-based computation
    if (computeProbTex_.cudaModule)
    {
        computeProbTex_.computeGeomInstProbBuffer(
            stream,
            computeProbTex_.computeGeomInstProbBuffer.calcGridDim(numGeomInsts),
            instData,
            instSlot,
            numGeomInsts,
            geomInstDataBuffer->getDevicePointer()
        );
        LOG(DBUG) << "Computed geometry instance probabilities for " << numGeomInsts << " instances";
    }
#endif

    #endif
}

void ClaudiaAreaLightHandler::computeInstProbabilities(CUstream stream, uint32_t numInsts)
{
    if (!sceneHandler_)
        return;
    
    #if 0
    // Only log once when distribution is first computed or when size changes
    static uint32_t lastLoggedNumInsts = 0;
    if (numInsts != lastLoggedNumInsts)
    {
        LOG(DBUG) << "Computing instance probabilities for " << numInsts << " instances";
        lastLoggedNumInsts = numInsts;
    }
    
    // Get the instance data buffer
    auto instDataBuffer = sceneHandler_->getInstanceDataBuffer();
    if (!instDataBuffer)
    {
        LOG(WARNING) << "Instance data buffer not available";
        return;
    }
    
#if USE_PROBABILITY_TEXTURE
    // Implementation for texture-based computation
    if (computeProbTex_.cudaModule && computeProbTex_.computeInstProbTexture.isInitialized())
    {
        uint2 dims = shared::computeProbabilityTextureDimentions(numInsts);
        computeProbTex_.computeInstProbTexture(
            stream,
            computeProbTex_.computeInstProbTexture.calcGridDim(dims.x * dims.y),
            &sceneLightDist_,
            numInsts,
            instDataBuffer->getDevicePointer(),
            sceneLightDist_.getSurfaceObject(0)
        );
        
        // Generate mip levels for hierarchical sampling
        uint2 curDims = dims;
        uint32_t numMipLevels = shared::nextPowOf2Exponent(dims.x) + 1;
        for (int dstMipLevel = 1; dstMipLevel < numMipLevels; ++dstMipLevel)
        {
            curDims = (curDims + uint2(1, 1)) / 2;
            computeProbTex_.computeMip(
                stream,
                computeProbTex_.computeMip.calcGridDim(curDims.x, curDims.y),
                &sceneLightDist_,
                dstMipLevel,
                sceneLightDist_.getSurfaceObject(dstMipLevel - 1),
                sceneLightDist_.getSurfaceObject(dstMipLevel)
            );
        }
    }
#else
    // Implementation for buffer-based computation
    if (computeProbTex_.cudaModule)
    {
        // First compute the weights for all instances
        computeProbTex_.computeInstProbBuffer(
            stream,
            computeProbTex_.computeInstProbBuffer.calcGridDim(numInsts),
            &sceneLightDist_,
            numInsts,
            instDataBuffer->getDevicePointer()
        );
        
        // Then compute CDF using exclusive sum
        if (scanScratchMem_.isInitialized())
        {
            CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
                scanScratchMem_.getDevicePointer(), scanScratchSize_,
                sceneLightDist_.weightsOnDevice(),
                sceneLightDist_.cdfOnDevice(),
                numInsts, stream));
            
            // Finally, finalize the distribution
            computeProbTex_.finalizeDiscreteDistribution1D(
                stream,
                computeProbTex_.finalizeDiscreteDistribution1D.calcGridDim(1),
                &sceneLightDist_
            );
        }
    }
#endif

    #endif
}

void ClaudiaAreaLightHandler::finalizeLightDistribution(
    CUstream stream,
    LightDistribution& dist,
    uint32_t numElements)
{
#if USE_PROBABILITY_TEXTURE
    // For probability texture, we generate mip levels
    LOG(DBUG) << "Generating mip levels for probability texture";
    // TODO: Implement kernel invocation when kernels are set
#else
    // For discrete distribution, compute CDF
    LOG(DBUG) << "Finalizing discrete distribution for " << numElements << " elements";
    // TODO: Implement kernel invocation when kernels are set
#endif
}

void ClaudiaAreaLightHandler::finalizeEmitterDistribution(
    CUstream stream,
    ClaudiaModelPtr model, 
    uint32_t geomInstSlot)
{
    if (!model || !modelHandler_)
    {
        LOG(WARNING) << "Model or handler is null in finalizeEmitterDistribution";
        return;
    }
    
    // Cast to triangle model to access triangle-specific methods
    auto triangleModel = std::dynamic_pointer_cast<ClaudiaTriangleModel>(model);
    if (!triangleModel)
    {
        LOG(WARNING) << "Model is not a triangle model";
        return;
    }
    
    // Get the emitter distribution from the model
    LightDistribution* emitterDist = triangleModel->getTriLightImportance();
    if (!emitterDist || !emitterDist->isInitialized())
    {
        LOG(WARNING) << "Emitter distribution not initialized";
        return;
    }
    
    uint32_t numTriangles = triangleModel->getTriangleBuffer().numElements();
    
#if !USE_PROBABILITY_TEXTURE
    // For discrete distribution, we need to compute the CDF using exclusive sum
    // Use the pre-allocated scan scratch memory
    if (scanScratchMem_.isInitialized())
    {
        CUDADRV_CHECK(cubd::DeviceScan::ExclusiveSum(
            scanScratchMem_.getDevicePointer(), scanScratchSize_,
            emitterDist->weightsOnDevice(),
            emitterDist->cdfOnDevice(),
            numTriangles, stream));
        
        LOG(DBUG) << "Computed CDF for " << numTriangles << " triangles";
    }
    else
    {
        LOG(WARNING) << "Scan scratch memory not initialized";
        return;
    }
#endif
    
    // Get the geometry instance data buffer from model handler
    auto geometryInstanceDataBuffer = modelHandler_->getGeometryInstanceDataBuffer();
    if (!geometryInstanceDataBuffer)
    {
        LOG(WARNING) << "Geometry instance data buffer is null";
        return;
    }
    
    // Get device pointer to the specific geometry instance
    shared::GeometryInstanceData* geomInstData = geometryInstanceDataBuffer->getDevicePointerAt(geomInstSlot);
    
    if (computeProbTex_.cudaModule)
    {
        // Finalize the discrete distribution to compute the integral
        computeProbTex_.finalizeDiscreteDistribution1D(
            stream,
            computeProbTex_.finalizeDiscreteDistribution1D.calcGridDim(1),
            &geomInstData->emitterPrimDist
        );
        
        LOG(DBUG) << "Finalized emitter distribution for geometry instance " << geomInstSlot;
    }
    else
    {
        LOG(WARNING) << "finalizeDiscreteDistribution1D kernel not initialized";
    }
}

uint2 ClaudiaAreaLightHandler::computeProbabilityTextureDimensions(uint32_t numElements) const
{
    // Compute texture dimensions based on number of elements
    // Use power-of-two dimensions for efficiency
    uint32_t size = 1;
    while (size * size < numElements)
    {
        size *= 2;
    }
    
    return make_uint2(size, size);
}

void ClaudiaAreaLightHandler::initializeLightKernels()
{
    if (!ctx_)
    {
        LOG(WARNING) << "RenderContext not available for kernel initialization";
        return;
    }
    
    // Get PTXManager from RenderContext
    auto ptxManager = ctx_->getPTXManager();
    if (!ptxManager)
    {
        LOG(WARNING) << "PTXManager not available";
        return;
    }
    
    // Load PTX for compute_light_probs kernels
    std::vector<char> probPtxData = ptxManager->getPTXData("compute_light_probs");
    if (probPtxData.empty())
    {
        LOG(WARNING) << "Failed to load PTX for compute_light_probs";
        return;
    }
    
    // Create null-terminated string for cuModuleLoadData
    probPtxData.push_back('\0');
    
    // Load CUDA module
    CUDADRV_CHECK(cuModuleLoadData(&computeProbTex_.cudaModule, probPtxData.data()));
    
    // Initialize all kernels
    computeProbTex_.computeFirstMip = cudau::Kernel(
        computeProbTex_.cudaModule, "computeProbabilityTextureFirstMip", cudau::dim3(32), 0);
    
    computeProbTex_.computeTriangleProbTexture = cudau::Kernel(
        computeProbTex_.cudaModule, "computeTriangleProbTexture", cudau::dim3(32), 0);
    
    computeProbTex_.computeGeomInstProbTexture = cudau::Kernel(
        computeProbTex_.cudaModule, "computeGeomInstProbTexture", cudau::dim3(32), 0);
    
    computeProbTex_.computeInstProbTexture = cudau::Kernel(
        computeProbTex_.cudaModule, "computeInstProbTexture", cudau::dim3(32), 0);
    
    computeProbTex_.computeMip = cudau::Kernel(
        computeProbTex_.cudaModule, "computeProbabilityTextureMip", cudau::dim3(8, 8), 0);
    
    computeProbTex_.computeTriangleProbBuffer = cudau::Kernel(
        computeProbTex_.cudaModule, "computeTriangleProbBuffer", cudau::dim3(32), 0);
    
    computeProbTex_.computeGeomInstProbBuffer = cudau::Kernel(
        computeProbTex_.cudaModule, "computeGeomInstProbBuffer", cudau::dim3(32), 0);
    
    computeProbTex_.computeInstProbBuffer = cudau::Kernel(
        computeProbTex_.cudaModule, "computeInstProbBuffer", cudau::dim3(32), 0);
    
    computeProbTex_.finalizeDiscreteDistribution1D = cudau::Kernel(
        computeProbTex_.cudaModule, "finalizeDiscreteDistribution1D", cudau::dim3(32), 0);
    
    computeProbTex_.test = cudau::Kernel(
        computeProbTex_.cudaModule, "testProbabilityTexture", cudau::dim3(32), 0);
    
    LOG(INFO) << "Light probability computation kernels initialized successfully";
}

void ClaudiaAreaLightHandler::cleanupLightKernels()
{
    if (computeProbTex_.cudaModule)
    {
        try
        {
            CUDADRV_CHECK(cuModuleUnload(computeProbTex_.cudaModule));
        }
        catch (...)
        {
            LOG(WARNING) << "Failed to unload compute_light_probs module";
        }
        computeProbTex_.cudaModule = nullptr;
    }
}