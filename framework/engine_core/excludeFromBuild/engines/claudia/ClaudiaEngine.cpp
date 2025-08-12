#include "ClaudiaEngine.h"
#include "../../RenderContext.h"
#include "handlers/ClaudiaSceneHandler.h"
#include "handlers/ClaudiaMaterialHandler.h"
#include "handlers/ClaudiaModelHandler.h"
#include "handlers/ClaudiaRenderHandler.h"
#include "handlers/ClaudiaDenoiserHandler.h"
#include "models/ClaudiaModel.h"

ClaudiaEngine::ClaudiaEngine()
{
    LOG(INFO) << "ClaudiaEngine created";
    engineName_ = "ClaudiaEngine";
}

ClaudiaEngine::~ClaudiaEngine()
{
    cleanup();
}

void ClaudiaEngine::initialize(RenderContext* ctx)
{
    LOG(INFO) << "ClaudiaEngine::initialize()";
    
    // Call base class initialization
    // This will set up renderContext_, context_, ptxManager_ and initialize dimensions
    BaseRenderingEngine::initialize(ctx);


    // Allocate device memory for pipeline parameters
    CUDADRV_CHECK (cuMemAlloc (&static_plp_on_device_, sizeof (static_plp_)));
    CUDADRV_CHECK (cuMemAlloc (&per_frame_plp_on_device_, sizeof (per_frame_plp_)));
    CUDADRV_CHECK (cuMemAlloc (&plp_on_device_, sizeof (plp_)));

     // Set up the pipeline parameter pointers
    plp_.s = reinterpret_cast<claudia_shared::StaticPipelineLaunchParameters*> (static_plp_on_device_);
    plp_.f = reinterpret_cast<claudia_shared::PerFramePipelineLaunchParameters*> (per_frame_plp_on_device_);


     // Initialize parameter structures to defaults
    static_plp_ = {};
    per_frame_plp_ = {};

    
    if (!isInitialized_)
    {
        LOG(WARNING) << "Base class initialization failed";
        return;
    }
    
    // Create handlers
    RenderContextPtr renderContext = ctx->shared_from_this();
    
    // Create material handler first
    materialHandler_ = ClaudiaMaterialHandler::create(renderContext);
    materialHandler_->initialize();
    
    // Create scene handler and give it the scene
    sceneHandler_ = ClaudiaSceneHandler::create(renderContext);
    sceneHandler_->setScene(&scene_);
    
    // Create model handler and connect it to other handlers
    modelHandler_ = ClaudiaModelHandler::create(renderContext);
    modelHandler_->initialize();
    modelHandler_->setHandlers(materialHandler_, sceneHandler_);
    
    // Also give model handler to scene handler
    sceneHandler_->setModelHandler(modelHandler_);
    
    // Set number of ray types
    constexpr uint32_t numRayTypes = claudia_shared::NumRayTypes;
    // Note: Ray types and material sets are set on geometry instances, not the scene
    // scene_.setNumRayTypes(numRayTypes);
    // scene_.setNumMaterialSets(MATERIAL_SETS);
    
    LOG(INFO) << "Claudia handlers created and configured with " << numRayTypes << " ray types";
    
    // Create and setup pipelines
    setupPipelines();
    
    // Create render handler
    renderHandler_ = ClaudiaRenderHandler::create(renderContext);
    initializeHandlerWithDimensions(renderHandler_, "RenderHandler");
    
    // Initialize Claudia-specific denoiser handler
    denoiserHandler_ = ClaudiaDenoiserHandler::create(ctx->shared_from_this());
    if (denoiserHandler_ && renderWidth_ > 0 && renderHeight_ > 0)
    {
        if (!denoiserHandler_->initialize(renderWidth_, renderHeight_, true)) // true = use temporal denoiser
        {
            LOG(WARNING) << "Failed to initialize ClaudiaDenoiserHandler";
        }
        else
        {
            // Setup denoiser state after initialization
            denoiserHandler_->setupState(renderContext_->getCudaStream());
            LOG(INFO) << "ClaudiaDenoiserHandler initialized successfully";
        }
    }
    
    // Allocate launch parameters on device
    allocateLaunchParameters();
    
    // Initialize RNG buffer with dimensions already set from camera
    if (renderWidth_ > 0 && renderHeight_ > 0)
    {
        rngBuffer_.initialize(renderContext_->getCudaContext(), cudau::BufferType::Device, 
                              renderWidth_, renderHeight_);
        
        // Initialize RNG states
        std::mt19937_64 rng(591842031321323413);
        rngBuffer_.map();
        for (int y = 0; y < renderHeight_; ++y)
        {
            for (int x = 0; x < renderWidth_; ++x)
            {
                rngBuffer_(x, y).setState(rng());
            }
        }
        rngBuffer_.unmap();
        
        LOG(INFO) << "RNG buffer initialized for " << renderWidth_ << "x" << renderHeight_;
    }
    else
    {
        LOG(WARNING) << "Invalid render dimensions for RNG buffer initialization";
    }
    
    // TODO: Initialize temporal buffers
}

void ClaudiaEngine::cleanup()
{
    if (!isInitialized_)
    {
        return;
    }
    
    LOG(INFO) << "ClaudiaEngine::cleanup()";
    
    //// Clean up device memory
    //if (plpOnDevice_)
    //{
    //    try {
    //        CUDADRV_CHECK(cuMemFree(plpOnDevice_));
    //    } catch (...) {
    //        LOG(WARNING) << "Failed to free plpOnDevice_";
    //    }
    //    plpOnDevice_ = 0;
    //}

    // Free device memory
    if (static_plp_on_device_)
    {
        CUDADRV_CHECK (cuMemFree (static_plp_on_device_));
        static_plp_on_device_ = 0;
    }
    if (per_frame_plp_on_device_)
    {
        CUDADRV_CHECK (cuMemFree (per_frame_plp_on_device_));
        per_frame_plp_on_device_ = 0;
    }
    if (plp_on_device_)
    {
        CUDADRV_CHECK (cuMemFree (plp_on_device_));
        plp_on_device_ = 0;
    }
    
    
    // Clean up pipeline
    
    if (pathTracePipeline_)
    {
        if (pathTracePipeline_->sbt.isInitialized())
            pathTracePipeline_->sbt.finalize();
        
        for (auto& pair : pathTracePipeline_->programs)
            pair.second.destroy();
        
        for (auto& pair : pathTracePipeline_->hitPrograms)
            pair.second.destroy();
        
        for (auto& pair : pathTracePipeline_->entryPoints)
            pair.second.destroy();
        
        if (pathTracePipeline_->optixModule)
            pathTracePipeline_->optixModule.destroy();
        
        if (pathTracePipeline_->optixPipeline)
            pathTracePipeline_->optixPipeline.destroy();
        
        pathTracePipeline_.reset();
    }
    
    // Clean up handlers
    if (renderHandler_)
    {
        renderHandler_->finalize();
        renderHandler_.reset();
    }
    
    if (modelHandler_)
    {
        modelHandler_->finalize();
        modelHandler_.reset();
    }
    
    if (sceneHandler_)
    {
        sceneHandler_->finalize();
        sceneHandler_.reset();
    }
    
    if (materialHandler_)
    {
        materialHandler_->finalize();
        materialHandler_.reset();
    }
    
    // Clean up scene
    if (scene_)
    {
        scene_.destroy();
    }
    
    // Clean up RNG buffer
    if (rngBuffer_.isInitialized())
    {
        rngBuffer_.finalize();
    }
    
    
    // Clean up Claudia-specific denoiser handler
    if (denoiserHandler_)
    {
        denoiserHandler_->finalize();
        denoiserHandler_.reset();
    }
    
    // Clean up light probability computation module
    if (computeProbTex_.cudaModule)
    {
        try {
            CUDADRV_CHECK(cuModuleUnload(computeProbTex_.cudaModule));
        } catch (...) {
            LOG(WARNING) << "Failed to unload compute_light_probs module";
        }
        computeProbTex_.cudaModule = nullptr;
    }
    
    // TODO: Clean up Claudia-specific resources
    // - Temporal buffers
    // - Light sampling structures
    
    // Call base class cleanup
    BaseRenderingEngine::cleanup();
}

void ClaudiaEngine::addGeometry(sabi::RenderableNode node)
{
    LOG(INFO) << "ClaudiaEngine::addGeometry()";
    
    if (!node)
    {
        LOG(WARNING) << "Invalid RenderableNode";
        return;
    }
    
    if (!modelHandler_)
    {
        LOG(WARNING) << "Model handler not initialized";
        return;
    }
    
    // Create a weak reference to pass to the model handler
    sabi::RenderableWeakRef weakRef = node;
    
    // Add the model through the model handler
    modelHandler_->addCgModel(weakRef);
    
    LOG(INFO) << "Added geometry: " << node->getName();
    
    // Get the model that was just created and update its material hit groups
    // This is necessary because materials are created during addCgModel
    // but pipelines/hit groups are already set up by this point
    ClaudiaModelPtr newModel = modelHandler_->getClaudiaModel(node->getClientID());
    if (newModel)
    {
        updateMaterialHitGroups(newModel);
    }
    
    // Update SBT after adding new geometry
    updateSBT();
    
    // Reset accumulation when scene changes
    restartRender_ = true;
    
    // TODO: Additional Claudia-specific handling
    // - Update light lists
    // - Invalidate temporal history
}

void ClaudiaEngine::clearScene()
{
    LOG(INFO) << "ClaudiaEngine::clearScene() - STUB";
    
    // TODO: Implement Claudia scene clearing
    // - Clear acceleration structures
    // - Reset temporal buffers
    // - Clear light lists
}

void ClaudiaEngine::render(const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber)
{
    if (!isInitialized_)
    {
        LOG(WARNING) << "ClaudiaEngine not initialized";
        return;
    }
    
    if (!renderContext_)
    {
        LOG(WARNING) << "No render context available";
        return;
    }
    
    if (!pathTracePipeline_ || !pathTracePipeline_->optixPipeline)
    {
        LOG(WARNING) << "Path tracing pipeline not ready";
        return;
    }
    
    // Get the CUDA stream from StreamChain for better GPU/CPU overlap
    CUstream stream = streamChain_->waitAvailableAndGetCurrentStream();
    
    // Get current timer
    auto timer = getCurrentTimer();
    if (!timer)
    {
        LOG(WARNING) << "GPU timer not available for " << getName();
        return;
    }
    
    // Start frame timer
    timer->frame.start(stream);
    
    // Update camera if needed
    updateCameraBody(input);
    
    // Handle motion updates if requested
    if (updateMotion && sceneHandler_)
    {
        
         bool motionUpdated = sceneHandler_->updateMotion();
         if (motionUpdated) restartRender_ = true;
    }
    
    // Update scene if needed
    if (sceneHandler_)
    {
        // TODO: Update IAS when scene handler supports it
        // sceneHandler_->updateIAS();
    }
    
    // Reset accumulation if needed
    if (restartRender_)
    {
        numAccumFrames_ = 0;
        restartRender_ = false;
        LOG(DBUG) << "Restarting accumulation";
        
    }
    
    // Update launch parameters with current state
    updateLaunchParameters(input);
    
    // Launch path tracing kernel - it will handle empty scenes and render the environment
    timer->pathTrace.start(stream);
    {
        // Log launch parameters for debugging
       /* LOG(DBUG) << "Launching OptiX pipeline:"
                  << " dimensions=" << renderWidth_ << "x" << renderHeight_
                  << " numAccumFrames=" << numAccumFrames_
                  << " IAS handle=" << plp_.s->travHandle;*/
        
        // Launch the path tracing pipeline with exact render dimensions
        try
        {
            pathTracePipeline_->optixPipeline.launch(
                stream,
                plpOnDevice_,
                renderWidth_,   // Use exact width, not rounded
                renderHeight_,  // Use exact height, not rounded
                1  // depth
            );
        }
        catch (const std::exception& e)
        {
            LOG(WARNING) << "OptiX launch failed: " << e.what();
            timer->frame.stop(stream);
            return;
        }
    }
    timer->pathTrace.stop(stream);
    
    // TODO: Run temporal resampling
    // TODO: Run spatial resampling
    
    // Update accumulation counter
    numAccumFrames_++;
    
    
    // Copy accumulation buffers to linear buffers for display
    if (renderHandler_ && renderHandler_->isInitialized())
    {
        // Copy accumulation buffers to linear buffers
        renderHandler_->copyAccumToLinearBuffers(stream);
        
        // Denoise if denoiser is available
        bool isNewSequence = (numAccumFrames_ == 1);
        renderHandler_->denoise(stream, isNewSequence, denoiserHandler_.get(), timer);
    }
    
    // Update camera sensor with rendered image
    updateCameraSensor();
    
    // Stop frame timer
    timer->frame.stop(stream);
    
    // Increment frame counter and report timings periodically
    frameCounter_++;
    reportTimings(frameCounter_);
    
    // Switch to next timer buffer for next frame
    switchTimerBuffer();
    
    // Swap StreamChain buffers for next frame
    streamChain_->swap();
 
}

void ClaudiaEngine::onEnvironmentChanged()
{
    LOG(INFO) << "ClaudiaEngine::onEnvironmentChanged()";
    
    // Mark environment as dirty
    environmentDirty_ = true;
    
    // Reset accumulation since lighting has changed
    numAccumFrames_ = 0;
    restartRender_ = true;
    
    // TODO: When temporal resampling is implemented:
    // - Invalidate temporal reservoir history
    // - Reset light sampling structures
    
    LOG(INFO) << "Environment changed - accumulation reset";
}

void ClaudiaEngine::setupPipelines()
{
    LOG(INFO) << "ClaudiaEngine::setupPipelines()";
    
    if (!renderContext_ || !renderContext_->getOptiXContext())
    {
        LOG(WARNING) << "Context not ready for pipeline setup";
        return;
    }
    
    optixu::Context optixContext = renderContext_->getOptiXContext();
    
    
    // Create path tracing pipeline
    pathTracePipeline_ = std::make_shared<engine_core::RenderPipeline<engine_core::PathTracingEntryPoint>>();
    pathTracePipeline_->optixPipeline = optixContext.createPipeline();
    
    // Calculate max payload size for path tracing
    uint32_t maxPayloadDwords = std::max(
        claudia_shared::SearchRayPayloadSignature::numDwords,
        claudia_shared::VisibilityRayPayloadSignature::numDwords);
    
    // Configure path tracing pipeline options
    pathTracePipeline_->optixPipeline.setPipelineOptions(
        maxPayloadDwords,
        optixu::calcSumDwords<float2>(),  // Attribute dwords for barycentrics
        "claudia_plp",  // Pipeline launch parameters name - matches CUDA code
        sizeof(claudia_shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    
    LOG(INFO) << "Claudia path tracing pipeline configured with max payload dwords: " << maxPayloadDwords;
    
    
    // Ray type configuration is now handled in the pipeline setup
    
    // Create modules
    createModules();
    
    // Initialize light probability computation kernels
    initializeLightProbabilityKernels();
    
    // Create programs
    createPrograms();
    
    // Link pipelines
    linkPipelines();
    
    // Generate SBT layout for the scene first
    // This is required before we can build the pipeline SBTs
    size_t dummySize;
    scene_.generateShaderBindingTableLayout(&dummySize);
    LOG(INFO) << "Generated scene SBT layout, size: " << dummySize << " bytes";
    
    // Create shader binding tables
    createSBT();
}

void ClaudiaEngine::initializeLightProbabilityKernels()
{
    LOG(INFO) << "ClaudiaEngine::initializeLightProbabilityKernels()";
    
    if (!ptxManager_ || !renderContext_)
    {
        LOG(WARNING) << "PTXManager or RenderContext not ready";
        return;
    }
    
    // Load PTX for compute_light_probs kernels
    std::vector<char> probPtxData = ptxManager_->getPTXData("compute_light_probs");
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

void ClaudiaEngine::createModules()
{
    LOG(INFO) << "ClaudiaEngine::createModules()";
    
    if (!ptxManager_ || !pathTracePipeline_ || !pathTracePipeline_->optixPipeline)
    {
        LOG(WARNING) << "PTXManager or Pipelines not ready";
        return;
    }
    
    
    // Load PTX for Claudia path tracing kernels
    std::vector<char> claudiaPtxData = ptxManager_->getPTXData("optix_claudia_kernels");
    if (claudiaPtxData.empty())
    {
        LOG(WARNING) << "Failed to load PTX for optix_claudia_kernels";
        return;
    }
    
    // Create module for path tracing pipeline
    std::string claudiaPtxString(claudiaPtxData.begin(), claudiaPtxData.end());
    pathTracePipeline_->optixModule = pathTracePipeline_->optixPipeline.createModuleFromPTXString(
        claudiaPtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));
    
    LOG(INFO) << "Claudia path tracing module created successfully";
    
}

void ClaudiaEngine::createPrograms()
{
    LOG(INFO) << "ClaudiaEngine::createPrograms()";
    
    // Only check path tracing pipeline since pick pipeline is disabled
    if (!pathTracePipeline_ || !pathTracePipeline_->optixModule || !pathTracePipeline_->optixPipeline)
    {
        LOG(WARNING) << "Path tracing pipeline not ready";
        return;
    }
    
    optixu::Module emptyModule;  // For empty programs
    
    
    // Path Tracing Pipeline Programs
    {
        auto& p = pathTracePipeline_->optixPipeline;
        auto& m = pathTracePipeline_->optixModule;
        
        // Create ray generation program for path tracing
        pathTracePipeline_->entryPoints[engine_core::PathTracingEntryPoint::PathTrace] = 
            p.createRayGenProgram(m, RT_RG_NAME_STR("pathTracing"));
        
        // Create miss program for path tracing
        pathTracePipeline_->programs[RT_MS_NAME_STR("miss")] = p.createMissProgram(
            m, RT_MS_NAME_STR("miss"));
        
        // Create hit group for shading
        pathTracePipeline_->hitPrograms[RT_CH_NAME_STR("shading")] = p.createHitProgramGroupForTriangleIS(
            m, RT_CH_NAME_STR("shading"),
            emptyModule, nullptr);
        
        // Create hit group for visibility rays (any hit only)
        pathTracePipeline_->hitPrograms[RT_AH_NAME_STR("visibility")] = p.createHitProgramGroupForTriangleIS(
            emptyModule, nullptr,
            m, RT_AH_NAME_STR("visibility"));
        
        // Create empty miss program for visibility rays
        pathTracePipeline_->programs["emptyMiss"] = p.createMissProgram(emptyModule, nullptr);
        
        // Set the entry point
        pathTracePipeline_->setEntryPoint(engine_core::PathTracingEntryPoint::PathTrace);
        
        // Configure miss programs for ray types
        p.setNumMissRayTypes(claudia_shared::NumRayTypes);
        p.setMissProgram(claudia_shared::RayType_Search, 
                         pathTracePipeline_->programs.at(RT_MS_NAME_STR("miss")));
        p.setMissProgram(claudia_shared::RayType_Visibility, 
                         pathTracePipeline_->programs.at("emptyMiss"));
        
        LOG(INFO) << "Path tracing pipeline programs created";
    }
    
}

void ClaudiaEngine::linkPipelines()
{
    LOG(INFO) << "ClaudiaEngine::linkPipelines()";
    
    
    // Link path tracing pipeline with depth 2 (for recursive rays)
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
    {
        // Set the scene on the pipeline
        pathTracePipeline_->optixPipeline.setScene(scene_);
        
        pathTracePipeline_->optixPipeline.link(2);
        LOG(INFO) << "Path tracing pipeline linked successfully";
    }
    
}

void ClaudiaEngine::createSBT()
{
    LOG(INFO) << "ClaudiaEngine::createSBT()";
    
    if (!renderContext_)
    {
        LOG(WARNING) << "No render context for SBT creation";
        return;
    }
    
    auto cuContext = renderContext_->getCudaContext();
    
    // Get hit group SBT size from scene
    size_t hitGroupSbtSize = 0;
    scene_.generateShaderBindingTableLayout(&hitGroupSbtSize);
    LOG(INFO) << "Scene hit group SBT size: " << hitGroupSbtSize << " bytes";
    
    
    // Create SBT for path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
    {
        auto& p = pathTracePipeline_->optixPipeline;
        size_t sbtSize;
        p.generateShaderBindingTableLayout(&sbtSize);
        
        LOG(INFO) << "Path tracing pipeline SBT size: " << sbtSize << " bytes";
        
        if (sbtSize > 0)
        {
            pathTracePipeline_->sbt.initialize(
                cuContext, cudau::BufferType::Device, sbtSize, 1);
            pathTracePipeline_->sbt.setMappedMemoryPersistent(true);
            p.setShaderBindingTable(pathTracePipeline_->sbt, pathTracePipeline_->sbt.getMappedPointer());
        }
        
        // Set hit group SBT for path tracing pipeline
        if (!pathTracePipeline_->hitGroupSbt.isInitialized())
        {
            if (hitGroupSbtSize > 0)
            {
                pathTracePipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
                pathTracePipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
            }
            else
            {
                // Even with no geometry, initialize a minimal buffer
                pathTracePipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, 1);
                pathTracePipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
            }
            pathTracePipeline_->optixPipeline.setHitGroupShaderBindingTable(
                pathTracePipeline_->hitGroupSbt, pathTracePipeline_->hitGroupSbt.getMappedPointer());
        }
        
        LOG(INFO) << "Path tracing pipeline SBT created";
    }
}

void ClaudiaEngine::updateMaterialHitGroups(ClaudiaModelPtr model)
{
    // OptiX Hit Groups:
    // A hit group is a collection of programs that execute when a ray intersects geometry.
    // Each hit group can contain up to three programs:
    // 1. Closest Hit (CH) - Executes for the closest intersection along a ray (used for shading)
    // 2. Any Hit (AH) - Executes for any intersection (used for transparency, shadows)
    // 3. Intersection (IS) - Custom intersection test (for non-triangle primitives)
    //
    // Materials in OptiX store references to hit groups for different ray types.
    // For example:
    // - Ray type 0 (primary rays) might use a hit group with complex shading
    // - Ray type 1 (shadow rays) might use a hit group with only an any-hit for shadows
    //
    // This function assigns the appropriate hit groups to each material based on ray type,
    // connecting the material to the actual GPU programs that will execute on ray hits.
    
    LOG(DBUG) << "ClaudiaEngine::updateMaterialHitGroups() - updating single model";
    
    if (!model)
    {
        LOG(WARNING) << "No model provided";
        return;
    }
    
    auto* triangleModel = dynamic_cast<ClaudiaTriangleModel*>(model.get());
    if (!triangleModel)
    {
        // Not a triangle model - nothing to update
        return;
    }
    
    auto* geomInst = triangleModel->getGeometryInstance();
    if (!geomInst)
    {
        LOG(WARNING) << "No geometry instance in model";
        return;
    }
    
    // Update hit groups for all materials in this geometry instance
    uint32_t numMaterials = geomInst->getNumMaterials();
    for (uint32_t i = 0; i < numMaterials; ++i)
    {
        optixu::Material mat = geomInst->getMaterial(0, i);  // Material set 0, index i
        if (!mat)
            continue;
            
        // Set hit groups for path tracing pipeline
        if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
        {
            // Set shading hit group for primary rays (ray type 0)
            auto shadingIt = pathTracePipeline_->hitPrograms.find(RT_CH_NAME_STR("shading"));
            if (shadingIt != pathTracePipeline_->hitPrograms.end())
            {
                mat.setHitGroup(0, shadingIt->second);
            }
            
            // Set visibility hit group for shadow rays (ray type 1)
            auto visibilityIt = pathTracePipeline_->hitPrograms.find(RT_AH_NAME_STR("visibility"));
            if (visibilityIt != pathTracePipeline_->hitPrograms.end())
            {
                mat.setHitGroup(1, visibilityIt->second);
            }
        }
    }
    
    LOG(DBUG) << "Updated hit groups for " << numMaterials << " material(s) in model";
}

void ClaudiaEngine::updateSBT()
{
    LOG(DBUG) << "ClaudiaEngine::updateSBT()";
    
    if (!renderContext_)
    {
        LOG(WARNING) << "No render context for SBT update";
        return;
    }
    
    auto cuContext = renderContext_->getCudaContext();
    
    // Get updated hit group SBT size from scene
    size_t hitGroupSbtSize = 0;
    scene_.generateShaderBindingTableLayout(&hitGroupSbtSize);
    LOG(DBUG) << "Updated scene hit group SBT size: " << hitGroupSbtSize << " bytes";
    
    
    // Update hit group SBT for path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline && hitGroupSbtSize > 0)
    {
        // Resize if needed
        if (pathTracePipeline_->hitGroupSbt.isInitialized())
        {
            size_t currentSize = pathTracePipeline_->hitGroupSbt.sizeInBytes();
            if (currentSize < hitGroupSbtSize)
            {
                LOG(DBUG) << "Resizing path tracing pipeline hit group SBT from " << currentSize << " to " << hitGroupSbtSize << " bytes";
                pathTracePipeline_->hitGroupSbt.resize(1, hitGroupSbtSize);
            }
        }
        else
        {
            pathTracePipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
            pathTracePipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
        }
        pathTracePipeline_->optixPipeline.setHitGroupShaderBindingTable(
            pathTracePipeline_->hitGroupSbt, pathTracePipeline_->hitGroupSbt.getMappedPointer());
    }
}

void ClaudiaEngine::allocateLaunchParameters()
{
    LOG(INFO) << "ClaudiaEngine::allocateLaunchParameters()";
    
    if (!renderContext_)
    {
        LOG(WARNING) << "Render context not available";
        return;
    }
    
    // Allocate path tracing launch parameters
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice_, sizeof(claudia_shared::PipelineLaunchParameters)));
    
    LOG(INFO) << "Launch parameters allocated on device";
}

void ClaudiaEngine::updateLaunchParameters(const mace::InputEvent& input)
{
    if (!renderContext_ || !plpOnDevice_)
    {
        return;
    }
    
    // Update path tracing launch parameters
    plp_.s->travHandle = 0;  // Default to 0
    if (sceneHandler_)
    {
        // Get IAS handle from scene handler
        plp_.s->travHandle = sceneHandler_->getHandle();
    }
    
    plp_.s->imageSize = make_int2(renderWidth_, renderHeight_);
    plp_.s->numAccumFrames = numAccumFrames_;
    plp_.s->bufferIndex = frameCounter_ & 1;  // TODO: Temporarily commented to debug crash
    plp_.s->camera = lastCamera_;
    plp_.s->prevCamera = prevCamera_;  // For temporal reprojection
    plp_.s->useCameraSpaceNormal = 1;
    plp_.s->bounceLimit = 8;  // Maximum path length
    
    // Experimental glass parameters (disabled)
    plp_.s->makeAllGlass = 0;
    plp_.s->globalGlassType = 1;
    plp_.s->globalGlassIOR = 1.52f;
    plp_.s->globalTransmittanceDist = 1.0f;
    
    // Firefly reduction parameter
    plp_.s->maxRadiance = DEFAULT_MAX_RADIANCE;  // Default value
    
    // Environment light parameters from property system
    const PropertyService& properties = renderContext_->getPropertyService();
    if (properties.renderProps)
    {
        // Get firefly reduction parameter
        plp_.s->maxRadiance = properties.renderProps->getValOr<float>(RenderKey::MaxRadiance, DEFAULT_MAX_RADIANCE);
        
        // Check if environment rendering is enabled
        plp_.s->enableEnvLight = properties.renderProps->getValOr<bool>(RenderKey::RenderEnviro, DEFAULT_RENDER_ENVIRO) ? 1 : 0;
        
        // EnviroIntensity is already a coefficient (0-2 range)
        plp_.s->envLightPowerCoeff = properties.renderProps->getValOr<float>(RenderKey::EnviroIntensity, DEFAULT_ENVIRO_INTENSITY_PERCENT);
        
        // EnviroRotation is in degrees, convert to radians for the shader
        float envRotationDegrees = properties.renderProps->getValOr<float>(RenderKey::EnviroRotation, DEFAULT_ENVIRO_ROTATION);
        plp_.s->envLightRotation = envRotationDegrees * (M_PI / 180.0f);
        
        // Use solid background when environment rendering is disabled
        plp_.s->useSolidBackground = properties.renderProps->getValOr<bool>(RenderKey::RenderEnviro, DEFAULT_RENDER_ENVIRO) ? 0 : 1;
        
        // Background color when not using environment
        Eigen::Vector3d bgColor = properties.renderProps->getValOr<Eigen::Vector3d>(RenderKey::BackgroundColor, DEFAULT_BACKGROUND_COLOR);
        plp_.s->backgroundColor = make_float3(bgColor.x(), bgColor.y(), bgColor.z());
    }
    else
    {
        // Fallback to defaults if properties not available
        plp_.s->enableEnvLight = DEFAULT_RENDER_ENVIRO ? 1 : 0;
        plp_.s->envLightPowerCoeff = DEFAULT_ENVIRO_INTENSITY_PERCENT;
        plp_.s->envLightRotation = DEFAULT_ENVIRO_ROTATION * (M_PI / 180.0f);
        plp_.s->useSolidBackground = DEFAULT_RENDER_ENVIRO ? 0 : 1;
        plp_.s->backgroundColor = make_float3(DEFAULT_BACKGROUND_COLOR.x(), DEFAULT_BACKGROUND_COLOR.y(), DEFAULT_BACKGROUND_COLOR.z());
    }
    
    // Set environment texture if available
    plp_.s->envLightTexture = 0;
    if (renderContext_)
    {
        auto& handlers = renderContext_->getHandlers();
        if (handlers.skyDomeHandler && handlers.skyDomeHandler->hasEnvironmentTexture())
        {
            plp_.s->envLightTexture = handlers.skyDomeHandler->getEnvironmentTexture();
            
            // Get the environment light importance map
            handlers.skyDomeHandler->getImportanceMap().getDeviceType(&plp_.s->envLightImportanceMap);
        }
        else
        {
            // Initialize with empty distribution
            plp_.s->envLightImportanceMap = shared::RegularConstantContinuousDistribution2D();
        }
    }
    
    // Set light distribution from scene handler
    if (sceneHandler_)
    {
        // Update emissive instances and build light distribution
        sceneHandler_->updateEmissiveInstances();
        sceneHandler_->buildLightInstanceDistribution();
        
        // Get the device representation of the light distribution
        sceneHandler_->getLightInstDistribution().getDeviceType(&plp_.s->lightInstDist);
        plp_.s->numLightInsts = sceneHandler_->getNumEmissiveInstances();
        
        // Read area light settings from properties
        if (properties.renderProps)
        {
            bool enableAreaLights = properties.renderProps->getValOr<bool>(RenderKey::EnableAreaLights, DEFAULT_ENABLE_AREA_LIGHTS);
            plp_.s->enableAreaLights = (enableAreaLights && plp_.s->numLightInsts > 0) ? 1 : 0;
            plp_.s->areaLightPowerCoeff = properties.renderProps->getValOr<float>(RenderKey::AreaLightPower, DEFAULT_AREA_LIGHT_POWER);
        }
        else
        {
            // Fallback to defaults
            plp_.s->enableAreaLights = plp_.s->numLightInsts > 0 ? 1 : 0;
            plp_.s->areaLightPowerCoeff = DEFAULT_AREA_LIGHT_POWER;
        }
    }
    else
    {
        // Set to empty distribution if no scene handler
        plp_.s->lightInstDist = shared::LightDistribution();
        plp_.s->numLightInsts = 0;
        plp_.s->enableAreaLights = 0;
        plp_.s->areaLightPowerCoeff = 1.0f;
    }
    
    // Set material data buffer from material handler
    if (materialHandler_ && materialHandler_->getMaterialDataBuffer())
    {
        plp_.s->materialDataBuffer = materialHandler_->getMaterialDataBuffer()->getROBuffer<shared::enableBufferOobCheck>();
    }
    else
    {
        // Set to empty buffer if no material handler
        plp_.s->materialDataBuffer = shared::ROBuffer<shared::DisneyData>();
    }
    
    // Set geometry instance data buffer from model handler
    if (modelHandler_ && modelHandler_->getGeometryInstanceDataBuffer())
    {
        plp_.s->geometryInstanceDataBuffer = modelHandler_->getGeometryInstanceDataBuffer()->getROBuffer<shared::enableBufferOobCheck>();
    }
    else
    {
        // Set to empty buffer if no model handler
        plp_.s->geometryInstanceDataBuffer = shared::ROBuffer<shared::GeometryInstanceData>();
    }
    
    // Set buffer pointers from RenderHandler
    if (renderHandler_)
    {
        plp_.s->colorAccumBuffer = renderHandler_->getBeautyAccumSurfaceObject();
        plp_.s->albedoAccumBuffer = renderHandler_->getAlbedoAccumSurfaceObject();
        plp_.s->normalAccumBuffer = renderHandler_->getNormalAccumSurfaceObject();
        plp_.s->flowAccumBuffer = renderHandler_->getFlowAccumSurfaceObject();
    }
    
    // Set RNG buffer
    if (rngBuffer_.isInitialized())
    {
        plp_.s->rngBuffer = rngBuffer_.getBlockBuffer2D();
    }
    
    // Set pick info buffer pointers
    if (renderHandler_)
    {
        for (int i = 0; i < 2; ++i)
        {
            plp_.s->pickInfoBuffer[i] = reinterpret_cast<claudia_shared::PickInfo*>(
                renderHandler_->getPickInfoPointer(i));
        }
    }
    
    // Set instance data buffer array
    if (sceneHandler_)
    {
        for (int i = 0; i < 2; ++i)
        {
            auto* buffer = sceneHandler_->getInstanceDataBuffer(i);
            if (buffer && buffer->isInitialized())
            {
                plp_.s->instanceDataBufferArray[i] = buffer->getROBuffer<shared::enableBufferOobCheck>();
            }
        }
    }
    
    // Upload to device
    CUDADRV_CHECK(cuMemcpyHtoDAsync(
        plpOnDevice_,
        &plp_,
        sizeof(claudia_shared::PipelineLaunchParameters),
        renderContext_->getCudaStream()));
    
}

void ClaudiaEngine::updateCameraBody(const mace::InputEvent& input)
{
    if (!renderContext_)
    {
        return;
    }
    
    // Store last input
    lastInput_ = input;
    
    // Save previous camera for temporal effects
    prevCamera_ = lastCamera_;
    
    // Get camera from render context
    auto camera = renderContext_->getCamera();
    if (!camera)
    {
        LOG(WARNING) << "No camera available";
        return;
    }
    
    // Check if camera has changed
    if (camera->isDirty() || camera->hasSettingsChanged())
    {
        cameraChanged_ = true;
        restartRender_ = true;
        
        // Update camera parameters
        sabi::CameraSensor* sensor = camera->getSensor();
        if (sensor)
        {
            lastCamera_.aspect = sensor->getPixelAspectRatio();
        }
        else
        {
            lastCamera_.aspect = static_cast<float>(renderWidth_) / static_cast<float>(renderHeight_);
        }
        lastCamera_.fovY = camera->getVerticalFOVradians();
        
        // Get camera position
        Eigen::Vector3f eyePoint = camera->getEyePoint();
        lastCamera_.position = Point3D(eyePoint.x(), eyePoint.y(), eyePoint.z());
        
        // Build camera orientation matrix from camera vectors
        Eigen::Vector3f right = camera->getRight();
        const Eigen::Vector3f& up = camera->getUp();
        const Eigen::Vector3f& forward = camera->getFoward();
        
        // Fix for standalone applications - negate right vector to correct trackball rotation
        // (not needed in LightWave but required in standalone)
        right *= -1.0f;
        
        // Convert to shared types
        Vector3D camRight(right.x(), right.y(), right.z());
        Vector3D camUp(up.x(), up.y(), up.z());
        Vector3D camForward(forward.x(), forward.y(), forward.z());
        
        // Build orientation matrix from camera basis vectors
        // Using the same constructor as production code: Matrix3x3(right, up, forward)
        lastCamera_.orientation = Matrix3x3(camRight, camUp, camForward);
        
        // Set lens parameters
        const PropertyService& properties = renderContext_->getPropertyService();
        if (properties.renderProps)
        {
            lastCamera_.lensSize = properties.renderProps->getValOr<float>(RenderKey::Aperture, 0.0f);
            lastCamera_.focusDistance = properties.renderProps->getValOr<float>(RenderKey::FocalLength, 5.0f);
        }
        else
        {
            lastCamera_.lensSize = 0.0f;
            lastCamera_.focusDistance = 5.0f;
        }
        
        // Mark camera as not dirty after processing
        camera->setDirty(false);
    }
}

void ClaudiaEngine::updateCameraSensor()
{
    // Get camera and check if it has a sensor
    if (!renderContext_)
    {
        LOG(WARNING) << "No render context available for camera sensor update";
        return;
    }
    
    auto camera = renderContext_->getCamera();
    if (!camera || !camera->getSensor())
    {
        LOG(WARNING) << "No camera or sensor available for update";
        return;
    }
    
    // Get the linear beauty buffer from our RenderHandler
    if (!renderHandler_ || !renderHandler_->isInitialized())
    {
        LOG(WARNING) << "RenderHandler not available or not initialized";
        return;
    }
    
    // Use the linear beauty buffer from ClaudiaRenderHandler
    auto& linearBeautyBuffer = renderHandler_->getLinearBeautyBuffer();
    
    // Since linear buffers are device-only, we need to copy to host
    std::vector<float4> hostPixels(renderWidth_ * renderHeight_);
    linearBeautyBuffer.read(hostPixels.data(), renderWidth_ * renderHeight_);
    
    // Update the camera sensor with the rendered image
    bool previewMode = false; // Full quality display
    uint32_t renderScale = 1; // No scaling
    
    Eigen::Vector2i renderSize(renderWidth_, renderHeight_);
    bool success = camera->getSensor()->updateImage(hostPixels.data(), renderSize, previewMode, renderScale);
    
    if (!success)
    {
        LOG(WARNING) << "Failed to update camera sensor with rendered image";
    }
}