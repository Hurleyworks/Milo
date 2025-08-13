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
    LOG (INFO) << "ClaudiaEngine created";
    engineName_ = "ClaudiaEngine";
}

ClaudiaEngine::~ClaudiaEngine()
{
    cleanup();
}

void ClaudiaEngine::initialize (RenderContext* ctx)
{
    LOG (INFO) << "ClaudiaEngine::initialize()";

    // Call base class initialization
    // This will set up renderContext_, context_, ptxManager_ and initialize dimensions
    BaseRenderingEngine::initialize (ctx);

    // Allocate device memory for pipeline parameters
    CUDADRV_CHECK (cuMemAlloc (&static_plp_on_device_, sizeof (static_plp_)));
    CUDADRV_CHECK (cuMemAlloc (&per_frame_plp_on_device_, sizeof (per_frame_plp_)));
    CUDADRV_CHECK (cuMemAlloc (&plp_on_device_, sizeof (plp_)));

    // Initialize parameter structures to defaults
    static_plp_ = {};
    per_frame_plp_ = {};

    // Set up the pipeline parameter pointers to point to device memory
    // These are device pointers that will be uploaded to the device
    plp_.s = reinterpret_cast<claudia_shared::StaticPipelineLaunchParameters*> (static_plp_on_device_);
    plp_.f = reinterpret_cast<claudia_shared::PerFramePipelineLaunchParameters*> (per_frame_plp_on_device_);

    if (!isInitialized_)
    {
        LOG (WARNING) << "Base class initialization failed";
        return;
    }

    // Create handlers
    RenderContextPtr renderContext = ctx->shared_from_this();

    // Create material handler first
    materialHandler_ = ClaudiaMaterialHandler::create (renderContext);
    materialHandler_->initialize();

    // Create scene handler and give it the scene
    sceneHandler_ = ClaudiaSceneHandler::create (renderContext);
    sceneHandler_->setScene (&scene_);

    // Create model handler and connect it to other handlers
    modelHandler_ = ClaudiaModelHandler::create (renderContext);
    modelHandler_->initialize();
    modelHandler_->setHandlers (materialHandler_, sceneHandler_);

    // Also give model handler to scene handler
    sceneHandler_->setModelHandler (modelHandler_);

    // Set number of ray types
    constexpr uint32_t numRayTypes = claudia_shared::NumRayTypes;
    // Note: Ray types and material sets are set on geometry instances, not the scene
    // scene_.setNumRayTypes(numRayTypes);
    // scene_.setNumMaterialSets(MATERIAL_SETS);

    LOG (INFO) << "Claudia handlers created and configured with " << numRayTypes << " ray types";

    // Create and setup pipelines
    setupPipelines();

    // Create render handler
    renderHandler_ = ClaudiaRenderHandler::create (renderContext);
    initializeHandlerWithDimensions (renderHandler_, "RenderHandler");

    // Initialize Claudia-specific denoiser handler
    denoiserHandler_ = ClaudiaDenoiserHandler::create (ctx->shared_from_this());
    if (denoiserHandler_ && renderWidth_ > 0 && renderHeight_ > 0)
    {
        if (!denoiserHandler_->initialize (renderWidth_, renderHeight_, true)) // true = use temporal denoiser
        {
            LOG (WARNING) << "Failed to initialize ClaudiaDenoiserHandler";
        }
        else
        {
            // Setup denoiser state after initialization
            denoiserHandler_->setupState (renderContext_->getCudaStream());
            LOG (INFO) << "ClaudiaDenoiserHandler initialized successfully";
        }
    }

    // Allocate launch parameters on device
    allocateLaunchParameters();

    // Initialize RNG buffer with dimensions already set from camera
    if (renderWidth_ > 0 && renderHeight_ > 0)
    {
        rngBuffer_.initialize (renderContext_->getCudaContext(), cudau::BufferType::Device,
                               renderWidth_, renderHeight_);

        // Initialize RNG states
        std::mt19937_64 rng (591842031321323413);
        rngBuffer_.map();
        for (int y = 0; y < renderHeight_; ++y)
        {
            for (int x = 0; x < renderWidth_; ++x)
            {
                rngBuffer_ (x, y).setState (rng());
            }
        }
        rngBuffer_.unmap();

        LOG (INFO) << "RNG buffer initialized for " << renderWidth_ << "x" << renderHeight_;
    }
    else
    {
        LOG (WARNING) << "Invalid render dimensions for RNG buffer initialization";
    }

    gbuffers_.initialize (renderContext_->getCudaContext(), renderWidth_, renderHeight_);
}

#if 1
// GBuffers implementation
void ClaudiaEngine::GBuffers::initialize (CUcontext cuContext, uint32_t width, uint32_t height)
{
    for (int i = 0; i < 2; ++i)
    {
        gBuffer0[i].initialize2D (
            cuContext, cudau::ArrayElementType::UInt32, (sizeof (claudia_shared::GBuffer0Elements) + 3) / 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        gBuffer1[i].initialize2D (
            cuContext, cudau::ArrayElementType::UInt32, (sizeof (claudia_shared::GBuffer1Elements) + 3) / 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
    }
    LOG (INFO) << "G-buffers initialized";
}

void ClaudiaEngine::GBuffers::resize (uint32_t width, uint32_t height)
{
    for (int i = 0; i < 2; ++i)
    {
        gBuffer0[i].resize (width, height);
        gBuffer1[i].resize (width, height);
    }
    LOG (INFO) << "G-buffers resized to " << width << "x" << height;
}

void ClaudiaEngine::GBuffers::finalize()
{
    for (int i = 1; i >= 0; --i)
    {
        gBuffer1[i].finalize();
        gBuffer0[i].finalize();
    }
    LOG (INFO) << "G-buffers finalized";
}

#endif
void ClaudiaEngine::cleanup()
{
    if (!isInitialized_)
    {
        return;
    }

    LOG (INFO) << "ClaudiaEngine::cleanup()";

    //// Clean up device memory
    // if (plpOnDevice_)
    //{
    //     try {
    //         CUDADRV_CHECK(cuMemFree(plpOnDevice_));
    //     } catch (...) {
    //         LOG(WARNING) << "Failed to free plpOnDevice_";
    //     }
    //     plpOnDevice_ = 0;
    // }
    gbuffers_.finalize();

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
        try
        {
            CUDADRV_CHECK (cuModuleUnload (computeProbTex_.cudaModule));
        }
        catch (...)
        {
            LOG (WARNING) << "Failed to unload compute_light_probs module";
        }
        computeProbTex_.cudaModule = nullptr;
    }

    // TODO: Clean up Claudia-specific resources
    // - Temporal buffers
    // - Light sampling structures

    // Call base class cleanup
    BaseRenderingEngine::cleanup();
}

void ClaudiaEngine::addGeometry (sabi::RenderableNode node)
{
    LOG (INFO) << "ClaudiaEngine::addGeometry()";

    if (!node)
    {
        LOG (WARNING) << "Invalid RenderableNode";
        return;
    }

    if (!modelHandler_)
    {
        LOG (WARNING) << "Model handler not initialized";
        return;
    }

    // Create a weak reference to pass to the model handler
    sabi::RenderableWeakRef weakRef = node;

    // Add the model through the model handler
    modelHandler_->addCgModel (weakRef);

    LOG (INFO) << "Added geometry: " << node->getName();

    // Get the model that was just created and update its material hit groups
    // This is necessary because materials are created during addCgModel
    // but pipelines/hit groups are already set up by this point
    ClaudiaModelPtr newModel = modelHandler_->getClaudiaModel (node->getClientID());
    if (newModel)
    {
        updateMaterialHitGroups (newModel);
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
    LOG (INFO) << "ClaudiaEngine::clearScene() - STUB";

    // TODO: Implement Claudia scene clearing
    // - Clear acceleration structures
    // - Reset temporal buffers
    // - Clear light lists
}

void ClaudiaEngine::render (const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber)
{
    if (!isInitialized_)
    {
        LOG (WARNING) << "ClaudiaEngine not initialized";
        return;
    }

    if (!renderContext_)
    {
        LOG (WARNING) << "No render context available";
        return;
    }

    if (!pathTracePipeline_ || !pathTracePipeline_->optixPipeline)
    {
        LOG (WARNING) << "Path tracing pipeline not ready";
        return;
    }

    // Get the CUDA stream from StreamChain for better GPU/CPU overlap
    CUstream stream = streamChain_->waitAvailableAndGetCurrentStream();

    // Get current timer
    auto timer = getCurrentTimer();
    if (!timer)
    {
        LOG (WARNING) << "GPU timer not available for " << getName();
        return;
    }

    // Start frame timer
    timer->frame.start (stream);

    // Update camera if needed
    updateCameraBody (input);

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
        LOG (DBUG) << "Restarting accumulation";
    }

    // Update launch parameters with current state
    updateLaunchParameters (input);

    // Render GBuffer first (for temporal reprojection and other effects)
    renderGBuffer (stream);

    // Synchronize after GBuffer launch to ensure pick info is ready
    CUDADRV_CHECK (cuStreamSynchronize (stream));

    // Output debug info for GBuffer (if enabled)
    // if (enableGBufferDebug_)
    if (true)
    {
        outputGBufferDebugInfo (stream);
    }

    // Launch path tracing kernel - it will handle empty scenes and render the environment
    timer->pathTrace.start (stream);
    {
        // Log launch parameters for debugging
        /* LOG(DBUG) << "Launching OptiX pipeline:"
                   << " dimensions=" << renderWidth_ << "x" << renderHeight_
                   << " numAccumFrames=" << numAccumFrames_
                   << " IAS handle=" << static_plp_.travHandle;*/

        // Launch the path tracing pipeline with exact render dimensions
        try
        {
            pathTracePipeline_->optixPipeline.launch (
                stream,
                plp_on_device_,
                renderWidth_,  // Use exact width, not rounded
                renderHeight_, // Use exact height, not rounded
                1              // depth
            );
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "OptiX launch failed: " << e.what();
            timer->frame.stop (stream);
            return;
        }
    }
    timer->pathTrace.stop (stream);

    // TODO: Run temporal resampling
    // TODO: Run spatial resampling

    // Update accumulation counter
    numAccumFrames_++;

    // Copy accumulation buffers to linear buffers for display
    if (renderHandler_ && renderHandler_->isInitialized())
    {
        // Copy accumulation buffers to linear buffers
        renderHandler_->copyAccumToLinearBuffers (stream);

        // Denoise if denoiser is available
        bool isNewSequence = (numAccumFrames_ == 1);
        renderHandler_->denoise (stream, isNewSequence, denoiserHandler_.get(), timer);
    }

    // Update camera sensor with rendered image
    updateCameraSensor();

    // Stop frame timer
    timer->frame.stop (stream);

    // Increment frame counter and report timings periodically
    frameCounter_++;
    reportTimings (frameCounter_);

    // Switch to next timer buffer for next frame
    switchTimerBuffer();

    // Swap StreamChain buffers for next frame
    streamChain_->swap();
}

void ClaudiaEngine::onEnvironmentChanged()
{
    LOG (INFO) << "ClaudiaEngine::onEnvironmentChanged()";

    // Mark environment as dirty
    environmentDirty_ = true;

    // Reset accumulation since lighting has changed
    numAccumFrames_ = 0;
    restartRender_ = true;

    // TODO: When temporal resampling is implemented:
    // - Invalidate temporal reservoir history
    // - Reset light sampling structures

    LOG (INFO) << "Environment changed - accumulation reset";
}

void ClaudiaEngine::setupPipelines()
{
    LOG (INFO) << "ClaudiaEngine::setupPipelines()";

    if (!renderContext_ || !renderContext_->getOptiXContext())
    {
        LOG (WARNING) << "Context not ready for pipeline setup";
        return;
    }

    optixu::Context optixContext = renderContext_->getOptiXContext();

    // Create default material for the scene (using inherited member from BaseRenderingEngine)
    defaultMaterial_ = optixContext.createMaterial();
    // Note: optixu::Scene doesn't have setMaterialDefault, materials are set per geometry instance

    // Create path tracing pipeline
    pathTracePipeline_ = std::make_shared<engine_core::RenderPipeline<engine_core::PathTracingEntryPoint>>();
    pathTracePipeline_->optixPipeline = optixContext.createPipeline();

    // Calculate max payload size for path tracing
    uint32_t maxPayloadDwords = std::max (
        claudia_shared::SearchRayPayloadSignature::numDwords,
        claudia_shared::VisibilityRayPayloadSignature::numDwords);

    // Configure path tracing pipeline options
    pathTracePipeline_->optixPipeline.setPipelineOptions (
        maxPayloadDwords,
        optixu::calcSumDwords<float2>(), // Attribute dwords for barycentrics
        "claudia_plp",                   // Pipeline launch parameters name - matches CUDA code
        sizeof (claudia_shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    LOG (INFO) << "Claudia path tracing pipeline configured with max payload dwords: " << maxPayloadDwords;

    // Create GBuffer pipeline
    gbufferPipeline_ = std::make_shared<engine_core::RenderPipeline<GBufferEntryPoint>>();
    gbufferPipeline_->optixPipeline = optixContext.createPipeline();

    // Configure GBuffer pipeline options
    // GBuffer needs payload for primary ray information
    uint32_t gbufferPayloadDwords = claudia_shared::PrimaryRayPayloadSignature::numDwords;

    gbufferPipeline_->optixPipeline.setPipelineOptions (
        gbufferPayloadDwords,
        optixu::calcSumDwords<float2>(), // Attribute dwords for barycentrics
        "claudia_plp",                   // Same pipeline launch parameters name
        sizeof (claudia_shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    LOG (INFO) << "Claudia GBuffer pipeline configured with payload dwords: " << gbufferPayloadDwords;

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
    scene_.generateShaderBindingTableLayout (&dummySize);
    LOG (INFO) << "Generated scene SBT layout, size: " << dummySize << " bytes";

    // Create shader binding tables
    createSBT();
}

void ClaudiaEngine::initializeLightProbabilityKernels()
{
    LOG (INFO) << "ClaudiaEngine::initializeLightProbabilityKernels()";

    if (!ptxManager_ || !renderContext_)
    {
        LOG (WARNING) << "PTXManager or RenderContext not ready";
        return;
    }

    // Load PTX for compute_light_probs kernels
    std::vector<char> probPtxData = ptxManager_->getPTXData ("compute_light_probs");
    if (probPtxData.empty())
    {
        LOG (WARNING) << "Failed to load PTX for compute_light_probs";
        return;
    }

    // Create null-terminated string for cuModuleLoadData
    probPtxData.push_back ('\0');

    // Load CUDA module
    CUDADRV_CHECK (cuModuleLoadData (&computeProbTex_.cudaModule, probPtxData.data()));

    // Initialize all kernels
    computeProbTex_.computeFirstMip = cudau::Kernel (
        computeProbTex_.cudaModule, "computeProbabilityTextureFirstMip", cudau::dim3 (32), 0);

    computeProbTex_.computeTriangleProbTexture = cudau::Kernel (
        computeProbTex_.cudaModule, "computeTriangleProbTexture", cudau::dim3 (32), 0);

    computeProbTex_.computeGeomInstProbTexture = cudau::Kernel (
        computeProbTex_.cudaModule, "computeGeomInstProbTexture", cudau::dim3 (32), 0);

    computeProbTex_.computeInstProbTexture = cudau::Kernel (
        computeProbTex_.cudaModule, "computeInstProbTexture", cudau::dim3 (32), 0);

    computeProbTex_.computeMip = cudau::Kernel (
        computeProbTex_.cudaModule, "computeProbabilityTextureMip", cudau::dim3 (8, 8), 0);

    computeProbTex_.computeTriangleProbBuffer = cudau::Kernel (
        computeProbTex_.cudaModule, "computeTriangleProbBuffer", cudau::dim3 (32), 0);

    computeProbTex_.computeGeomInstProbBuffer = cudau::Kernel (
        computeProbTex_.cudaModule, "computeGeomInstProbBuffer", cudau::dim3 (32), 0);

    computeProbTex_.computeInstProbBuffer = cudau::Kernel (
        computeProbTex_.cudaModule, "computeInstProbBuffer", cudau::dim3 (32), 0);

    computeProbTex_.finalizeDiscreteDistribution1D = cudau::Kernel (
        computeProbTex_.cudaModule, "finalizeDiscreteDistribution1D", cudau::dim3 (32), 0);

    computeProbTex_.test = cudau::Kernel (
        computeProbTex_.cudaModule, "testProbabilityTexture", cudau::dim3 (32), 0);

    LOG (INFO) << "Light probability computation kernels initialized successfully";
}

void ClaudiaEngine::createModules()
{
    LOG (INFO) << "ClaudiaEngine::createModules()";

    if (!ptxManager_ || !pathTracePipeline_ || !pathTracePipeline_->optixPipeline ||
        !gbufferPipeline_ || !gbufferPipeline_->optixPipeline)
    {
        LOG (WARNING) << "PTXManager or Pipelines not ready";
        return;
    }

    // Load PTX for Claudia path tracing kernels
    std::vector<char> claudiaPtxData = ptxManager_->getPTXData ("optix_claudia_kernels");
    if (claudiaPtxData.empty())
    {
        LOG (WARNING) << "Failed to load PTX for optix_claudia_kernels";
        return;
    }

    // Create module for path tracing pipeline
    std::string claudiaPtxString (claudiaPtxData.begin(), claudiaPtxData.end());
    pathTracePipeline_->optixModule = pathTracePipeline_->optixPipeline.createModuleFromPTXString (
        claudiaPtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT (OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT (OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    LOG (INFO) << "Claudia path tracing module created successfully";

    // Load PTX for GBuffer kernels
    std::vector<char> gbufferPtxData = ptxManager_->getPTXData ("optix_claudia_gbuffer");
    if (gbufferPtxData.empty())
    {
        LOG (WARNING) << "Failed to load PTX for optix_claudia_gbuffer";
        return;
    }

    // Create module for GBuffer pipeline
    std::string gbufferPtxString (gbufferPtxData.begin(), gbufferPtxData.end());
    gbufferPipeline_->optixModule = gbufferPipeline_->optixPipeline.createModuleFromPTXString (
        gbufferPtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT (OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT (OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    LOG (INFO) << "Claudia GBuffer module created successfully";
}

void ClaudiaEngine::createPrograms()
{
    LOG (INFO) << "ClaudiaEngine::createPrograms()";

    // Check both pipelines
    if (!pathTracePipeline_ || !pathTracePipeline_->optixModule || !pathTracePipeline_->optixPipeline ||
        !gbufferPipeline_ || !gbufferPipeline_->optixModule || !gbufferPipeline_->optixPipeline)
    {
        LOG (WARNING) << "Pipelines not ready";
        return;
    }

    optixu::Module emptyModule; // For empty programs

    // Path Tracing Pipeline Programs
    {
        auto& p = pathTracePipeline_->optixPipeline;
        auto& m = pathTracePipeline_->optixModule;

        // Create ray generation program for path tracing
        pathTracePipeline_->entryPoints[engine_core::PathTracingEntryPoint::PathTrace] =
            p.createRayGenProgram (m, RT_RG_NAME_STR ("pathTracing"));

        // Create miss program for path tracing
        pathTracePipeline_->programs[RT_MS_NAME_STR ("miss")] = p.createMissProgram (
            m, RT_MS_NAME_STR ("miss"));

        // Create hit group for shading
        pathTracePipeline_->hitPrograms[RT_CH_NAME_STR ("shading")] = p.createHitProgramGroupForTriangleIS (
            m, RT_CH_NAME_STR ("shading"),
            emptyModule, nullptr);

        // Create hit group for visibility rays (any hit only)
        pathTracePipeline_->hitPrograms[RT_AH_NAME_STR ("visibility")] = p.createHitProgramGroupForTriangleIS (
            emptyModule, nullptr,
            m, RT_AH_NAME_STR ("visibility"));

        // Create empty miss program for visibility rays
        pathTracePipeline_->programs["emptyMiss"] = p.createMissProgram (emptyModule, nullptr);

        // Set the entry point
        pathTracePipeline_->setEntryPoint (engine_core::PathTracingEntryPoint::PathTrace);

        // Configure miss programs for ray types
        p.setNumMissRayTypes (claudia_shared::NumRayTypes);
        p.setMissProgram (claudia_shared::RayType_Search,
                          pathTracePipeline_->programs.at (RT_MS_NAME_STR ("miss")));
        p.setMissProgram (claudia_shared::RayType_Visibility,
                          pathTracePipeline_->programs.at ("emptyMiss"));

        LOG (INFO) << "Path tracing pipeline programs created";

        // Setup material hit groups for path tracing pipeline on the default material
        if (defaultMaterial_)
        {
            // Set hit group for search rays (shading)
            defaultMaterial_.setHitGroup (claudia_shared::RayType_Search,
                                          pathTracePipeline_->hitPrograms.at (RT_CH_NAME_STR ("shading")));

            // Set hit group for visibility rays
            defaultMaterial_.setHitGroup (claudia_shared::RayType_Visibility,
                                          pathTracePipeline_->hitPrograms.at (RT_AH_NAME_STR ("visibility")));

            LOG (INFO) << "Path tracing material hit groups configured on default material";
        }
    }

    // GBuffer Pipeline Programs
    {
        auto& p = gbufferPipeline_->optixPipeline;
        auto& m = gbufferPipeline_->optixModule;

        // Create ray generation program for GBuffer setup
        gbufferPipeline_->entryPoints[GBufferEntryPoint_SetupGBuffers] =
            p.createRayGenProgram (m, RT_RG_NAME_STR ("setupGBuffers"));

        // Create miss program for GBuffer
        gbufferPipeline_->programs[RT_MS_NAME_STR ("setupGBuffers")] =
            p.createMissProgram (m, RT_MS_NAME_STR ("setupGBuffers"));

        // Create hit group for GBuffer generation
        gbufferPipeline_->hitPrograms[RT_CH_NAME_STR ("setupGBuffers")] =
            p.createHitProgramGroupForTriangleIS (
                m, RT_CH_NAME_STR ("setupGBuffers"),
                emptyModule, nullptr);

        // Create empty hit group for unused ray types
        gbufferPipeline_->hitPrograms["emptyHitGroup"] = p.createEmptyHitProgramGroup();

        // Set the entry point
        gbufferPipeline_->setEntryPoint (GBufferEntryPoint_SetupGBuffers);

        // Configure miss programs for GBuffer ray types
        p.setNumMissRayTypes (claudia_shared::GBufferRayType::NumTypes);
        p.setMissProgram (claudia_shared::GBufferRayType::Primary,
                          gbufferPipeline_->programs.at (RT_MS_NAME_STR ("setupGBuffers")));

        LOG (INFO) << "GBuffer pipeline programs created";
    }

    // Setup material hit groups for GBuffer pipeline on the default material
    if (defaultMaterial_)
    {
        // Set hit group for primary rays
        defaultMaterial_.setHitGroup (claudia_shared::GBufferRayType::Primary,
                                      gbufferPipeline_->hitPrograms.at (RT_CH_NAME_STR ("setupGBuffers")));

        // Set empty hit groups for unused ray types
        for (uint32_t rayType = claudia_shared::GBufferRayType::NumTypes;
             rayType < claudia_shared::maxNumRayTypes; ++rayType)
        {
            defaultMaterial_.setHitGroup (rayType, gbufferPipeline_->hitPrograms.at ("emptyHitGroup"));
        }

        LOG (INFO) << "GBuffer material hit groups configured on default material";
    }
}

void ClaudiaEngine::linkPipelines()
{
    LOG (INFO) << "ClaudiaEngine::linkPipelines()";

    // Link path tracing pipeline with depth 2 (for recursive rays)
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
    {
        // Set the scene on the pipeline
        pathTracePipeline_->optixPipeline.setScene (scene_);

        pathTracePipeline_->optixPipeline.link (2);
        LOG (INFO) << "Path tracing pipeline linked successfully";
    }

    // Link GBuffer pipeline with depth 1 (no recursion needed)
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline)
    {
        // Set the scene on the pipeline
        gbufferPipeline_->optixPipeline.setScene (scene_);

        gbufferPipeline_->optixPipeline.link (1);
        LOG (INFO) << "GBuffer pipeline linked successfully";
    }
}

void ClaudiaEngine::createSBT()
{
    LOG (INFO) << "ClaudiaEngine::createSBT()";

    if (!renderContext_)
    {
        LOG (WARNING) << "No render context for SBT creation";
        return;
    }

    auto cuContext = renderContext_->getCudaContext();

    // Get hit group SBT size from scene
    size_t hitGroupSbtSize = 0;
    scene_.generateShaderBindingTableLayout (&hitGroupSbtSize);
    LOG (INFO) << "Scene hit group SBT size: " << hitGroupSbtSize << " bytes";

    // Create SBT for path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
    {
        auto& p = pathTracePipeline_->optixPipeline;
        size_t sbtSize;
        p.generateShaderBindingTableLayout (&sbtSize);

        LOG (INFO) << "Path tracing pipeline SBT size: " << sbtSize << " bytes";

        if (sbtSize > 0)
        {
            pathTracePipeline_->sbt.initialize (
                cuContext, cudau::BufferType::Device, sbtSize, 1);
            pathTracePipeline_->sbt.setMappedMemoryPersistent (true);
            p.setShaderBindingTable (pathTracePipeline_->sbt, pathTracePipeline_->sbt.getMappedPointer());
        }

        // Set hit group SBT for path tracing pipeline
        if (!pathTracePipeline_->hitGroupSbt.isInitialized())
        {
            if (hitGroupSbtSize > 0)
            {
                pathTracePipeline_->hitGroupSbt.initialize (cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
                pathTracePipeline_->hitGroupSbt.setMappedMemoryPersistent (true);
            }
            else
            {
                // Even with no geometry, initialize a minimal buffer
                pathTracePipeline_->hitGroupSbt.initialize (cuContext, cudau::BufferType::Device, 1, 1);
                pathTracePipeline_->hitGroupSbt.setMappedMemoryPersistent (true);
            }
            pathTracePipeline_->optixPipeline.setHitGroupShaderBindingTable (
                pathTracePipeline_->hitGroupSbt, pathTracePipeline_->hitGroupSbt.getMappedPointer());
        }

        LOG (INFO) << "Path tracing pipeline SBT created";
    }

    // Create SBT for GBuffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline)
    {
        auto& p = gbufferPipeline_->optixPipeline;
        size_t sbtSize;
        p.generateShaderBindingTableLayout (&sbtSize);

        LOG (INFO) << "GBuffer pipeline SBT size: " << sbtSize << " bytes";

        if (sbtSize > 0)
        {
            gbufferPipeline_->sbt.initialize (
                cuContext, cudau::BufferType::Device, sbtSize, 1);
            gbufferPipeline_->sbt.setMappedMemoryPersistent (true);
            p.setShaderBindingTable (gbufferPipeline_->sbt, gbufferPipeline_->sbt.getMappedPointer());
        }

        // Set hit group SBT for GBuffer pipeline
        if (!gbufferPipeline_->hitGroupSbt.isInitialized())
        {
            if (hitGroupSbtSize > 0)
            {
                gbufferPipeline_->hitGroupSbt.initialize (cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
                gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent (true);
            }
            else
            {
                // Even with no geometry, initialize a minimal buffer
                gbufferPipeline_->hitGroupSbt.initialize (cuContext, cudau::BufferType::Device, 1, 1);
                gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent (true);
            }
            gbufferPipeline_->optixPipeline.setHitGroupShaderBindingTable (
                gbufferPipeline_->hitGroupSbt, gbufferPipeline_->hitGroupSbt.getMappedPointer());
        }

        LOG (INFO) << "GBuffer pipeline SBT created";
    }
}

void ClaudiaEngine::updateMaterialHitGroups (ClaudiaModelPtr model)
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

    LOG (DBUG) << "ClaudiaEngine::updateMaterialHitGroups() - updating single model";

    if (!model)
    {
        LOG (WARNING) << "No model provided";
        return;
    }

    auto* triangleModel = dynamic_cast<ClaudiaTriangleModel*> (model.get());
    if (!triangleModel)
    {
        // Not a triangle model - nothing to update
        return;
    }

    auto* geomInst = triangleModel->getGeometryInstance();
    if (!geomInst)
    {
        LOG (WARNING) << "No geometry instance in model";
        return;
    }

    // Update hit groups for all materials in this geometry instance
    uint32_t numMaterials = geomInst->getNumMaterials();
    for (uint32_t i = 0; i < numMaterials; ++i)
    {
        optixu::Material mat = geomInst->getMaterial (0, i); // Material set 0, index i
        if (!mat)
            continue;

        // Set hit groups for path tracing pipeline
        if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
        {
            // Set shading hit group for primary rays (ray type 0)
            auto shadingIt = pathTracePipeline_->hitPrograms.find (RT_CH_NAME_STR ("shading"));
            if (shadingIt != pathTracePipeline_->hitPrograms.end())
            {
                mat.setHitGroup (0, shadingIt->second);
            }

            // Set visibility hit group for shadow rays (ray type 1)
            auto visibilityIt = pathTracePipeline_->hitPrograms.find (RT_AH_NAME_STR ("visibility"));
            if (visibilityIt != pathTracePipeline_->hitPrograms.end())
            {
                mat.setHitGroup (1, visibilityIt->second);
            }
        }

        // Set hit groups for GBuffer pipeline
        if (gbufferPipeline_ && gbufferPipeline_->optixPipeline)
        {
            // Set hit group for primary rays
            auto gbufferIt = gbufferPipeline_->hitPrograms.find (RT_CH_NAME_STR ("setupGBuffers"));
            if (gbufferIt != gbufferPipeline_->hitPrograms.end())
            {
                mat.setHitGroup (claudia_shared::GBufferRayType::Primary, gbufferIt->second);
            }

            // Set empty hit groups for unused ray types
            auto emptyIt = gbufferPipeline_->hitPrograms.find ("emptyHitGroup");
            if (emptyIt != gbufferPipeline_->hitPrograms.end())
            {
                for (uint32_t rayType = claudia_shared::GBufferRayType::NumTypes;
                     rayType < claudia_shared::maxNumRayTypes; ++rayType)
                {
                    mat.setHitGroup (rayType, emptyIt->second);
                }
            }
        }
    }

    LOG (DBUG) << "Updated hit groups for " << numMaterials << " material(s) in model";
}

void ClaudiaEngine::updateSBT()
{
    LOG (DBUG) << "ClaudiaEngine::updateSBT()";

    if (!renderContext_)
    {
        LOG (WARNING) << "No render context for SBT update";
        return;
    }

    auto cuContext = renderContext_->getCudaContext();

    // Get updated hit group SBT size from scene
    size_t hitGroupSbtSize = 0;
    scene_.generateShaderBindingTableLayout (&hitGroupSbtSize);
    LOG (DBUG) << "Updated scene hit group SBT size: " << hitGroupSbtSize << " bytes";

    // Update hit group SBT for path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline && hitGroupSbtSize > 0)
    {
        // Resize if needed
        if (pathTracePipeline_->hitGroupSbt.isInitialized())
        {
            size_t currentSize = pathTracePipeline_->hitGroupSbt.sizeInBytes();
            if (currentSize < hitGroupSbtSize)
            {
                LOG (DBUG) << "Resizing path tracing pipeline hit group SBT from " << currentSize << " to " << hitGroupSbtSize << " bytes";
                pathTracePipeline_->hitGroupSbt.resize (1, hitGroupSbtSize);
            }
        }
        else
        {
            pathTracePipeline_->hitGroupSbt.initialize (cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
            pathTracePipeline_->hitGroupSbt.setMappedMemoryPersistent (true);
        }
        pathTracePipeline_->optixPipeline.setHitGroupShaderBindingTable (
            pathTracePipeline_->hitGroupSbt, pathTracePipeline_->hitGroupSbt.getMappedPointer());
    }

    // Update hit group SBT for GBuffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline && hitGroupSbtSize > 0)
    {
        // Resize if needed
        if (gbufferPipeline_->hitGroupSbt.isInitialized())
        {
            size_t currentSize = gbufferPipeline_->hitGroupSbt.sizeInBytes();
            if (currentSize < hitGroupSbtSize)
            {
                LOG (DBUG) << "Resizing GBuffer pipeline hit group SBT from " << currentSize << " to " << hitGroupSbtSize << " bytes";
                gbufferPipeline_->hitGroupSbt.resize (1, hitGroupSbtSize);
            }
        }
        else
        {
            LOG (DBUG) << "Initializing GBuffer pipeline hit group SBT with size: " << hitGroupSbtSize << " bytes";
            gbufferPipeline_->hitGroupSbt.initialize (cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
            gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent (true);
        }

        // Re-set on pipeline to ensure update is applied
        gbufferPipeline_->optixPipeline.setHitGroupShaderBindingTable (
            gbufferPipeline_->hitGroupSbt, gbufferPipeline_->hitGroupSbt.getMappedPointer());
    }
}

void ClaudiaEngine::renderGBuffer (CUstream stream)
{
    if (!gbufferPipeline_ || !gbufferPipeline_->optixPipeline)
    {
        LOG (WARNING) << "GBuffer pipeline not ready";
        return;
    }

    // Launch the GBuffer generation kernel
    try
    {
        gbufferPipeline_->optixPipeline.launch (
            stream,
            plp_on_device_,
            renderWidth_,
            renderHeight_,
            1 // depth
        );
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "GBuffer OptiX launch failed: " << e.what();
    }
}

void ClaudiaEngine::allocateLaunchParameters()
{
    LOG (INFO) << "ClaudiaEngine::allocateLaunchParameters()";

    if (!renderContext_)
    {
        LOG (WARNING) << "Render context not available";
        return;
    }

    // Note: Launch parameters are already allocated in initialize()
    // This function is kept for compatibility but does nothing
    LOG (INFO) << "Launch parameters already allocated in initialize()";
}

void ClaudiaEngine::updateLaunchParameters (const mace::InputEvent& input)
{
    if (!renderContext_ || !plp_on_device_)
    {
        return;
    }

    // Update STATIC parameters in host buffer
    static_plp_.travHandle = 0; // Default to 0
    if (sceneHandler_)
    {
        // Get IAS handle from scene handler
        static_plp_.travHandle = sceneHandler_->getHandle();
    }

    static_plp_.imageSize = make_int2 (renderWidth_, renderHeight_);
    static_plp_.numAccumFrames = numAccumFrames_;
    static_plp_.bufferIndex = frameCounter_ & 1; // TODO: Temporarily commented to debug crash
    static_plp_.camera = lastCamera_;
    static_plp_.prevCamera = prevCamera_; // For temporal reprojection
    static_plp_.useCameraSpaceNormal = 1;
    static_plp_.bounceLimit = 8; // Maximum path length

    static_plp_.geoBuffer0[0] = gbuffers_.gBuffer0[0].getSurfaceObject (0);
    static_plp_.geoBuffer0[1] = gbuffers_.gBuffer0[1].getSurfaceObject (0);
    static_plp_.geoBuffer1[0] = gbuffers_.gBuffer1[0].getSurfaceObject (0);
    static_plp_.geoBuffer1[1] = gbuffers_.gBuffer1[1].getSurfaceObject (0);

    // Experimental glass parameters (disabled)
    static_plp_.makeAllGlass = 0;
    static_plp_.globalGlassType = 1;
    static_plp_.globalGlassIOR = 1.52f;
    static_plp_.globalTransmittanceDist = 1.0f;

    // Firefly reduction parameter
    static_plp_.maxRadiance = DEFAULT_MAX_RADIANCE; // Default value

    static_plp_.mousePosition = int2 (static_cast<int32_t> (input.getX()), static_cast<int32_t> (input.getY()));

    // Debug: Log mouse position periodically to verify input is working
    static int debugCounter = 0;
    if (debugCounter++ % 60 == 0) // Log every 60 frames (about once per second at 60fps)
    {
        LOG (DBUG) << "Mouse input: (" << input.getX() << ", " << input.getY() << ")";
    }

    // Environment light parameters from property system
    const PropertyService& properties = renderContext_->getPropertyService();
    if (properties.renderProps)
    {
        // Get firefly reduction parameter
        static_plp_.maxRadiance = properties.renderProps->getValOr<float> (RenderKey::MaxRadiance, DEFAULT_MAX_RADIANCE);

        // Check if environment rendering is enabled
        static_plp_.enableEnvLight = properties.renderProps->getValOr<bool> (RenderKey::RenderEnviro, DEFAULT_RENDER_ENVIRO) ? 1 : 0;

        // EnviroIntensity is already a coefficient (0-2 range)
        static_plp_.envLightPowerCoeff = properties.renderProps->getValOr<float> (RenderKey::EnviroIntensity, DEFAULT_ENVIRO_INTENSITY_PERCENT);

        // EnviroRotation is in degrees, convert to radians for the shader
        float envRotationDegrees = properties.renderProps->getValOr<float> (RenderKey::EnviroRotation, DEFAULT_ENVIRO_ROTATION);
        static_plp_.envLightRotation = envRotationDegrees * (M_PI / 180.0f);

        // Use solid background when environment rendering is disabled
        static_plp_.useSolidBackground = properties.renderProps->getValOr<bool> (RenderKey::RenderEnviro, DEFAULT_RENDER_ENVIRO) ? 0 : 1;

        // Background color when not using environment
        Eigen::Vector3d bgColor = properties.renderProps->getValOr<Eigen::Vector3d> (RenderKey::BackgroundColor, DEFAULT_BACKGROUND_COLOR);
        static_plp_.backgroundColor = make_float3 (bgColor.x(), bgColor.y(), bgColor.z());
    }
    else
    {
        // Fallback to defaults if properties not available
        static_plp_.enableEnvLight = DEFAULT_RENDER_ENVIRO ? 1 : 0;
        static_plp_.envLightPowerCoeff = DEFAULT_ENVIRO_INTENSITY_PERCENT;
        static_plp_.envLightRotation = DEFAULT_ENVIRO_ROTATION * (M_PI / 180.0f);
        static_plp_.useSolidBackground = DEFAULT_RENDER_ENVIRO ? 0 : 1;
        static_plp_.backgroundColor = make_float3 (DEFAULT_BACKGROUND_COLOR.x(), DEFAULT_BACKGROUND_COLOR.y(), DEFAULT_BACKGROUND_COLOR.z());
    }

    // Set environment texture if available
    static_plp_.envLightTexture = 0;
    if (renderContext_)
    {
        auto& handlers = renderContext_->getHandlers();
        if (handlers.skyDomeHandler && handlers.skyDomeHandler->hasEnvironmentTexture())
        {
            static_plp_.envLightTexture = handlers.skyDomeHandler->getEnvironmentTexture();

            // Get the environment light importance map
            handlers.skyDomeHandler->getImportanceMap().getDeviceType (&static_plp_.envLightImportanceMap);
        }
        else
        {
            // Initialize with empty distribution
            static_plp_.envLightImportanceMap = shared::RegularConstantContinuousDistribution2D();
        }
    }

    // Set light distribution from scene handler
    if (sceneHandler_)
    {
        // Update emissive instances and build light distribution
        sceneHandler_->updateEmissiveInstances();
        sceneHandler_->buildLightInstanceDistribution();

        // Get the device representation of the light distribution
        sceneHandler_->getLightInstDistribution().getDeviceType (&static_plp_.lightInstDist);
        static_plp_.numLightInsts = sceneHandler_->getNumEmissiveInstances();

        // Read area light settings from properties
        if (properties.renderProps)
        {
            bool enableAreaLights = properties.renderProps->getValOr<bool> (RenderKey::EnableAreaLights, DEFAULT_ENABLE_AREA_LIGHTS);
            static_plp_.enableAreaLights = (enableAreaLights && static_plp_.numLightInsts > 0) ? 1 : 0;
            static_plp_.areaLightPowerCoeff = properties.renderProps->getValOr<float> (RenderKey::AreaLightPower, DEFAULT_AREA_LIGHT_POWER);
        }
        else
        {
            // Fallback to defaults
            static_plp_.enableAreaLights = static_plp_.numLightInsts > 0 ? 1 : 0;
            static_plp_.areaLightPowerCoeff = DEFAULT_AREA_LIGHT_POWER;
        }
    }
    else
    {
        // Set to empty distribution if no scene handler
        static_plp_.lightInstDist = shared::LightDistribution();
        static_plp_.numLightInsts = 0;
        static_plp_.enableAreaLights = 0;
        static_plp_.areaLightPowerCoeff = 1.0f;
    }

    // Set material data buffer from material handler
    if (materialHandler_ && materialHandler_->getMaterialDataBuffer())
    {
        static_plp_.materialDataBuffer = materialHandler_->getMaterialDataBuffer()->getROBuffer<shared::enableBufferOobCheck>();
    }
    else
    {
        // Set to empty buffer if no material handler
        static_plp_.materialDataBuffer = shared::ROBuffer<shared::DisneyData>();
    }

    // Set geometry instance data buffer from model handler
    if (modelHandler_ && modelHandler_->getGeometryInstanceDataBuffer())
    {
        static_plp_.geometryInstanceDataBuffer = modelHandler_->getGeometryInstanceDataBuffer()->getROBuffer<shared::enableBufferOobCheck>();
    }
    else
    {
        // Set to empty buffer if no model handler
        static_plp_.geometryInstanceDataBuffer = shared::ROBuffer<shared::GeometryInstanceData>();
    }

    // Set buffer pointers from RenderHandler
    if (renderHandler_)
    {
        static_plp_.colorAccumBuffer = renderHandler_->getBeautyAccumSurfaceObject();
        static_plp_.albedoAccumBuffer = renderHandler_->getAlbedoAccumSurfaceObject();
        static_plp_.normalAccumBuffer = renderHandler_->getNormalAccumSurfaceObject();
        static_plp_.flowAccumBuffer = renderHandler_->getFlowAccumSurfaceObject();
    }

    // Set RNG buffer
    if (rngBuffer_.isInitialized())
    {
        static_plp_.rngBuffer = rngBuffer_.getBlockBuffer2D();
    }

    // Set pick info buffer pointers
    if (renderHandler_)
    {
        for (int i = 0; i < 2; ++i)
        {
            static_plp_.pickInfoBuffer[i] = reinterpret_cast<claudia_shared::PickInfo*> (
                renderHandler_->getPickInfoPointer (i));
        }
    }

    // Set instance data buffer array
    if (sceneHandler_)
    {
        for (int i = 0; i < 2; ++i)
        {
            auto* buffer = sceneHandler_->getInstanceDataBuffer (i);
            if (buffer && buffer->isInitialized())
            {
                static_plp_.instanceDataBufferArray[i] = buffer->getROBuffer<shared::enableBufferOobCheck>();
            }
        }
    }

    // Copy static parameters to device
    CUDADRV_CHECK (cuMemcpyHtoDAsync (
        static_plp_on_device_,
        &static_plp_,
        sizeof (claudia_shared::StaticPipelineLaunchParameters),
        renderContext_->getCudaStream()));

    // Note: Per-frame parameters are currently empty
    // per_frame_plp_ is reserved for future use

    // Copy the pointer structure to device
    CUDADRV_CHECK (cuMemcpyHtoDAsync (
        plp_on_device_,
        &plp_,
        sizeof (claudia_shared::PipelineLaunchParameters),
        renderContext_->getCudaStream()));
}

void ClaudiaEngine::updateCameraBody (const mace::InputEvent& input)
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
        LOG (WARNING) << "No camera available";
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
            lastCamera_.aspect = static_cast<float> (renderWidth_) / static_cast<float> (renderHeight_);
        }
        lastCamera_.fovY = camera->getVerticalFOVradians();

        // Get camera position
        Eigen::Vector3f eyePoint = camera->getEyePoint();
        lastCamera_.position = Point3D (eyePoint.x(), eyePoint.y(), eyePoint.z());

        // Build camera orientation matrix from camera vectors
        Eigen::Vector3f right = camera->getRight();
        const Eigen::Vector3f& up = camera->getUp();
        const Eigen::Vector3f& forward = camera->getFoward();

        // Fix for standalone applications - negate right vector to correct trackball rotation
        // (not needed in LightWave but required in standalone)
        right *= -1.0f;

        // Convert to shared types
        Vector3D camRight (right.x(), right.y(), right.z());
        Vector3D camUp (up.x(), up.y(), up.z());
        Vector3D camForward (forward.x(), forward.y(), forward.z());

        // Build orientation matrix from camera basis vectors
        // Using the same constructor as production code: Matrix3x3(right, up, forward)
        lastCamera_.orientation = Matrix3x3 (camRight, camUp, camForward);

        // Set lens parameters
        const PropertyService& properties = renderContext_->getPropertyService();
        if (properties.renderProps)
        {
            lastCamera_.lensSize = properties.renderProps->getValOr<float> (RenderKey::Aperture, 0.0f);
            lastCamera_.focusDistance = properties.renderProps->getValOr<float> (RenderKey::FocalLength, 5.0f);
        }
        else
        {
            lastCamera_.lensSize = 0.0f;
            lastCamera_.focusDistance = 5.0f;
        }

        // Mark camera as not dirty after processing
        camera->setDirty (false);
    }
}

void ClaudiaEngine::updateCameraSensor()
{
    // Get camera and check if it has a sensor
    if (!renderContext_)
    {
        LOG (WARNING) << "No render context available for camera sensor update";
        return;
    }

    auto camera = renderContext_->getCamera();
    if (!camera || !camera->getSensor())
    {
        LOG (WARNING) << "No camera or sensor available for update";
        return;
    }

    // Get the linear beauty buffer from our RenderHandler
    if (!renderHandler_ || !renderHandler_->isInitialized())
    {
        LOG (WARNING) << "RenderHandler not available or not initialized";
        return;
    }

    // Use the linear beauty buffer from ClaudiaRenderHandler
    auto& linearBeautyBuffer = renderHandler_->getLinearBeautyBuffer();

    // Since linear buffers are device-only, we need to copy to host
    std::vector<float4> hostPixels (renderWidth_ * renderHeight_);
    linearBeautyBuffer.read (hostPixels.data(), renderWidth_ * renderHeight_);

    // Update the camera sensor with the rendered image
    bool previewMode = false; // Full quality display
    uint32_t renderScale = 1; // No scaling

    Eigen::Vector2i renderSize (renderWidth_, renderHeight_);
    bool success = camera->getSensor()->updateImage (hostPixels.data(), renderSize, previewMode, renderScale);

    if (!success)
    {
        LOG (WARNING) << "Failed to update camera sensor with rendered image";
    }
}

void ClaudiaEngine::outputGBufferDebugInfo (CUstream stream)
{
    if (!renderHandler_)
    {
        LOG (WARNING) << "RenderHandler not available for debug output";
        return;
    }

    // Get the current buffer index
    uint32_t bufferIndex = frameCounter_ & 1;

    // Read pick info - exactly like Shocker code
    claudia_shared::PickInfo pickInfoOnHost;
    renderHandler_->getPickInfo (bufferIndex).read (&pickInfoOnHost, 1, stream);

    // Only output debug information if geometry was hit (not environment/background)
    if (pickInfoOnHost.hit && pickInfoOnHost.matSlot != 0xFFFFFFFF)
    {
        LOG (INFO) << "========== GBuffer Pick Info ==========";
        LOG (INFO) << "Mouse Position: (" << lastInput_.getX() << ", " << lastInput_.getY() << ")";
        LOG (INFO) << "Instance: " << pickInfoOnHost.instSlot;
        LOG (INFO) << "Geometry Instance: " << pickInfoOnHost.geomInstSlot;
        LOG (INFO) << "Primitive Index: " << pickInfoOnHost.primIndex;
        LOG (INFO) << "Material: " << pickInfoOnHost.matSlot;
        LOG (INFO) << "Position: "
                   << pickInfoOnHost.positionInWorld.x << ", "
                   << pickInfoOnHost.positionInWorld.y << ", "
                   << pickInfoOnHost.positionInWorld.z;
        LOG (INFO) << "Normal: "
                   << pickInfoOnHost.normalInWorld.x << ", "
                   << pickInfoOnHost.normalInWorld.y << ", "
                   << pickInfoOnHost.normalInWorld.z;
        LOG (INFO) << "Albedo: "
                   << pickInfoOnHost.albedo.r << ", "
                   << pickInfoOnHost.albedo.g << ", "
                   << pickInfoOnHost.albedo.b;
        LOG (INFO) << "Emittance: "
                   << pickInfoOnHost.emittance.r << ", "
                   << pickInfoOnHost.emittance.g << ", "
                   << pickInfoOnHost.emittance.b;
        LOG (INFO) << "========================================";
    }
}