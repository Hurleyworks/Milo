#include "ShockerEngine.h"
#include "../../RenderContext.h"
#include "handlers/ShockerSceneHandler.h"
#include "handlers/ShockerMaterialHandler.h"
#include "handlers/ShockerModelHandler.h"
#include "handlers/ShockerRenderHandler.h"
#include "handlers/ShockerDenoiserHandler.h"
#include "models/ShockerModel.h"

ShockerEngine::ShockerEngine()
{
    LOG (INFO) << "ShockerEngine created";
    engineName_ = "ShockerEngine";
}

ShockerEngine::~ShockerEngine()
{
    cleanup();
}

void ShockerEngine::initialize (RenderContext* ctx)
{
    LOG (INFO) << "ShockerEngine::initialize()";

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
    plp_.s = reinterpret_cast<shocker_shared::StaticPipelineLaunchParameters*> (static_plp_on_device_);
    plp_.f = reinterpret_cast<shocker_shared::PerFramePipelineLaunchParameters*> (per_frame_plp_on_device_);

    if (!isInitialized_)
    {
        LOG (WARNING) << "Base class initialization failed";
        return;
    }

    // Create handlers
    RenderContextPtr renderContext = ctx->shared_from_this();

    // Create material handler first
    materialHandler_ = ShockerMaterialHandler::create (renderContext);
    materialHandler_->initialize();

    // Create scene handler and give it the scene
    sceneHandler_ = ShockerSceneHandler::create (renderContext);
    sceneHandler_->setScene (&scene_);

    // Create model handler and connect it to other handlers
    modelHandler_ = ShockerModelHandler::create (renderContext);
    modelHandler_->initialize();
    modelHandler_->setHandlers (materialHandler_, sceneHandler_);

    // Also give model handler to scene handler
    sceneHandler_->setModelHandler (modelHandler_);

    // Set number of ray types
    constexpr uint32_t numRayTypes = shocker_shared::maxNumRayTypes;
    // Note: Ray types and material sets are set on geometry instances, not the scene
    // scene_.setNumRayTypes(numRayTypes);
    // scene_.setNumMaterialSets(MATERIAL_SETS);

    LOG (INFO) << "Shocker handlers created and configured with " << numRayTypes << " ray types";

    // Create and setup pipelines
    setupPipelines();

    // Create render handler
    renderHandler_ = ShockerRenderHandler::create (renderContext);
    initializeHandlerWithDimensions (renderHandler_, "RenderHandler");

    // Initialize Shocker-specific denoiser handler
    denoiserHandler_ = ShockerDenoiserHandler::create (ctx->shared_from_this());
    if (denoiserHandler_ && renderWidth_ > 0 && renderHeight_ > 0)
    {
        if (!denoiserHandler_->initialize (renderWidth_, renderHeight_, true)) // true = use temporal denoiser
        {
            LOG (WARNING) << "Failed to initialize ShockerDenoiserHandler";
        }
        else
        {
            // Setup denoiser state after initialization
            denoiserHandler_->setupState (renderContext_->getCudaStream());
            LOG (INFO) << "ShockerDenoiserHandler initialized successfully";
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
void ShockerEngine::GBuffers::initialize (CUcontext cuContext, uint32_t width, uint32_t height)
{
    for (int i = 0; i < 2; ++i)
    {
        gBuffer0[i].initialize2D (
            cuContext, cudau::ArrayElementType::UInt32, (sizeof (shocker_shared::GBuffer0Elements) + 3) / 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        gBuffer1[i].initialize2D (
            cuContext, cudau::ArrayElementType::UInt32, (sizeof (shocker_shared::GBuffer1Elements) + 3) / 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
    }
    LOG (INFO) << "G-buffers initialized";
}

void ShockerEngine::GBuffers::resize (uint32_t width, uint32_t height)
{
    for (int i = 0; i < 2; ++i)
    {
        gBuffer0[i].resize (width, height);
        gBuffer1[i].resize (width, height);
    }
    LOG (INFO) << "G-buffers resized to " << width << "x" << height;
}

void ShockerEngine::GBuffers::finalize()
{
    for (int i = 1; i >= 0; --i)
    {
        gBuffer1[i].finalize();
        gBuffer0[i].finalize();
    }
    LOG (INFO) << "G-buffers finalized";
}

#endif
void ShockerEngine::cleanup()
{
    if (!isInitialized_)
    {
        return;
    }

    LOG (INFO) << "ShockerEngine::cleanup()";

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

    // Clean up Shocker-specific denoiser handler
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

    // TODO: Clean up Shocker-specific resources
    // - Temporal buffers
    // - Light sampling structures

    // Call base class cleanup
    BaseRenderingEngine::cleanup();
}

void ShockerEngine::addGeometry (sabi::RenderableNode node)
{
    LOG (INFO) << "ShockerEngine::addGeometry()";

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
    ShockerModelPtr newModel = modelHandler_->getShockerModel (node->getClientID());
    if (newModel)
    {
        updateMaterialHitGroups (newModel);
    }

    // Update SBT after adding new geometry
    updateSBT();

    // Reset accumulation when scene changes
    restartRender_ = true;

    // TODO: Additional Shocker-specific handling
    // - Update light lists
    // - Invalidate temporal history
}

void ShockerEngine::clearScene()
{
    LOG (INFO) << "ShockerEngine::clearScene() - STUB";

    // TODO: Implement Shocker scene clearing
    // - Clear acceleration structures
    // - Reset temporal buffers
    // - Clear light lists
}

void ShockerEngine::render (const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber)
{
    if (!isInitialized_)
    {
        LOG (WARNING) << "ShockerEngine not initialized";
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

void ShockerEngine::onEnvironmentChanged()
{
    LOG (INFO) << "ShockerEngine::onEnvironmentChanged()";

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

void ShockerEngine::setupPipelines()
{
    LOG (INFO) << "ShockerEngine::setupPipelines()";

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
    uint32_t maxPayloadDwords = std::max ({shocker_shared::PathTraceRayPayloadSignature::numDwords,
                                           shocker_shared::VisibilityRayPayloadSignature::numDwords});

    // Configure path tracing pipeline options
    pathTracePipeline_->optixPipeline.setPipelineOptions (
        maxPayloadDwords,
        optixu::calcSumDwords<float2>(), // Attribute dwords for barycentrics
        "shocker_plp",                   // Pipeline launch parameters name - matches CUDA code
        sizeof (shocker_shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    LOG (INFO) << "Shocker path tracing pipeline configured with max payload dwords: " << maxPayloadDwords;

    // Create GBuffer pipeline
    gbufferPipeline_ = std::make_shared<engine_core::RenderPipeline<GBufferEntryPoint>>();
    gbufferPipeline_->optixPipeline = optixContext.createPipeline();

    // Configure GBuffer pipeline options
    // GBuffer needs payload for primary ray information
    uint32_t gbufferPayloadDwords = shocker_shared::PrimaryRayPayloadSignature::numDwords;

    gbufferPipeline_->optixPipeline.setPipelineOptions (
        gbufferPayloadDwords,
        optixu::calcSumDwords<float2>(), // Attribute dwords for barycentrics
        "shocker_plp",                   // Same pipeline launch parameters name
        sizeof (shocker_shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    LOG (INFO) << "Shocker GBuffer pipeline configured with payload dwords: " << gbufferPayloadDwords;

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

void ShockerEngine::initializeLightProbabilityKernels()
{
    LOG (INFO) << "ShockerEngine::initializeLightProbabilityKernels()";

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

void ShockerEngine::createModules()
{
    LOG (INFO) << "ShockerEngine::createModules()";

    if (!ptxManager_ || !pathTracePipeline_ || !pathTracePipeline_->optixPipeline ||
        !gbufferPipeline_ || !gbufferPipeline_->optixPipeline)
    {
        LOG (WARNING) << "PTXManager or Pipelines not ready";
        return;
    }

    // Load PTX for Shocker path tracing kernels
    std::vector<char> shockerPtxData = ptxManager_->getPTXData ("optix_shocker_kernels");
    if (shockerPtxData.empty())
    {
        LOG (WARNING) << "Failed to load PTX for optix_shocker_kernels";
        return;
    }

    // Create module for path tracing pipeline
    std::string shockerPtxString (shockerPtxData.begin(), shockerPtxData.end());
    pathTracePipeline_->optixModule = pathTracePipeline_->optixPipeline.createModuleFromPTXString (
        shockerPtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT (OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT (OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    LOG (INFO) << "Shocker path tracing module created successfully";

    // Load PTX for GBuffer kernels
    std::vector<char> gbufferPtxData = ptxManager_->getPTXData ("optix_shocker_gbuffer");
    if (gbufferPtxData.empty())
    {
        LOG (WARNING) << "Failed to load PTX for optix_shocker_gbuffer";
        return;
    }

    // Create module for GBuffer pipeline
    std::string gbufferPtxString (gbufferPtxData.begin(), gbufferPtxData.end());
    gbufferPipeline_->optixModule = gbufferPipeline_->optixPipeline.createModuleFromPTXString (
        gbufferPtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT (OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT (OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    LOG (INFO) << "Shocker GBuffer module created successfully";
}

void ShockerEngine::createPrograms()
{
    LOG (INFO) << "ShockerEngine::createPrograms()";

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
            p.createRayGenProgram (m, RT_RG_NAME_STR ("pathTraceBaseline"));

        // Create miss program for path tracing
        pathTracePipeline_->programs[RT_MS_NAME_STR ("pathTraceBaseline")] = p.createMissProgram (
            m, RT_MS_NAME_STR ("pathTraceBaseline"));

        // Create hit group for shading
        pathTracePipeline_->hitPrograms[RT_CH_NAME_STR ("pathTraceBaseline")] = p.createHitProgramGroupForTriangleIS (
            m, RT_CH_NAME_STR ("pathTraceBaseline"),
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
        p.setNumMissRayTypes (shocker_shared::PathTracingRayType::NumTypes);
        p.setMissProgram (shocker_shared::PathTracingRayType::Closest,
                          pathTracePipeline_->programs.at (RT_MS_NAME_STR ("pathTraceBaseline")));
        p.setMissProgram (shocker_shared::PathTracingRayType::Visibility,
                          pathTracePipeline_->programs.at ("emptyMiss"));

        LOG (INFO) << "Path tracing pipeline programs created";

        // Setup material hit groups for path tracing pipeline on the default material
        if (defaultMaterial_)
        {
            // Set hit group for search rays (shading)
            defaultMaterial_.setHitGroup (shocker_shared::PathTracingRayType::Closest,
                                          pathTracePipeline_->hitPrograms.at (RT_CH_NAME_STR ("pathTraceBaseline")));

            // Set hit group for visibility rays
            defaultMaterial_.setHitGroup (shocker_shared::PathTracingRayType::Visibility,
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
        p.setNumMissRayTypes (shocker_shared::GBufferRayType::NumTypes);
        p.setMissProgram (shocker_shared::GBufferRayType::Primary,
                          gbufferPipeline_->programs.at (RT_MS_NAME_STR ("setupGBuffers")));

        LOG (INFO) << "GBuffer pipeline programs created";
    }

    // Setup material hit groups for GBuffer pipeline on the default material
    if (defaultMaterial_)
    {
        // Set hit group for primary rays
        defaultMaterial_.setHitGroup (shocker_shared::GBufferRayType::Primary,
                                      gbufferPipeline_->hitPrograms.at (RT_CH_NAME_STR ("setupGBuffers")));

        // Set empty hit groups for unused ray types
        for (uint32_t rayType = shocker_shared::GBufferRayType::NumTypes;
             rayType < shocker_shared::maxNumRayTypes; ++rayType)
        {
            defaultMaterial_.setHitGroup (rayType, gbufferPipeline_->hitPrograms.at ("emptyHitGroup"));
        }

        LOG (INFO) << "GBuffer material hit groups configured on default material";
    }
}

void ShockerEngine::linkPipelines()
{
    LOG (INFO) << "ShockerEngine::linkPipelines()";

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

void ShockerEngine::createSBT()
{
    LOG (INFO) << "ShockerEngine::createSBT()";

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

void ShockerEngine::updateMaterialHitGroups (ShockerModelPtr model)
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

    LOG (DBUG) << "ShockerEngine::updateMaterialHitGroups() - updating single model";

    if (!model)
    {
        LOG (WARNING) << "No model provided";
        return;
    }

    auto* triangleModel = dynamic_cast<ShockerTriangleModel*> (model.get());
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
            auto shadingIt = pathTracePipeline_->hitPrograms.find (RT_CH_NAME_STR ("pathTraceBaseline"));
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
                mat.setHitGroup (shocker_shared::GBufferRayType::Primary, gbufferIt->second);
            }

            // Set empty hit groups for unused ray types
            auto emptyIt = gbufferPipeline_->hitPrograms.find ("emptyHitGroup");
            if (emptyIt != gbufferPipeline_->hitPrograms.end())
            {
                for (uint32_t rayType = shocker_shared::GBufferRayType::NumTypes;
                     rayType < shocker_shared::maxNumRayTypes; ++rayType)
                {
                    mat.setHitGroup (rayType, emptyIt->second);
                }
            }
        }
    }

    LOG (DBUG) << "Updated hit groups for " << numMaterials << " material(s) in model";
}

void ShockerEngine::updateSBT()
{
    LOG (DBUG) << "ShockerEngine::updateSBT()";

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

void ShockerEngine::renderGBuffer (CUstream stream)
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

void ShockerEngine::allocateLaunchParameters()
{
    LOG (INFO) << "ShockerEngine::allocateLaunchParameters()";

    if (!renderContext_)
    {
        LOG (WARNING) << "Render context not available";
        return;
    }

    // Note: Launch parameters are already allocated in initialize()
    // This function is kept for compatibility but does nothing
    LOG (INFO) << "Launch parameters already allocated in initialize()";
}

void ShockerEngine::updateLaunchParameters (const mace::InputEvent& input)
{
    if (!renderContext_ || !plp_on_device_)
    {
        return;
    }

    // ===== UPDATE STATIC PARAMETERS =====
    // These parameters rarely change during rendering
    
    // Set image size
    static_plp_.imageSize = make_int2 (renderWidth_, renderHeight_);
    
    // Set RNG buffer
    if (rngBuffer_.isInitialized())
    {
        static_plp_.rngBuffer = rngBuffer_.getBlockBuffer2D();
    }
    
    // Set GBuffers
    static_plp_.GBuffer0[0] = gbuffers_.gBuffer0[0].getSurfaceObject (0);
    static_plp_.GBuffer0[1] = gbuffers_.gBuffer0[1].getSurfaceObject (0);
    static_plp_.GBuffer1[0] = gbuffers_.gBuffer1[0].getSurfaceObject (0);
    static_plp_.GBuffer1[1] = gbuffers_.gBuffer1[1].getSurfaceObject (0);
    
    // Set material data buffer from material handler
    if (materialHandler_ && materialHandler_->getMaterialDataBuffer())
    {
        static_plp_.materialDataBuffer = materialHandler_->getMaterialDataBuffer()->getROBuffer<shared::enableBufferOobCheck>();
    }
    else
    {
        static_plp_.materialDataBuffer = shared::ROBuffer<shared::DisneyData>();
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
    
    // Set geometry instance data buffer from model handler
    if (modelHandler_ && modelHandler_->getGeometryInstanceDataBuffer())
    {
        static_plp_.geometryInstanceDataBuffer = modelHandler_->getGeometryInstanceDataBuffer()->getROBuffer<shared::enableBufferOobCheck>();
    }
    else
    {
        static_plp_.geometryInstanceDataBuffer = shared::ROBuffer<shared::GeometryInstanceData>();
    }
    
    // Set light distribution from scene handler
    if (sceneHandler_)
    {
        // Update emissive instances and build light distribution
        sceneHandler_->updateEmissiveInstances();
        sceneHandler_->buildLightInstanceDistribution();
        
        // Get the device representation of the light distribution
        sceneHandler_->getLightInstDistribution().getDeviceType (&static_plp_.lightInstDist);
    }
    else
    {
        static_plp_.lightInstDist = shared::LightDistribution();
    }
    
    // Set environment light importance map and texture
    if (renderContext_)
    {
        auto& handlers = renderContext_->getHandlers();
        if (handlers.skyDomeHandler && handlers.skyDomeHandler->hasEnvironmentTexture())
        {
            static_plp_.envLightTexture = handlers.skyDomeHandler->getEnvironmentTexture();
            handlers.skyDomeHandler->getImportanceMap().getDeviceType (&static_plp_.envLightImportanceMap);
        }
        else
        {
            static_plp_.envLightTexture = 0;
            static_plp_.envLightImportanceMap = shared::RegularConstantContinuousDistribution2D();
        }
    }
    
    // Set accumulation buffer pointers from RenderHandler
    if (renderHandler_)
    {
        static_plp_.beautyAccumBuffer = renderHandler_->getBeautyAccumSurfaceObject();
        static_plp_.albedoAccumBuffer = renderHandler_->getAlbedoAccumSurfaceObject();
        static_plp_.normalAccumBuffer = renderHandler_->getNormalAccumSurfaceObject();
    }
    
    // Set pick info buffer pointers
    if (renderHandler_)
    {
        for (int i = 0; i < 2; ++i)
        {
            static_plp_.pickInfos[i] = reinterpret_cast<shocker_shared::PickInfo*> (
                renderHandler_->getPickInfoPointer (i));
        }
    }
    
    // Set flow accumulation buffer
    if (renderHandler_)
    {
        static_plp_.flowAccumBuffer = renderHandler_->getFlowAccumSurfaceObject();
    }
    
    // Set experimental glass parameters (default values)
    static_plp_.makeAllGlass = 0;
    static_plp_.globalGlassType = 0;
    static_plp_.globalGlassIOR = 1.5f;
    static_plp_.globalTransmittanceDist = 10.0f;
    
    // Set background parameters
    static_plp_.useSolidBackground = 0;
    static_plp_.backgroundColor = make_float3(0.05f, 0.05f, 0.05f);
    
    // Set area light parameters
    if (sceneHandler_)
    {
        static_plp_.numLightInsts = sceneHandler_->getNumEmissiveInstances();
        static_plp_.enableAreaLights = static_plp_.numLightInsts > 0 ? 1 : 0;
    }
    else
    {
        static_plp_.numLightInsts = 0;
        static_plp_.enableAreaLights = 0;
    }
    static_plp_.areaLightPowerCoeff = 1.0f;
    
    // Set firefly reduction parameter
    static_plp_.maxRadiance = 10.0f;

    // ===== UPDATE PER-FRAME PARAMETERS =====
    // These parameters change every frame
    
    // Set traversable handle
    per_frame_plp_.travHandle = 0; // Default to 0
    if (sceneHandler_)
    {
        per_frame_plp_.travHandle = sceneHandler_->getHandle();
    }
    
    // Frame counters
    per_frame_plp_.numAccumFrames = numAccumFrames_;
    per_frame_plp_.frameIndex = frameCounter_;

    //LOG (DBUG) << per_frame_plp_.numAccumFrames;
    
    // Camera parameters
    per_frame_plp_.camera = lastCamera_;
    per_frame_plp_.prevCamera = prevCamera_;
    
    // Environment light parameters from property system
    const PropertyService& properties = renderContext_->getPropertyService();
    if (properties.renderProps)
    {
        // EnviroIntensity is already a coefficient (0-2 range)
        per_frame_plp_.envLightPowerCoeff = properties.renderProps->getValOr<float> (RenderKey::EnviroIntensity, DEFAULT_ENVIRO_INTENSITY_PERCENT);
        
        // EnviroRotation is in degrees, convert to radians for the shader
        float envRotationDegrees = properties.renderProps->getValOr<float> (RenderKey::EnviroRotation, DEFAULT_ENVIRO_ROTATION);
        per_frame_plp_.envLightRotation = envRotationDegrees * (M_PI / 180.0f);
        
        // Check if environment rendering is enabled
        per_frame_plp_.enableEnvLight = properties.renderProps->getValOr<bool> (RenderKey::RenderEnviro, DEFAULT_RENDER_ENVIRO) ? 1 : 0;
    }
    else
    {
        per_frame_plp_.envLightPowerCoeff = DEFAULT_ENVIRO_INTENSITY_PERCENT;
        per_frame_plp_.envLightRotation = DEFAULT_ENVIRO_ROTATION * (M_PI / 180.0f);
        per_frame_plp_.enableEnvLight = DEFAULT_RENDER_ENVIRO ? 1 : 0;
    }
    
    // Mouse position for picking
    per_frame_plp_.mousePosition = int2 (static_cast<int32_t> (input.getX()), static_cast<int32_t> (input.getY()));
    
    // Rendering control flags
    per_frame_plp_.maxPathLength = 4; // Maximum path length for path tracing
    per_frame_plp_.bufferIndex = frameCounter_ & 1; // Alternating buffer index
    per_frame_plp_.resetFlowBuffer = (numAccumFrames_ == 0) ? 1 : 0; // Reset flow on first frame
    per_frame_plp_.enableJittering = 1; // Enable anti-aliasing jitter
    per_frame_plp_.enableBumpMapping = 1; // Enable bump/normal mapping
    per_frame_plp_.enableDebugPrint = 0; // Disable debug output by default
    
    // Debug switches (initially all off)
    per_frame_plp_.debugSwitches = 0;

    // Copy parameters to device
    CUstream stream = renderContext_->getCudaStream();
    
    // Copy static parameters
    CUDADRV_CHECK (cuMemcpyHtoDAsync (
        static_plp_on_device_,
        &static_plp_,
        sizeof (shocker_shared::StaticPipelineLaunchParameters),
        stream));
    
    // Copy per-frame parameters
    CUDADRV_CHECK (cuMemcpyHtoDAsync (
        per_frame_plp_on_device_,
        &per_frame_plp_,
        sizeof (shocker_shared::PerFramePipelineLaunchParameters),
        stream));
    
    // Copy main pipeline parameter structure (just the pointers)
    CUDADRV_CHECK (cuMemcpyHtoDAsync (
        plp_on_device_,
        &plp_,
        sizeof (shocker_shared::PipelineLaunchParameters),
        stream));
}

void ShockerEngine::updateCameraBody (const mace::InputEvent& input)
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

void ShockerEngine::updateCameraSensor()
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

    // Use the linear beauty buffer from ShockerRenderHandler
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

void ShockerEngine::outputGBufferDebugInfo (CUstream stream)
{
    if (!renderHandler_)
    {
        LOG (WARNING) << "RenderHandler not available for debug output";
        return;
    }

    // Get the current buffer index
    uint32_t bufferIndex = frameCounter_ & 1;

    // Read pick info - exactly like Shocker code
    shocker_shared::PickInfo pickInfoOnHost;
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