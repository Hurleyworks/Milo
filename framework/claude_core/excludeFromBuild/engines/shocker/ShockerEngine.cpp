#include "ShockerEngine.h"
#include "../../RenderContext.h"
#include "handlers/ShockerSceneHandler.h"
#include "handlers/ShockerMaterialHandler.h"
#include "handlers/ShockerModelHandler.h"
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
    sceneHandler_->setScene (renderContext->getScene());

    // Create model handler and connect it to other handlers
    modelHandler_ = ShockerModelHandler::create (renderContext);
    modelHandler_->initialize();
    modelHandler_->setHandlers (materialHandler_, sceneHandler_);
    modelHandler_->setEngine (this);

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

    // Initialize SBT for empty scene (must be done after pipeline setup)
    // This ensures the SBT is properly set even when no geometry has been added yet
    auto pipelineHandler = renderContext_->getHandlers().pipelineHandler;
    if (pipelineHandler)
    {
        pipelineHandler->updateSceneSBT();
        LOG (INFO) << "Initial SBT setup for empty scene completed";
    }

    // Initialize ScreenBufferHandler from RenderContext if needed
    auto screenBufferHandler = renderContext->getHandlers().screenBufferHandler;
    if (screenBufferHandler && renderWidth_ > 0 && renderHeight_ > 0)
    {
        if (!screenBufferHandler->isInitialized())
        {
            if (!screenBufferHandler->initialize (renderWidth_, renderHeight_))
            {
                LOG (WARNING) << "Failed to initialize ScreenBufferHandler";
            }
            else
            {
                LOG (INFO) << "ScreenBufferHandler initialized with dimensions "
                           << renderWidth_ << "x" << renderHeight_;
            }
        }
    }

    // Initialize DenoiserHandler after ScreenBufferHandler
    auto denoiserHandler = renderContext->getHandlers().denoiserHandler;
    if (denoiserHandler && renderWidth_ > 0 && renderHeight_ > 0)
    {
        if (!denoiserHandler->isInitialized())
        {
            // Configure denoiser for HDR mode (spatial only, no temporal)
            DenoiserConfig config;
            config.model = DenoiserConfig::Model::HDR; // Basic HDR denoiser
            config.useAlbedo = true;
            config.useNormal = true;
            config.useFlow = false; // HDR model doesn't use motion vectors
            config.alphaMode = OPTIX_DENOISER_ALPHA_MODE_COPY;
            config.useTiling = false; // Can enable for very large framebuffers

            if (!denoiserHandler->initialize (renderWidth_, renderHeight_, config))
            {
                LOG (WARNING) << "Failed to initialize DenoiserHandler";
            }
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

    // Pipeline cleanup is now handled by PipelineHandler in Handlers::cleanup()
    // ScreenBufferHandler cleanup is handled by RenderContext's Handlers::cleanup()

    // Clean up handlers
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

    // Clean up RNG buffer
    if (rngBuffer_.isInitialized())
    {
        rngBuffer_.finalize();
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
    auto pipelineHandler = renderContext_->getHandlers().pipelineHandler;
    if (!pipelineHandler)
    {
        LOG (DBUG) << "PipelineHandler not initialized";
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
    pipelineHandler->updateSceneSBT();

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
    CUstream stream = streamChain_->waitAvailableAndGetCurrentStream();
    auto timer = getCurrentTimer();
    timer->frame.start (stream);

    // Get camera from render context
    auto camera = renderContext_->getCamera();

    // Update camera only if dirty
    if (camera && camera->isDirty())
    {
        updateCameraBody (input);
    }

    // Handle motion updates
    if (updateMotion && sceneHandler_->updateMotion())
    {
        restartRender_ = true;
    }

    // Reset accumulation if needed
    if (restartRender_)
    {
        numAccumFrames_ = 0;
        restartRender_ = false;
    }

    // Update launch parameters (keeping as-is for now)
    updateLaunchParameters (input);

    // Get PipelineHandler from RenderContext
    auto& handlers = renderContext_->getHandlers();
    auto pipelineHandler = handlers.pipelineHandler;

    // Define render sequence
    std::vector<EntryPointType> renderSequence = {
        EntryPointType::GBuffer,
        EntryPointType::PathTrace};

    // Execute the render sequence using PipelineHandler
    pipelineHandler->renderSequence (renderSequence, stream, plp_on_device_,
                                     renderWidth_, renderHeight_);

    // Synchronize for pick info (always required)
    CUDADRV_CHECK (cuStreamSynchronize (stream));

    // Output GBuffer debug info if enabled
    if (enableGBufferDebug_)
    {
        outputGBufferDebugInfo (stream);
    }

    // Increment accumulation counter
    numAccumFrames_++;

    // Post-processing (required every frame)
    if (handlers.screenBufferHandler && handlers.screenBufferHandler->isInitialized())
    {
        handlers.screenBufferHandler->copyAccumToLinearBuffers (stream);

        // Apply denoising if handler is available
        if (handlers.denoiserHandler && handlers.denoiserHandler->isInitialized())
        {
            // Start denoise timer
            timer->denoise.start (stream);

            // Use the simplified denoising interface that handles all buffer setup internally
            bool isFirstFrame = (numAccumFrames_ <= 1);
            float blendFactor = 0.0f; // 0 = full denoise, 1 = full noisy

            handlers.denoiserHandler->denoiseFrame (stream, isFirstFrame, blendFactor);

            // Synchronize to ensure denoising completes
            CUDADRV_CHECK (cuStreamSynchronize (stream));

            // Stop denoise timer
            timer->denoise.stop (stream);
        }
    }

    // Update camera sensor
    updateCameraSensor();

    // Stop frame timer
    timer->frame.stop (stream);

    // Frame accounting
    frameCounter_++;
    reportTimings (frameCounter_);
    switchTimerBuffer();
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

// setupPipelines() using PipelineHandler
void ShockerEngine::setupPipelines()
{
    LOG (INFO) << "ShockerEngine::setupPipelines() - Using PipelineHandler";

    if (!renderContext_ || !renderContext_->isInitialized())
    {
        LOG (WARNING) << "RenderContext not ready for pipeline setup";
        return;
    }

    // Get the PipelineHandler from the RenderContext's handlers
    auto pipelineHandler = renderContext_->getHandlers().pipelineHandler;
    if (!pipelineHandler)
    {
        LOG (WARNING) << "PipelineHandler not available";
        return;
    }

    // Setup path tracing pipeline using PipelineHandler
    PipelineData pathTraceData;
    pathTraceData.entryPoint = EntryPointType::PathTrace;
    pathTraceData.rayGenName = RT_RG_NAME_STR ("pathTraceBaseline");
    pathTraceData.missName = RT_MS_NAME_STR ("pathTraceBaseline");
    pathTraceData.closestHitName = RT_CH_NAME_STR ("pathTraceBaseline");
    pathTraceData.anyHitName = RT_AH_NAME_STR ("visibility");
    pathTraceData.numRayTypes = shocker_shared::PathTracingRayType::NumTypes;
    pathTraceData.searchRayIndex = shocker_shared::PathTracingRayType::Closest;
    pathTraceData.visibilityRayIndex = shocker_shared::PathTracingRayType::Visibility;

    // Configure pipeline options
    pathTraceData.config.maxTraceDepth = 8;
    pathTraceData.config.numPayloadDwords = std::max ({shocker_shared::PathTraceRayPayloadSignature::numDwords,
                                                       shocker_shared::VisibilityRayPayloadSignature::numDwords});
    pathTraceData.config.numAttributeDwords = optixu::calcSumDwords<float2>();
    pathTraceData.config.launchParamsName = "shocker_plp";
    pathTraceData.config.launchParamsSize = sizeof (shocker_shared::PipelineLaunchParameters);
    pathTraceData.config.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pathTraceData.config.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pathTraceData.config.primitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    // Setup path tracing pipeline
    pipelineHandler->setupPipeline (pathTraceData, std::string ("optix_shocker_kernels"));

    // Setup GBuffer pipeline using PipelineHandler
    PipelineData gbufferData;
    gbufferData.entryPoint = EntryPointType::GBuffer;
    gbufferData.rayGenName = RT_RG_NAME_STR ("setupGBuffers");
    gbufferData.missName = RT_MS_NAME_STR ("setupGBuffers");
    gbufferData.closestHitName = RT_CH_NAME_STR ("setupGBuffers");
    gbufferData.numRayTypes = shocker_shared::GBufferRayType::NumTypes;
    gbufferData.searchRayIndex = shocker_shared::GBufferRayType::Primary;

    // Configure GBuffer pipeline options
    gbufferData.config.maxTraceDepth = 2;
    gbufferData.config.numPayloadDwords = shocker_shared::PrimaryRayPayloadSignature::numDwords;
    gbufferData.config.numAttributeDwords = optixu::calcSumDwords<float2>();
    gbufferData.config.launchParamsName = "shocker_plp";
    gbufferData.config.launchParamsSize = sizeof (shocker_shared::PipelineLaunchParameters);
    gbufferData.config.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    gbufferData.config.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    gbufferData.config.primitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    // Setup GBuffer pipeline
    pipelineHandler->setupPipeline (gbufferData, std::string ("optix_shocker_gbuffer"));

    // Ensure scene is set on all pipelines after setup
    if (renderContext_ && renderContext_->getScene())
    {
        pipelineHandler->setScene (renderContext_->getScene());
        LOG (INFO) << "Scene set on PipelineHandler after pipeline setup";
    }

    // Initialize light probability computation kernels (still needed)
    initializeLightProbabilityKernels();

    // SBT setup is handled by PipelineHandler
    LOG (INFO) << "ShockerEngine pipelines setup complete with PipelineHandler";
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

void ShockerEngine::updateMaterialHitGroups (ShockerModelPtr model)
{
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

    // Get the PipelineHandler from render context
    if (!renderContext_)
    {
        LOG (WARNING) << "No render context available for setting hit groups";
        return;
    }

    auto pipelineHandler = renderContext_->getHandlers().pipelineHandler;
    if (!pipelineHandler)
    {
        LOG (WARNING) << "No pipeline handler available for setting hit groups";
        return;
    }

    // Define hit group configuration for Shocker pipelines
    std::map<EntryPointType, std::vector<std::pair<uint32_t, std::string>>> hitGroupConfig;

    // Path tracing pipeline configuration
    hitGroupConfig[EntryPointType::PathTrace] = {
        {shocker_shared::PathTracingRayType::Closest, RT_CH_NAME_STR ("pathTraceBaseline")}, // Ray type 0: shading
        {shocker_shared::PathTracingRayType::Visibility, RT_AH_NAME_STR ("visibility")}      // Ray type 1: shadow rays
    };

    // GBuffer pipeline configuration
    hitGroupConfig[EntryPointType::GBuffer] = {
        {shocker_shared::GBufferRayType::Primary, RT_CH_NAME_STR ("setupGBuffers")} // Ray type 0: primary rays
    };

    // Add empty hit groups for unused ray types in GBuffer (optional)
    for (uint32_t rayType = shocker_shared::GBufferRayType::NumTypes;
         rayType < shocker_shared::maxNumRayTypes; ++rayType)
    {
        hitGroupConfig[EntryPointType::GBuffer].push_back ({rayType, "emptyHitGroup"});
    }

    // Use PipelineHandler's generic method to configure all hit groups
    pipelineHandler->configureMaterialHitGroups (geomInst, hitGroupConfig);

    // Note: updateSceneSBT() is called by addGeometry() after this function returns,
    // so we don't need to call it here - that would cause duplicate SBT updates
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

    // Set accumulation buffer pointers from ScreenBufferHandler
    auto screenBufferHandler = renderContext_->getHandlers().screenBufferHandler;
    if (screenBufferHandler && screenBufferHandler->isInitialized())
    {
        static_plp_.beautyAccumBuffer = screenBufferHandler->getBeautyAccumSurfaceObject();
        static_plp_.albedoAccumBuffer = screenBufferHandler->getAlbedoAccumSurfaceObject();
        static_plp_.normalAccumBuffer = screenBufferHandler->getNormalAccumSurfaceObject();
        static_plp_.flowAccumBuffer = screenBufferHandler->getFlowAccumSurfaceObject();

        // Set pick info buffer pointers
        for (int i = 0; i < 2; ++i)
        {
            static_plp_.pickInfos[i] = reinterpret_cast<shocker_shared::PickInfo*> (
                screenBufferHandler->getPickInfoPointer (i));
        }
    }

    // Set experimental glass parameters (default values)
    static_plp_.makeAllGlass = 0;
    static_plp_.globalGlassType = 0;
    static_plp_.globalGlassIOR = 1.5f;
    static_plp_.globalTransmittanceDist = 10.0f;

    // Set background parameters
    static_plp_.useSolidBackground = 0;
    static_plp_.backgroundColor = make_float3 (0.05f, 0.05f, 0.05f);

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

    // LOG (DBUG) << per_frame_plp_.numAccumFrames;

    // Camera parameters
    per_frame_plp_.camera = lastCamera_;
    per_frame_plp_.prevCamera = prevCamera_;

    // Environment light parameters from property system
    const PropertyService& properties = renderContext_->getPropertyService();
    if (properties.renderProps)
    {
        // EnviroIntensity is already a coefficient (0-2 range)
        per_frame_plp_.envLightPowerCoeff = static_cast<float> (properties.renderProps->getValOr<double> (RenderKey::EnviroIntensity, DEFAULT_ENVIRO_INTENSITY_PERCENT));

        // EnviroRotation is in degrees, convert to radians for the shader
        float envRotationDegrees = static_cast<float> (properties.renderProps->getValOr<double> (RenderKey::EnviroRotation, DEFAULT_ENVIRO_ROTATION));
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
    per_frame_plp_.maxPathLength = 4;                                // Maximum path length for path tracing
    per_frame_plp_.bufferIndex = frameCounter_ & 1;                  // Alternating buffer index
    per_frame_plp_.resetFlowBuffer = (numAccumFrames_ == 0) ? 1 : 0; // Reset flow on first frame
    per_frame_plp_.enableJittering = 1;                              // Enable anti-aliasing jitter
    per_frame_plp_.enableBumpMapping = 1;                            // Enable bump/normal mapping
    per_frame_plp_.enableDebugPrint = 0;                             // Disable debug output by default

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
            lastCamera_.lensSize = static_cast<float> (properties.renderProps->getValOr<double> (RenderKey::Aperture, 0.0));
            lastCamera_.focusDistance = static_cast<float> (properties.renderProps->getValOr<double> (RenderKey::FocalLength, 5.0));
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

    // Get the handlers from RenderContext
    auto& handlers = renderContext_->getHandlers();
    auto screenBufferHandler = handlers.screenBufferHandler;
    if (!screenBufferHandler || !screenBufferHandler->isInitialized())
    {
        LOG (WARNING) << "ScreenBufferHandler not available or not initialized";
        return;
    }

    // Choose which buffer to display: denoised if available, otherwise linear beauty
    cudau::TypedBuffer<float4>* displayBuffer = nullptr;

    if (handlers.denoiserHandler && handlers.denoiserHandler->isInitialized())
    {
        // Use denoised buffer when denoiser is available
        displayBuffer = &screenBufferHandler->getLinearDenoisedBeautyBuffer();
    }
    else
    {
        // Fall back to linear beauty buffer
        displayBuffer = &screenBufferHandler->getLinearBeautyBuffer();
    }

    // Since linear buffers are device-only, we need to copy to host
    std::vector<float4> hostPixels (renderWidth_ * renderHeight_);
    displayBuffer->read (hostPixels.data(), renderWidth_ * renderHeight_);

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
    auto screenBufferHandler = renderContext_->getHandlers().screenBufferHandler;
    if (!screenBufferHandler || !screenBufferHandler->isInitialized())
    {
        LOG (WARNING) << "ScreenBufferHandler not available for debug output";
        return;
    }

    // Get the current buffer index
    uint32_t bufferIndex = frameCounter_ & 1;

    // Read pick info - exactly like Shocker code
    shocker_shared::PickInfo pickInfoOnHost;
    screenBufferHandler->getPickInfoBuffer (bufferIndex).read (&pickInfoOnHost, 1, stream);

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