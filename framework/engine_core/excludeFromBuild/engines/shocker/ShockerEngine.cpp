#include "ShockerEngine.h"
#include "handlers/ShockerSceneHandler.h"
#include "handlers/ShockerMaterialHandler.h"
#include "handlers/ShockerModelHandler.h"
#include "handlers/ShockerRenderHandler.h"
#include "handlers/ShockerDenoiserHandler.h"
#include "handlers/AreaLightHandler.h"
#include "models/ShockerModel.h"
#include "../../tools/PTXManager.h"

ShockerEngine::ShockerEngine()
{
    LOG (DBUG) << "ShockerEngine constructor";
}

ShockerEngine::~ShockerEngine()
{
    LOG (DBUG) << "ShockerEngine destructor";
    cleanup();
}

void ShockerEngine::initialize (RenderContext* ctx)
{
    LOG (DBUG) << "Initializing ShockerEngine";

    // Call base class initialization first
    // This will set up renderContext_, context_, ptxManager_ and initialize dimensions
    BaseRenderingEngine::initialize (ctx);

    if (!isInitialized_)
    {
        LOG (WARNING) << "Base class initialization failed";
        return;
    }

    // Initialize handlers
    auto ctxPtr = std::shared_ptr<RenderContext> (renderContext_, [] (RenderContext*) {});
    sceneHandler_ = ShockerSceneHandler::create (ctxPtr);
    materialHandler_ = std::make_shared<ShockerMaterialHandler>();
    materialHandler_->initialize (renderContext_);
    modelHandler_ = std::make_shared<ShockerModelHandler>();
    modelHandler_->initialize (ctxPtr);
    renderHandler_ = ShockerRenderHandler::create (ctxPtr);
    denoiserHandler_ = ShockerDenoiserHandler::create (ctxPtr);
    areaLightHandler_ = std::make_shared<AreaLightHandler>();
    areaLightHandler_->initialize (renderContext_->getCudaContext(), 100000);
    
    // Pass the scene from BaseRenderingEngine to the scene handler
    sceneHandler_->setScene(&scene_);
    
    // Set up handler dependencies
    modelHandler_->setMaterialHandler(materialHandler_.get());
    modelHandler_->setAreaLightHandler(areaLightHandler_);
    sceneHandler_->setModelHandler(modelHandler_);
    sceneHandler_->setMaterialHandler(materialHandler_);
    sceneHandler_->setAreaLightHandler(areaLightHandler_);

    if (!sceneHandler_ || !materialHandler_ || !modelHandler_ || !renderHandler_)
    {
        LOG (WARNING) << "Failed to create handlers";
        return;
    }

    // Initialize render handler with dimensions from base class
    initializeHandlerWithDimensions (renderHandler_, "RenderHandler");

    // Initialize denoiser if available
    if (initializeHandlerWithDimensions (denoiserHandler_, "DenoiserHandler"))
    {
        // Setup denoiser state after initialization
        denoiserHandler_->setupState (renderContext_->getCudaStream());
        LOG (INFO) << "ShockerDenoiserHandler state setup completed";
    }

    // Setup pipelines
    setupPipelines();

    // Initialize buffers first (before allocating launch parameters)
    if (renderWidth_ > 0 && renderHeight_ > 0 && renderContext_)
    {
        CUcontext cuContext = renderContext_->getCudaContext();

        // Initialize RNG buffer
        rngBuffer_.initialize (cuContext, cudau::BufferType::Device, renderWidth_, renderHeight_);
        uint64_t seed = 12345;
        rngBuffer_.map();
        for (int y = 0; y < renderHeight_; ++y)
        {
            for (int x = 0; x < renderWidth_; ++x)
            {
                shared::PCG32RNG& rng = rngBuffer_ (x, y);
                rng.setState (seed + (y * renderWidth_ + x) * 1234567);
            }
        }
        rngBuffer_.unmap();

        // Initialize G-buffers (double buffered, matching sample code pattern)
        for (int i = 0; i < 2; ++i)
        {
            gBuffer0_[i].initialize2D (
                cuContext, cudau::ArrayElementType::UInt32,
                (sizeof (shocker_shared::GBuffer0Elements) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderWidth_, renderHeight_, 1);

            gBuffer1_[i].initialize2D (
                cuContext, cudau::ArrayElementType::UInt32,
                (sizeof (shocker_shared::GBuffer1Elements) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderWidth_, renderHeight_, 1);
        }
        LOG (DBUG) << "G-buffers initialized";

        // Initialize pick info buffers (double buffered for temporal stability)
        shocker_shared::PickInfo initPickInfo = {};
        initPickInfo.hit = false;
        initPickInfo.instSlot = 0xFFFFFFFF;
        initPickInfo.geomInstSlot = 0xFFFFFFFF;
        initPickInfo.matSlot = 0xFFFFFFFF;
        initPickInfo.primIndex = 0xFFFFFFFF;
        initPickInfo.positionInWorld = Point3D (0.0f);
        initPickInfo.normalInWorld = Normal3D (0.0f);
        initPickInfo.albedo = RGB (0.0f);
        initPickInfo.emittance = RGB (0.0f);

        for (int i = 0; i < 2; ++i)
        {
            pickInfoBuffers_[i].initialize (cuContext, cudau::BufferType::Device, 1, initPickInfo);
        }
        LOG (DBUG) << "Pick info buffers initialized";

        // Now populate static launch parameters with buffer pointers
        // IMPORTANT: Initialize ALL fields to avoid accessing uninitialized memory
        memset (&staticPlp_, 0, sizeof (staticPlp_)); // Zero initialize everything first

        staticPlp_.imageSize = make_int2 (renderWidth_, renderHeight_);
        staticPlp_.rngBuffer = rngBuffer_.getBlockBuffer2D();

        // Setup G-buffers
        for (int i = 0; i < 2; ++i)
        {
            staticPlp_.GBuffer0[i] = gBuffer0_[i].getSurfaceObject (0);
            staticPlp_.GBuffer1[i] = gBuffer1_[i].getSurfaceObject (0);
        }

        // Setup pick info buffers
        staticPlp_.pickInfos[0] = pickInfoBuffers_[0].getDevicePointer();
        staticPlp_.pickInfos[1] = pickInfoBuffers_[1].getDevicePointer();

        // Initialize material and instance buffers to empty (will be set when models are loaded)
        staticPlp_.disneyMaterialBuffer = shared::ROBuffer<shared::DisneyData>();
        staticPlp_.instanceDataBufferArray[0] = shared::ROBuffer<shocker::ShockerNodeData>();
        staticPlp_.instanceDataBufferArray[1] = shared::ROBuffer<shocker::ShockerNodeData>();
        
        // Set geometry instance data buffer if model handler has already initialized it
        if (modelHandler_) {
            auto* buffer = modelHandler_->getGeometryInstanceDataBuffer();
            if (buffer && buffer->isInitialized()) {
                staticPlp_.geometryInstanceDataBuffer = buffer->getROBuffer<shared::enableBufferOobCheck>();
                LOG(INFO) << "Set geometry instance data buffer in static launch parameters";
            } else {
                staticPlp_.geometryInstanceDataBuffer = shared::ROBuffer<shocker::ShockerSurfaceData>();
            }
        } else {
            staticPlp_.geometryInstanceDataBuffer = shared::ROBuffer<shocker::ShockerSurfaceData>();
        }

        // Initialize light distribution to empty
        staticPlp_.lightInstDist = shared::LightDistribution();

        // Initialize environment map structures
        staticPlp_.envLightImportanceMap = shared::RegularConstantContinuousDistribution2D();
        staticPlp_.envLightTexture = 0; // No texture initially

        // Update accumulation buffer references from render handler if available
        if (renderHandler_)
        {
            staticPlp_.beautyAccumBuffer = renderHandler_->getBeautyAccumSurfaceObject();
            staticPlp_.albedoAccumBuffer = renderHandler_->getAlbedoAccumSurfaceObject();
            staticPlp_.normalAccumBuffer = renderHandler_->getNormalAccumSurfaceObject();
            staticPlp_.motionAccumBuffer = renderHandler_->getMotionAccumSurfaceObject();
        }
    }

    // Now allocate and upload launch parameters with populated data
    allocateLaunchParameters();

    // Initialize camera
    lastCamera_.aspect = static_cast<float> (renderWidth_) / renderHeight_;
    lastCamera_.fovY = 45.0f * pi_v<float> / 180.0f;
    lastCamera_.position = Point3D (0, 0, 5);
    lastCamera_.orientation.c0 = Vector3D (1, 0, 0);
    lastCamera_.orientation.c1 = Vector3D (0, 1, 0);
    lastCamera_.orientation.c2 = Vector3D (0, 0, 1);
    prevCamera_ = lastCamera_;

    isInitialized_ = true;
    LOG (DBUG) << "ShockerEngine initialized successfully";
}

void ShockerEngine::cleanup()
{
    if (!isInitialized_)
    {
        return;
    }

    LOG (DBUG) << "Cleaning up ShockerEngine";
    
    // Clean up default material
    if (defaultMaterial_) {
        defaultMaterial_.destroy();
    }

    // Clean up pipelines
    if (gbufferPipeline_)
    {
        gbufferPipeline_->destroy();
        gbufferPipeline_.reset();
    }

    if (pathTracePipeline_)
    {
        pathTracePipeline_->destroy();
        pathTracePipeline_.reset();
    }

    // Clean up launch parameters (matching sample code)
    if (plpOnDevice_)
    {
        CUDADRV_CHECK (cuMemFree (plpOnDevice_));
        plpOnDevice_ = 0;
    }
    if (staticPlpOnDevice_)
    {
        CUDADRV_CHECK (cuMemFree (staticPlpOnDevice_));
        staticPlpOnDevice_ = 0;
    }
    if (perFramePlpOnDevice_)
    {
        CUDADRV_CHECK (cuMemFree (perFramePlpOnDevice_));
        perFramePlpOnDevice_ = 0;
    }

    // Clean up RNG buffer
    if (rngBuffer_.isInitialized())
    {
        rngBuffer_.finalize();
    }

    // Clean up G-buffers
    for (int i = 1; i >= 0; --i)
    {
        if (gBuffer1_[i].isInitialized())
        {
            gBuffer1_[i].finalize();
        }
        if (gBuffer0_[i].isInitialized())
        {
            gBuffer0_[i].finalize();
        }
    }

    // Clean up pick info buffers
    for (int i = 1; i >= 0; --i)
    {
        if (pickInfoBuffers_[i].isInitialized())
        {
            pickInfoBuffers_[i].finalize();
        }
    }

    // Clean up handlers
    // Note: Scene is destroyed in BaseRenderingEngine::cleanup()
    if (denoiserHandler_)
    {
        denoiserHandler_->finalize();
        denoiserHandler_.reset();
    }

    if (renderHandler_)
    {
        renderHandler_->finalize();
        renderHandler_.reset();
    }

    if (areaLightHandler_)
    {
        areaLightHandler_.reset();
    }

    sceneHandler_.reset();
    materialHandler_.reset();
    modelHandler_.reset();

    // Call base class cleanup to destroy scene and reset state
    BaseRenderingEngine::cleanup();
    
    LOG (DBUG) << "ShockerEngine cleanup complete";
}

void ShockerEngine::addGeometry (sabi::RenderableNode node)
{
    if (!isInitialized_)
    {
        LOG (WARNING) << "ShockerEngine not initialized";
        return;
    }

    LOG (DBUG) << "Adding geometry to ShockerEngine: " << node->getName();

    // Process the node through the scene handler
    // This will create the ShockerModel and ShockerNode
    if (sceneHandler_)
    {
        sceneHandler_->processRenderableNode(node);
        
        // Update geometry instance data buffer with surface data from model handler
        if (modelHandler_) {
            modelHandler_->updateGeometryInstanceDataBuffer();
        }
        
        // Build/update acceleration structures after adding geometry
        sceneHandler_->buildAccelerationStructures();
        
        // Update SBT to include the new geometry
        updateSBT();
        
        
        // Update traversable handle from scene
        perFramePlp_.travHandle = sceneHandler_->getTraversableHandle();
        
        // Mark that we need to restart accumulation
        restartRender_ = true;
        perFramePlp_.numAccumFrames = 0;
        
        LOG (INFO) << "Added geometry: " << node->getName() 
                   << " (Total nodes: " << sceneHandler_->getNodeCount() 
                   << ", traversable: " << perFramePlp_.travHandle << ")";
    }
    else
    {
        LOG (WARNING) << "Scene handler not initialized";
    }
}

void ShockerEngine::clearScene()
{
    if (!isInitialized_)
    {
        return;
    }

    LOG (DBUG) << "Clearing ShockerEngine scene";

    // Clear all nodes and models through scene handler
    if (sceneHandler_)
    {
        sceneHandler_->clear();
    }

    // Reset scene traversable handle  
    perFramePlp_.travHandle = 0;  // Will be 0 after clearing
    
    // Update SBT after clearing the scene
    updateSBT();

    // Mark that we need to restart accumulation
    restartRender_ = true;
    perFramePlp_.numAccumFrames = 0;
    
    LOG (INFO) << "Scene cleared";
}

void ShockerEngine::resize (uint32_t width, uint32_t height)
{
    if (!isInitialized_ || width == 0 || height == 0)
    {
        return;
    }

    LOG (DBUG) << "Resizing ShockerEngine from " << renderWidth_ << "x" << renderHeight_
               << " to " << width << "x" << height;

    renderWidth_ = width;
    renderHeight_ = height;

    CUcontext cuContext = renderContext_->getCudaContext();

    // Resize RNG buffer
    if (rngBuffer_.isInitialized())
    {
        rngBuffer_.finalize();
    }
    rngBuffer_.initialize (cuContext, cudau::BufferType::Device, width, height);
    uint64_t seed = 12345;
    rngBuffer_.map();
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            shared::PCG32RNG& rng = rngBuffer_ (x, y);
            rng.setState (seed + (y * width + x) * 1234567);
        }
    }
    rngBuffer_.unmap();

    // Resize G-buffers
    for (int i = 0; i < 2; ++i)
    {
        if (gBuffer0_[i].isInitialized())
        {
            gBuffer0_[i].resize (width, height);
        }
        else
        {
            gBuffer0_[i].initialize2D (
                cuContext, cudau::ArrayElementType::UInt32,
                (sizeof (shocker_shared::GBuffer0Elements) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
        }

        if (gBuffer1_[i].isInitialized())
        {
            gBuffer1_[i].resize (width, height);
        }
        else
        {
            gBuffer1_[i].initialize2D (
                cuContext, cudau::ArrayElementType::UInt32,
                (sizeof (shocker_shared::GBuffer1Elements) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
        }
    }

    // Resize render handler
    if (renderHandler_)
    {
        renderHandler_->resize (width, height);
    }

    // Resize denoiser handler
    if (denoiserHandler_)
    {
        denoiserHandler_->resize (width, height);
        // Setup state after resize
        denoiserHandler_->setupState (renderContext_->getCudaStream());
    }

    // Update camera aspect ratio
    lastCamera_.aspect = static_cast<float> (width) / height;

    // Mark for render restart
    restartRender_ = true;
    LOG (DBUG) << "ShockerEngine resize complete";
}

void ShockerEngine::render (const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber)
{
    if (!isInitialized_)
    {
        return;
    }

    // Get the CUDA stream from StreamChain for better GPU/CPU overlap
    CUstream stream = streamChain_->waitAvailableAndGetCurrentStream();
    
    // Get current timer
    auto timer = getCurrentTimer();
    if (!timer)
    {
        LOG(WARNING) << "GPU timer not available for " << getName();
        // Continue without timing
    }
    
    // Start frame timer
    if (timer)
    {
        timer->frame.start(stream);
    }

    // Ensure launch parameters are initialized on first render
    static bool firstRender = true;
    if (firstRender)
    {
        // Force initial upload of all launch parameters
        updateLaunchParameters (input);
        restartRender_ = true; // Force static parameter upload
        firstRender = false;
    }

    // Update camera if needed

    auto camera = renderContext_->getCamera();
    if (camera->isDirty())
    {
        updateCameraBody (input);

        lastInput_ = input;
    }

    // Update launch parameters
    updateLaunchParameters (input);

    // Start path trace timer (includes both G-buffer and path tracing)
    if (timer)
    {
        timer->pathTrace.start(stream);
    }
    
    renderGBuffer(stream);
    
    // Synchronize to ensure G-buffer is complete before path tracing
    CUDADRV_CHECK(cuStreamSynchronize(stream));
    
    renderPathTracing(stream);
    
    // Stop path trace timer
    if (timer)
    {
        timer->pathTrace.stop(stream);
    }

    updateCameraSensor();
    
    // Stop frame timer
    if (timer)
    {
        timer->frame.stop(stream);
    }

    // Update frame counter
    frameCounter_++;
    
    // Report timings periodically
    reportTimings(frameCounter_);
    
    // Switch to next timer buffer for next frame
    switchTimerBuffer();
    
    // Swap StreamChain buffers for next frame
    streamChain_->swap();

    // Update motion state
    if (updateMotion)
    {
        prevCamera_ = lastCamera_;
    }
}

void ShockerEngine::onEnvironmentChanged()
{
    LOG (INFO) << "ShockerEngine::onEnvironmentChanged()";

    // Mark environment as dirty
    environmentDirty_ = true;

    // Reset accumulation since lighting has changed
    restartRender_ = true;
    perFramePlp_.numAccumFrames = 0;

    // Update environment texture immediately
    if (renderContext_)
    {
        auto& handlers = renderContext_->getHandlers();
        if (handlers.skyDomeHandler && handlers.skyDomeHandler->hasEnvironmentTexture())
        {
            staticPlp_.envLightTexture = handlers.skyDomeHandler->getEnvironmentTexture();

            // Get the environment light importance map
            handlers.skyDomeHandler->getImportanceMap().getDeviceType (&staticPlp_.envLightImportanceMap);

            LOG (INFO) << "Environment texture updated: " << std::hex << staticPlp_.envLightTexture << std::dec;
        }
        else
        {
            // Clear environment if no texture
            staticPlp_.envLightTexture = 0;
            staticPlp_.envLightImportanceMap = shared::RegularConstantContinuousDistribution2D();
            LOG (INFO) << "Environment texture cleared";
        }

        // Upload the updated static parameters to device immediately
        if (staticPlpOnDevice_)
        {
            CUcontext prevContext;
            CUDADRV_CHECK (cuCtxGetCurrent (&prevContext));
            CUDADRV_CHECK (cuCtxSetCurrent (renderContext_->getCudaContext()));

            CUDADRV_CHECK (cuMemcpyHtoD (staticPlpOnDevice_, &staticPlp_, sizeof (staticPlp_)));

            // Also update the main plp structure
            plp_.s = reinterpret_cast<shocker_shared::StaticPipelineLaunchParameters*> (staticPlpOnDevice_);
            CUDADRV_CHECK (cuMemcpyHtoD (plpOnDevice_, &plp_, sizeof (plp_)));

            CUDADRV_CHECK (cuCtxSetCurrent (prevContext));

            LOG (INFO) << "Environment parameters uploaded to device";
        }
    }

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

    // Create G-buffer pipeline
    gbufferPipeline_ = std::make_shared<engine_core::RenderPipeline<GBufferEntryPoint>>();
    gbufferPipeline_->optixPipeline = optixContext.createPipeline();

    // Configure G-buffer pipeline options
    gbufferPipeline_->optixPipeline.setPipelineOptions (
        std::max ({shocker_shared::PrimaryRayPayloadSignature::numDwords}), // Payload dwords for G-buffer
        optixu::calcSumDwords<float2>(),
        "plp", // Pipeline launch parameters name
        sizeof (shocker_shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    LOG (INFO) << "Shocker G-buffer pipeline configured";

    // Create path tracing pipeline
    pathTracePipeline_ = std::make_shared<engine_core::RenderPipeline<PathTracingEntryPoint>>();
    pathTracePipeline_->optixPipeline = optixContext.createPipeline();

    // Configure path tracing pipeline options
    pathTracePipeline_->optixPipeline.setPipelineOptions (
        std::max ({shocker_shared::PathTraceRayPayloadSignature::numDwords,
                   shocker_shared::VisibilityRayPayloadSignature::numDwords}), // Payload dwords for path tracing
        optixu::calcSumDwords<float2>(),                                       // Attribute dwords for barycentrics
        "plp",                                                                 // Pipeline launch parameters name
        sizeof (shocker_shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    LOG (INFO) << "Shocker path tracing pipeline configured";

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

void ShockerEngine::createModules()
{
    LOG (INFO) << "ShockerEngine::createModules()";

    if (!ptxManager_ || !gbufferPipeline_ || !gbufferPipeline_->optixPipeline ||
        !pathTracePipeline_ || !pathTracePipeline_->optixPipeline)
    {
        LOG (WARNING) << "PTXManager or Pipelines not ready";
        return;
    }

    // Load PTX for G-buffer kernels
    std::vector<char> gbufferPtxData = ptxManager_->getPTXData ("optix_shocker_gbuffer");
    if (gbufferPtxData.empty())
    {
        LOG (WARNING) << "Failed to load PTX for optix_shocker_gbuffer";
        return;
    }

    // Create module for G-buffer pipeline
    std::string gbufferPtxString (gbufferPtxData.begin(), gbufferPtxData.end());
    gbufferPipeline_->optixModule = gbufferPipeline_->optixPipeline.createModuleFromPTXString (
        gbufferPtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT (OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT (OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    LOG (INFO) << "Shocker G-buffer module created successfully";

    // Load PTX for path tracing kernels
    std::vector<char> pathTracePtxData = ptxManager_->getPTXData ("optix_shocker_kernels");
    if (pathTracePtxData.empty())
    {
        LOG (WARNING) << "Failed to load PTX for optix_shocker_kernels";
        return;
    }

    // Create module for path tracing pipeline
    std::string pathTracePtxString (pathTracePtxData.begin(), pathTracePtxData.end());
    pathTracePipeline_->optixModule = pathTracePipeline_->optixPipeline.createModuleFromPTXString (
        pathTracePtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT (OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT (OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    LOG (INFO) << "Shocker path tracing module created successfully";
}

void ShockerEngine::createPrograms()
{
    LOG (INFO) << "ShockerEngine::createPrograms()";

    if (!gbufferPipeline_ || !gbufferPipeline_->optixModule || !gbufferPipeline_->optixPipeline ||
        !pathTracePipeline_ || !pathTracePipeline_->optixModule || !pathTracePipeline_->optixPipeline)
    {
        LOG (WARNING) << "Pipelines not ready";
        return;
    }

    optixu::Module emptyModule; // For empty programs

    // G-Buffer Pipeline Programs
    {
        auto& p = gbufferPipeline_->optixPipeline;
        auto& m = gbufferPipeline_->optixModule;

        // Create ray generation program for G-buffer setup
        gbufferPipeline_->entryPoints[GBufferEntryPoint::setupGBuffers] =
            p.createRayGenProgram (m, RT_RG_NAME_STR ("setupGBuffers"));

        // Create miss program for G-buffer
        gbufferPipeline_->programs[RT_MS_NAME_STR ("setupGBuffers")] =
            p.createMissProgram (m, RT_MS_NAME_STR ("setupGBuffers"));

        // Create hit group for G-buffer generation
        gbufferPipeline_->hitPrograms[RT_CH_NAME_STR("setupGBuffers")] =
            p.createHitProgramGroupForTriangleIS (
                m, RT_CH_NAME_STR("setupGBuffers"),
                emptyModule, nullptr);

        // Set the entry point
        gbufferPipeline_->setEntryPoint (GBufferEntryPoint::setupGBuffers);

        // Configure miss programs for ray types
        p.setNumMissRayTypes (shocker_shared::maxNumRayTypes);
        p.setMissProgram (0, // Primary ray type
                          gbufferPipeline_->programs.at (RT_MS_NAME_STR ("setupGBuffers")));
        p.setMissProgram (1, // Shadow ray type (not used in G-buffer but needs to be set)
                          gbufferPipeline_->programs.at (RT_MS_NAME_STR ("setupGBuffers")));

        LOG (INFO) << "G-buffer pipeline programs created";
    }
    
    // Create default material and set hit groups (following working sample pattern)
    // This must be done AFTER hit programs are created but BEFORE any geometry is added
    {
        optixu::Context optixContext = renderContext_->getOptiXContext();
        defaultMaterial_ = optixContext.createMaterial();
        
        // Create empty hit group for unused ray types (exactly as in sample code)
        gbufferPipeline_->hitPrograms["emptyHitGroup"] = 
            gbufferPipeline_->optixPipeline.createEmptyHitProgramGroup();
        
        // Set hit group for G-buffer pipeline primary ray type
        defaultMaterial_.setHitGroup(shocker_shared::GBufferRayType::Primary, 
                                    gbufferPipeline_->hitPrograms.at(RT_CH_NAME_STR("setupGBuffers")));
        
        // Set empty hit groups for all other ray types (exactly as in sample code)
        for (uint32_t rayType = shocker_shared::GBufferRayType::NumTypes; 
             rayType < shocker_shared::maxNumRayTypes; ++rayType) {
            defaultMaterial_.setHitGroup(rayType, gbufferPipeline_->hitPrograms.at("emptyHitGroup"));
        }
        
        // Pass the default material to the scene handler
        if (sceneHandler_) {
            sceneHandler_->setDefaultMaterial(defaultMaterial_);
        }
        
        LOG(INFO) << "Created default material with G-buffer hit groups for all ray types";
    }

    // Path Tracing Pipeline Programs
    {
        auto& p = pathTracePipeline_->optixPipeline;
        auto& m = pathTracePipeline_->optixModule;

        // Create ray generation program for path tracing
        pathTracePipeline_->entryPoints[PathTracingEntryPoint::pathTrace] =
            p.createRayGenProgram (m, RT_RG_NAME_STR ("raygen"));

        // Create miss program for path tracing
        pathTracePipeline_->programs[RT_MS_NAME_STR ("miss")] =
            p.createMissProgram (m, RT_MS_NAME_STR ("miss"));

        // Create hit group for path tracing (closest hit)
        pathTracePipeline_->hitPrograms[RT_CH_NAME_STR ("shading")] =
            p.createHitProgramGroupForTriangleIS (
                m, RT_CH_NAME_STR ("shading"),
                emptyModule, nullptr);

        // Create hit group for visibility rays (any hit only)
        pathTracePipeline_->hitPrograms[RT_AH_NAME_STR ("visibility")] =
            p.createHitProgramGroupForTriangleIS (
                emptyModule, nullptr,
                m, RT_AH_NAME_STR ("visibility"));

        // Create empty miss program for visibility rays
        pathTracePipeline_->programs["emptyMiss"] = p.createMissProgram (emptyModule, nullptr);

        // Set the entry point
        pathTracePipeline_->setEntryPoint (PathTracingEntryPoint::pathTrace);

        // Configure miss programs for ray types
        p.setNumMissRayTypes (shocker_shared::maxNumRayTypes);
        p.setMissProgram (0, // Closest ray type
                          pathTracePipeline_->programs.at (RT_MS_NAME_STR ("miss")));
        p.setMissProgram (1, // Visibility ray type
                          pathTracePipeline_->programs.at ("emptyMiss"));

        LOG (INFO) << "Path tracing pipeline programs created";
        
        // CRITICAL: Set path tracing hit groups on the default material
        // This was missing and is why the closest hit shader was never called!
        defaultMaterial_.setHitGroup(shocker_shared::PathTracingRayType::Closest,
                                    pathTracePipeline_->hitPrograms.at(RT_CH_NAME_STR("shading")));
        defaultMaterial_.setHitGroup(shocker_shared::PathTracingRayType::Visibility,
                                    pathTracePipeline_->hitPrograms.at(RT_AH_NAME_STR("visibility")));
        
        LOG(INFO) << "Set path tracing hit groups on default material";
    }
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

    LOG (INFO) << "Light probability kernels initialized successfully";
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

    // Create SBT for G-buffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline)
    {
        auto& p = gbufferPipeline_->optixPipeline;
        size_t sbtSize;
        p.generateShaderBindingTableLayout (&sbtSize);

        LOG (INFO) << "G-buffer pipeline SBT size: " << sbtSize << " bytes";

        if (sbtSize > 0)
        {
            gbufferPipeline_->sbt.initialize (
                cuContext, cudau::BufferType::Device, sbtSize, 1);
            gbufferPipeline_->sbt.setMappedMemoryPersistent (true);
            p.setShaderBindingTable (gbufferPipeline_->sbt, gbufferPipeline_->sbt.getMappedPointer());
        }

        // Set hit group SBT for G-buffer pipeline
        if (!gbufferPipeline_->hitGroupSbt.isInitialized())
        {
            if (hitGroupSbtSize > 0)
            {
                gbufferPipeline_->hitGroupSbt.initialize (cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
                gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent (true);
            }
        }
        gbufferPipeline_->optixPipeline.setHitGroupShaderBindingTable (
            gbufferPipeline_->hitGroupSbt, gbufferPipeline_->hitGroupSbt.getMappedPointer());

        LOG (INFO) << "G-buffer pipeline SBT created";
    }

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
}

void ShockerEngine::updateSBT()
{
    LOG (DBUG) << "Updating shader binding tables";

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

    // Update hit group SBT for G-buffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline && hitGroupSbtSize > 0)
    {
        // Resize if needed
        if (gbufferPipeline_->hitGroupSbt.isInitialized())
        {
            size_t currentSize = gbufferPipeline_->hitGroupSbt.sizeInBytes();
            if (currentSize < hitGroupSbtSize)
            {
                LOG (DBUG) << "Resizing G-buffer pipeline hit group SBT from " << currentSize << " to " << hitGroupSbtSize << " bytes";
                gbufferPipeline_->hitGroupSbt.resize (1, hitGroupSbtSize);
            }
        }
        else
        {
            gbufferPipeline_->hitGroupSbt.initialize (cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
            gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent (true);
        }
        gbufferPipeline_->optixPipeline.setHitGroupShaderBindingTable (
            gbufferPipeline_->hitGroupSbt, gbufferPipeline_->hitGroupSbt.getMappedPointer());
    }

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
    
    // NOTE: Pipelines do NOT need to be relinked after SBT updates.
    // The SBT can be updated without relinking as long as we're just resizing buffers.
    // Relinking is only needed when the pipeline structure changes (programs, modules, etc.)
}

void ShockerEngine::linkPipelines()
{
    LOG (INFO) << "ShockerEngine::linkPipelines()";

    // Link G-buffer pipeline with depth 1 (no recursive rays)
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline)
    {
        // Set the scene on the pipeline  
        gbufferPipeline_->optixPipeline.setScene (scene_);
        
        gbufferPipeline_->optixPipeline.link (1);
        LOG (INFO) << "G-buffer pipeline linked successfully";
    }

    // Link path tracing pipeline with depth 2 (for recursive rays)
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
    {
        // Set the scene on the pipeline
        pathTracePipeline_->optixPipeline.setScene (scene_);
        
        pathTracePipeline_->optixPipeline.link (2);
        LOG (INFO) << "Path tracing pipeline linked successfully";
    }
}

void ShockerEngine::updateMaterialHitGroups (ShockerModelPtr model)
{
    LOG (DBUG) << "Updating material hit groups";

    if (!model) {
        // If no specific model, update all models
        if (!modelHandler_) {
            return;
        }
        
        // Get all models and update their hit groups
        auto models = modelHandler_->getAllModels();
        for (const auto& [id, m] : models) {
            if (m) {
                updateMaterialHitGroups(m);
            }
        }
        return;
    }

    // Get the hit program for G-buffer
    if (!gbufferPipeline_ || !gbufferPipeline_->optixPipeline) {
        LOG(WARNING) << "G-buffer pipeline not ready yet";
        return;
    }
    
    auto it = gbufferPipeline_->hitPrograms.find(RT_CH_NAME_STR("setupGBuffers"));
    if (it == gbufferPipeline_->hitPrograms.end()) {
        LOG(WARNING) << "setupGBuffers hit program not found";
        return;
    }
    
    optixu::HitProgramGroup hitProgram = it->second;

    // Set the hit group on all surfaces' materials
    for (const auto& surface : model->getSurfaces()) {
        if (!surface || !surface->optixGeomInst) continue;
        
        // Get the material and set the hit group
        optixu::Material mat = surface->optixGeomInst.getMaterial(0, 0);
        if (mat) {
            mat.setHitGroup(0, hitProgram);  // Ray type 0 for primary rays
            LOG(DBUG) << "Set hit group for surface material";
        }
    }
}

void ShockerEngine::updateLaunchParameters (const mace::InputEvent& input)
{
    // Update per-frame parameters
    // Get traversable handle from scene handler
    if (sceneHandler_) {
        perFramePlp_.travHandle = sceneHandler_->getTraversableHandle();
    } else {
        perFramePlp_.travHandle = 0;
    }
    perFramePlp_.numAccumFrames = restartRender_ ? 0 : perFramePlp_.numAccumFrames + 1;
    perFramePlp_.frameIndex = frameCounter_;
    perFramePlp_.camera = lastCamera_;
    perFramePlp_.prevCamera = prevCamera_;
    perFramePlp_.mousePosition = make_int2 (0, 0); // TODO: get mouse position from input
    perFramePlp_.maxPathLength = 8;
    perFramePlp_.bufferIndex = frameCounter_ & 1;
    perFramePlp_.enableJittering = true;
    perFramePlp_.enableEnvLight = true;
    perFramePlp_.enableDenoiser = (denoiserHandler_ != nullptr);
    perFramePlp_.renderMode = static_cast<uint32_t> (renderMode_);

    // Update static parameters if needed
    // NOTE: Don't update buffer pointers here as they were set during initialization
    // Only update things that might change during runtime
    if (restartRender_ && renderHandler_)
    {
        staticPlp_.imageSize = make_int2 (renderHandler_->getWidth(), renderHandler_->getHeight());

        // Only update accumulation buffer references from render handler if they might have changed
        staticPlp_.beautyAccumBuffer = renderHandler_->getBeautyAccumSurfaceObject();
        staticPlp_.albedoAccumBuffer = renderHandler_->getAlbedoAccumSurfaceObject();
        staticPlp_.normalAccumBuffer = renderHandler_->getNormalAccumSurfaceObject();
        staticPlp_.motionAccumBuffer = renderHandler_->getMotionAccumSurfaceObject();

        // TODO: Update material and instance buffers from handlers when they change
    }

    // Update environment texture and importance map (following MiloEngine pattern)
    staticPlp_.envLightTexture = 0;
    if (renderContext_)
    {
        auto& handlers = renderContext_->getHandlers();
        if (handlers.skyDomeHandler && handlers.skyDomeHandler->hasEnvironmentTexture())
        {
            staticPlp_.envLightTexture = handlers.skyDomeHandler->getEnvironmentTexture();

            // Get the environment light importance map
            handlers.skyDomeHandler->getImportanceMap().getDeviceType (&staticPlp_.envLightImportanceMap);
        }
        else
        {
            // Initialize with empty distribution
            staticPlp_.envLightImportanceMap = shared::RegularConstantContinuousDistribution2D();
        }
    }

    // Ensure device pointers are allocated
    if (!plpOnDevice_ || !staticPlpOnDevice_ || !perFramePlpOnDevice_)
    {
        LOG (WARNING) << "Launch parameters not allocated, cannot update";
        return;
    }

    // Upload to device (exactly matching sample code pattern)
    // Upload per-frame parameters
    CUDADRV_CHECK (cuMemcpyHtoDAsync (perFramePlpOnDevice_, &perFramePlp_, sizeof (perFramePlp_), renderContext_->getCudaStream()));

    // Make sure the pointers in plp_ are set correctly (they get lost during updates)
    plp_.s = reinterpret_cast<shocker_shared::StaticPipelineLaunchParameters*> (staticPlpOnDevice_);
    plp_.f = reinterpret_cast<shocker_shared::PerFramePipelineLaunchParameters*> (perFramePlpOnDevice_);

    // Upload the main launch parameters structure
    CUDADRV_CHECK (cuMemcpyHtoDAsync (plpOnDevice_, &plp_, sizeof (plp_), renderContext_->getCudaStream()));

    // Upload static parameters on first render or when needed
    static bool firstRender = true;
    if (firstRender || restartRender_)
    {
        CUDADRV_CHECK (cuMemcpyHtoD (staticPlpOnDevice_, &staticPlp_, sizeof (staticPlp_)));
        firstRender = false;
        restartRender_ = false;
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

    // Initialize per-frame parameters before uploading
    memset (&perFramePlp_, 0, sizeof (perFramePlp_)); // Zero initialize
    perFramePlp_.travHandle = 0;                      // Empty scene initially
    perFramePlp_.numAccumFrames = 0;
    perFramePlp_.frameIndex = 0;
    perFramePlp_.camera = lastCamera_;
    perFramePlp_.prevCamera = lastCamera_;
    perFramePlp_.envLightPowerCoeff = 1.0f;
    perFramePlp_.envLightRotation = 0.0f;
    perFramePlp_.mousePosition = make_int2 (0, 0);
    perFramePlp_.maxPathLength = 8;
    perFramePlp_.bufferIndex = 0;
    perFramePlp_.resetFlowBuffer = 0;
    perFramePlp_.enableJittering = 1;
    perFramePlp_.enableEnvLight = 1;
    perFramePlp_.enableBumpMapping = 1;
    perFramePlp_.enableDebugPrint = 0;
    perFramePlp_.enableDenoiser = 0;
    perFramePlp_.renderMode = 0; // GBuffer mode
    perFramePlp_.debugSwitches = 0;

    // Allocate device memory for static parameters (matching sample code)
    CUDADRV_CHECK (cuMemAlloc (&staticPlpOnDevice_, sizeof (shocker_shared::StaticPipelineLaunchParameters)));
    CUDADRV_CHECK (cuMemcpyHtoD (staticPlpOnDevice_, &staticPlp_, sizeof (staticPlp_)));

    // Allocate device memory for per-frame parameters
    CUDADRV_CHECK (cuMemAlloc (&perFramePlpOnDevice_, sizeof (shocker_shared::PerFramePipelineLaunchParameters)));
    CUDADRV_CHECK (cuMemcpyHtoD (perFramePlpOnDevice_, &perFramePlp_, sizeof (perFramePlp_)));

    // Setup pointers in main launch parameter structure
    plp_.s = reinterpret_cast<shocker_shared::StaticPipelineLaunchParameters*> (staticPlpOnDevice_);
    plp_.f = reinterpret_cast<shocker_shared::PerFramePipelineLaunchParameters*> (perFramePlpOnDevice_);

    // Allocate device memory for the main launch parameters structure
    CUDADRV_CHECK (cuMemAlloc (&plpOnDevice_, sizeof (shocker_shared::PipelineLaunchParameters)));

    // Upload the main launch parameters structure with the pointers (critical!)
    CUDADRV_CHECK (cuMemcpyHtoD (plpOnDevice_, &plp_, sizeof (plp_)));

    LOG (INFO) << "Launch parameters allocated and uploaded to device";
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

        // Note: Lens parameters (lensSize, focusDistance) are not in the basic PerspectiveCamera
        // These would be needed for depth of field effects in a more advanced version

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

void ShockerEngine::renderGBuffer(CUstream stream)
{
    if (!gbufferPipeline_ || !gbufferPipeline_->optixPipeline)
    {
        LOG (WARNING) << "G-buffer pipeline not ready";
        return;
    }

    if (!plpOnDevice_ || !staticPlpOnDevice_ || !perFramePlpOnDevice_)
    {
        LOG (WARNING) << "Launch parameters not allocated for G-buffer pipeline";
        return;
    }

    uint32_t width = renderHandler_->getWidth();
    uint32_t height = renderHandler_->getHeight();

    // Launch the G-buffer pipeline - let exceptions propagate
    gbufferPipeline_->optixPipeline.launch (stream, plpOnDevice_, width, height, 1);
}

void ShockerEngine::renderPathTracing(CUstream stream)
{
    if (!pathTracePipeline_ || !pathTracePipeline_->optixPipeline)
    {
        LOG (WARNING) << "Path tracing pipeline not ready";
        return;
    }

    //LOG (DBUG) << _FN_;
    uint32_t width = renderHandler_->getWidth();
    uint32_t height = renderHandler_->getHeight();

    // Launch path tracing pipeline
    pathTracePipeline_->setEntryPoint (PathTracingEntryPoint::pathTrace);

    // TODO: Launch when pipeline is fully configured
    pathTracePipeline_->optixPipeline.launch (stream, plpOnDevice_, width, height, 1);

    // Copy accumulation buffers to linear buffers for display
    renderHandler_->copyAccumToLinearBuffers (stream);

    // Apply denoising if enabled
    if (denoiserHandler_ && perFramePlp_.enableDenoiser)
    {
        bool isNewSequence = (perFramePlp_.numAccumFrames == 0);
        // Pass timer to denoise method if available
        auto timer = getCurrentTimer();
        renderHandler_->denoise (stream, isNewSequence, denoiserHandler_.get(), timer);
    }
}