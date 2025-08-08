#include "ShockerEngine.h"
#include "../handlers/ShockerSceneHandler.h"
#include "../handlers/ShockerMaterialHandler.h"
#include "../handlers/ShockerModelHandler.h"
#include "../handlers/ShockerRenderHandler.h"
#include "../handlers/ShockerDenoiserHandler.h"
#include "../handlers/AreaLightHandler.h"
#include "../model/ShockerModel.h"
#include "../tools/PTXManager.h"

ShockerEngine::ShockerEngine()
{
    LOG(DBUG) << "ShockerEngine constructor";
}

ShockerEngine::~ShockerEngine()
{
    LOG(DBUG) << "ShockerEngine destructor";
    cleanup();
}

void ShockerEngine::initialize(RenderContext* ctx)
{
    LOG(DBUG) << "Initializing ShockerEngine";
    
    // Call base class initialization first
    // This will set up renderContext_, context_, ptxManager_ and initialize dimensions
    BaseRenderingEngine::initialize(ctx);
    
    if (!isInitialized_) {
        LOG(WARNING) << "Base class initialization failed";
        return;
    }
    
    // Initialize handlers
    auto ctxPtr = std::shared_ptr<RenderContext>(renderContext_, [](RenderContext*){});
    sceneHandler_ = ShockerSceneHandler::create(ctxPtr);
    materialHandler_ = std::make_shared<ShockerMaterialHandler>();
    modelHandler_ = std::make_shared<ShockerModelHandler>();
    modelHandler_->initialize(ctxPtr);
    renderHandler_ = ShockerRenderHandler::create(ctxPtr);
    denoiserHandler_ = ShockerDenoiserHandler::create(ctxPtr);
    areaLightHandler_ = std::make_shared<AreaLightHandler>();
    
    if (!sceneHandler_ || !materialHandler_ || !modelHandler_ || !renderHandler_) {
        LOG(WARNING) << "Failed to create handlers";
        return;
    }
    
    // Initialize render handler with dimensions from base class
    initializeHandlerWithDimensions(renderHandler_, "RenderHandler");
    
    // Initialize denoiser if available
    initializeHandlerWithDimensions(denoiserHandler_, "DenoiserHandler");
    
    // Setup pipelines
    setupPipelines();
    
    // Allocate launch parameters
    allocateLaunchParameters();
    
    // Initialize buffers
    if (renderWidth_ > 0 && renderHeight_ > 0 && renderContext_)
    {
        CUcontext cuContext = renderContext_->getCudaContext();
        
        // Initialize RNG buffer
        rngBuffer_.initialize(cuContext, cudau::BufferType::Device, renderWidth_, renderHeight_);
        uint64_t seed = 12345;
        rngBuffer_.map();
        for (int y = 0; y < renderHeight_; ++y) {
            for (int x = 0; x < renderWidth_; ++x) {
                shared::PCG32RNG& rng = rngBuffer_(x, y);
                rng.setState(seed + (y * renderWidth_ + x) * 1234567);
            }
        }
        rngBuffer_.unmap();
        
        // Initialize G-buffers (double buffered, matching sample code pattern)
        for (int i = 0; i < 2; ++i)
        {
            gBuffer0_[i].initialize2D(
                cuContext, cudau::ArrayElementType::UInt32, 
                (sizeof(shocker_shared::GBuffer0Elements) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderWidth_, renderHeight_, 1);
            
            gBuffer1_[i].initialize2D(
                cuContext, cudau::ArrayElementType::UInt32,
                (sizeof(shocker_shared::GBuffer1Elements) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                renderWidth_, renderHeight_, 1);
        }
        LOG(DBUG) << "G-buffers initialized";
    }
    
    // Initialize camera
    lastCamera_.aspect = static_cast<float>(renderWidth_) / renderHeight_;
    lastCamera_.fovY = 45.0f * pi_v<float> / 180.0f;
    lastCamera_.position = Point3D(0, 0, 5);
    lastCamera_.orientation.c0 = Vector3D(1, 0, 0);
    lastCamera_.orientation.c1 = Vector3D(0, 1, 0);
    lastCamera_.orientation.c2 = Vector3D(0, 0, 1);
    prevCamera_ = lastCamera_;
    
    isInitialized_ = true;
    LOG(DBUG) << "ShockerEngine initialized successfully";
}

void ShockerEngine::cleanup()
{
    if (!isInitialized_) {
        return;
    }
    
    LOG(DBUG) << "Cleaning up ShockerEngine";
    
    // Clean up pipelines
    if (gbufferPipeline_) {
        gbufferPipeline_->destroy();
        gbufferPipeline_.reset();
    }
    
    if (pathTracePipeline_) {
        pathTracePipeline_->destroy();
        pathTracePipeline_.reset();
    }
    
    // Clean up buffers
    if (staticPlpBuffer_.isInitialized()) {
        staticPlpBuffer_.finalize();
    }
    if (perFramePlpBuffer_.isInitialized()) {
        perFramePlpBuffer_.finalize();
    }
    
    // Clean up RNG buffer
    if (rngBuffer_.isInitialized()) {
        rngBuffer_.finalize();
    }
    
    // Clean up G-buffers
    for (int i = 1; i >= 0; --i) {
        if (gBuffer1_[i].isInitialized()) {
            gBuffer1_[i].finalize();
        }
        if (gBuffer0_[i].isInitialized()) {
            gBuffer0_[i].finalize();
        }
    }
    
    // Clean up scene
    if (scene_) {
        scene_.destroy();
    }
    
    // Clean up handlers
    if (denoiserHandler_) {
        denoiserHandler_->finalize();
        denoiserHandler_.reset();
    }
    
    if (renderHandler_) {
        renderHandler_->finalize();
        renderHandler_.reset();
    }
    
    if (areaLightHandler_) {
        areaLightHandler_.reset();
    }
    
    sceneHandler_.reset();
    materialHandler_.reset();
    modelHandler_.reset();
    
    isInitialized_ = false;
    LOG(DBUG) << "ShockerEngine cleanup complete";
}

void ShockerEngine::addGeometry(sabi::RenderableNode node)
{
    if (!isInitialized_) {
        LOG(WARNING) << "ShockerEngine not initialized";
        return;
    }
    
    LOG(DBUG) << "Adding geometry to ShockerEngine";
    
    // TODO: Convert RenderableNode to ShockerModel and add to scene
    // This will be implemented when we have the full ShockerModel implementation
    
    // For now, just mark that we need to update the scene
    restartRender_ = true;
}

void ShockerEngine::clearScene()
{
    if (!isInitialized_) {
        return;
    }
    
    LOG(DBUG) << "Clearing ShockerEngine scene";
    
    // Clear models from handlers
    if (modelHandler_) {
        // modelHandler doesn't have clearScene, models are cleared in clear()
    }
    
    // Reset scene
    if (scene_) {
        scene_.destroy();
        scene_ = context_->createScene();
    }
    
    restartRender_ = true;
}

void ShockerEngine::resize(uint32_t width, uint32_t height)
{
    if (!isInitialized_ || width == 0 || height == 0) {
        return;
    }
    
    LOG(DBUG) << "Resizing ShockerEngine from " << renderWidth_ << "x" << renderHeight_ 
              << " to " << width << "x" << height;
    
    renderWidth_ = width;
    renderHeight_ = height;
    
    CUcontext cuContext = renderContext_->getCudaContext();
    
    // Resize RNG buffer
    if (rngBuffer_.isInitialized()) {
        rngBuffer_.finalize();
    }
    rngBuffer_.initialize(cuContext, cudau::BufferType::Device, width, height);
    uint64_t seed = 12345;
    rngBuffer_.map();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            shared::PCG32RNG& rng = rngBuffer_(x, y);
            rng.setState(seed + (y * width + x) * 1234567);
        }
    }
    rngBuffer_.unmap();
    
    // Resize G-buffers
    for (int i = 0; i < 2; ++i) {
        if (gBuffer0_[i].isInitialized()) {
            gBuffer0_[i].resize(width, height);
        } else {
            gBuffer0_[i].initialize2D(
                cuContext, cudau::ArrayElementType::UInt32,
                (sizeof(shocker_shared::GBuffer0Elements) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
        }
        
        if (gBuffer1_[i].isInitialized()) {
            gBuffer1_[i].resize(width, height);
        } else {
            gBuffer1_[i].initialize2D(
                cuContext, cudau::ArrayElementType::UInt32,
                (sizeof(shocker_shared::GBuffer1Elements) + 3) / 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
        }
    }
    
    // Resize render handler
    if (renderHandler_) {
        renderHandler_->resize(width, height);
    }
    
    // Resize denoiser handler
    if (denoiserHandler_) {
        denoiserHandler_->resize(width, height);
    }
    
    // Update camera aspect ratio
    lastCamera_.aspect = static_cast<float>(width) / height;
    
    // Mark for render restart
    restartRender_ = true;
    LOG(DBUG) << "ShockerEngine resize complete";
}

void ShockerEngine::render(const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber)
{
    if (!isInitialized_) {
        return;
    }
    
    // Update camera if needed
    // Update camera (simplified for now)
    if (true) { // TODO: proper input comparison
        updateCameraBody(input);
        updateCameraSensor();
        lastInput_ = input;
    }
    
    // Update launch parameters
    updateLaunchParameters(input);
    
    // Choose rendering path based on mode
    switch (renderMode_) {
        case RenderMode::GBufferPreview:
        case RenderMode::DebugNormals:
        case RenderMode::DebugAlbedo:
        case RenderMode::DebugDepth:
        case RenderMode::DebugMotion:
            renderGBuffer();
            break;
            
        case RenderMode::PathTraceFinal:
            renderPathTracing();
            break;
    }
    
    // Update frame counter
    frameCounter_++;
    
    // Update motion state
    if (updateMotion) {
        prevCamera_ = lastCamera_;
    }
}

void ShockerEngine::onEnvironmentChanged()
{
    LOG(DBUG) << "Environment changed in ShockerEngine";
    environmentDirty_ = true;
    restartRender_ = true;
}

void ShockerEngine::setupPipelines()
{
    LOG(DBUG) << "Setting up ShockerEngine pipelines";
    
    // Create pipeline instances
    gbufferPipeline_ = std::make_shared<engine_core::RenderPipeline<GBufferEntryPoint>>();
    pathTracePipeline_ = std::make_shared<engine_core::RenderPipeline<PathTracingEntryPoint>>();
    
    // Setup G-buffer pipeline
    createGBufferPipeline();
    
    // Setup path tracing pipeline
    createPathTracingPipeline();
    
    // Create shader binding tables
    createSBTs();
    
    // Link pipelines
    linkPipelines();
}

void ShockerEngine::createGBufferPipeline()
{
    LOG(DBUG) << "Creating G-buffer pipeline";
    
    if (!ptxManager_ || !gbufferPipeline_) {
        LOG(WARNING) << "PTXManager or G-buffer pipeline not ready";
        return;
    }
    
    // Pipeline configuration
    engine_core::PipelineConfig config;
    config.maxTraceDepth = 1;  // Single bounce for G-buffer
    config.numPayloadDwords = 8;  // For G-buffer payload
    config.numAttributeDwords = 2;
    
    // Initialize pipeline
    gbufferPipeline_->initialize(renderContext_, config);
    
    // Load PTX for G-buffer kernels
    std::vector<char> gbufferPtxData = ptxManager_->getPTXData("optix_shocker_gbuffer");
    if (gbufferPtxData.empty()) {
        LOG(WARNING) << "Failed to load PTX for optix_shocker_gbuffer";
        return;
    }
    
    // Create module from PTX
    std::string gbufferPtxString(gbufferPtxData.begin(), gbufferPtxData.end());
    gbufferPipeline_->optixModule = gbufferPipeline_->optixPipeline.createModuleFromPTXString(
        gbufferPtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));
    
    // Create program groups
    gbufferPipeline_->entryPoints[GBufferEntryPoint::setupGBuffers] = 
        gbufferPipeline_->optixPipeline.createRayGenProgram(
            gbufferPipeline_->optixModule, "__raygen__setupGBuffers");
    
    gbufferPipeline_->programs["__miss__setupGBuffers"] = 
        gbufferPipeline_->optixPipeline.createMissProgram(
            gbufferPipeline_->optixModule, "__miss__setupGBuffers");
    
    // Create hit group for G-buffer generation
    gbufferPipeline_->hitPrograms["__closesthit__setupGBuffers"] = 
        gbufferPipeline_->optixPipeline.createHitProgramGroupForTriangleIS(
            gbufferPipeline_->optixModule, "__closesthit__setupGBuffers",
            optixu::Module(), nullptr);
    
    LOG(DBUG) << "G-buffer pipeline created successfully";
}

void ShockerEngine::createPathTracingPipeline()
{
    LOG(DBUG) << "Creating path tracing pipeline";
    
    if (!ptxManager_ || !pathTracePipeline_) {
        LOG(WARNING) << "PTXManager or path tracing pipeline not ready";
        return;
    }
    
    // Pipeline configuration
    engine_core::PipelineConfig config;
    config.maxTraceDepth = 8;  // Multiple bounces for path tracing
    config.numPayloadDwords = 16;  // For path trace payload
    config.numAttributeDwords = 2;
    
    // Initialize pipeline
    pathTracePipeline_->initialize(renderContext_, config);
    
    // Load PTX for path tracing kernels
    std::vector<char> pathTracePtxData = ptxManager_->getPTXData("optix_shocker_kernels");
    if (pathTracePtxData.empty()) {
        LOG(WARNING) << "Failed to load PTX for optix_shocker_kernels";
        return;
    }
    
    // Create module from PTX
    std::string pathTracePtxString(pathTracePtxData.begin(), pathTracePtxData.end());
    pathTracePipeline_->optixModule = pathTracePipeline_->optixPipeline.createModuleFromPTXString(
        pathTracePtxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));
    
    // Create program groups
    pathTracePipeline_->entryPoints[PathTracingEntryPoint::pathTrace] = 
        pathTracePipeline_->optixPipeline.createRayGenProgram(
            pathTracePipeline_->optixModule, "__raygen__pathTrace");
    
    pathTracePipeline_->programs["__miss__pathTrace"] = 
        pathTracePipeline_->optixPipeline.createMissProgram(
            pathTracePipeline_->optixModule, "__miss__pathTrace");
    
    // Create hit groups for path tracing
    pathTracePipeline_->hitPrograms["__closesthit__pathTrace"] = 
        pathTracePipeline_->optixPipeline.createHitProgramGroupForTriangleIS(
            pathTracePipeline_->optixModule, "__closesthit__pathTrace",
            optixu::Module(), nullptr);
    
    // Shadow hit group for shadow rays  
    pathTracePipeline_->hitPrograms["__anyhit__visibility"] = 
        pathTracePipeline_->optixPipeline.createHitProgramGroupForTriangleIS(
            optixu::Module(), nullptr,
            pathTracePipeline_->optixModule, "__anyhit__visibility");
    
    LOG(DBUG) << "Path tracing pipeline created successfully";
}

void ShockerEngine::createSBTs()
{
    LOG(DBUG) << "Creating shader binding tables";
    
    if (!renderContext_)
    {
        LOG(WARNING) << "No render context for SBT creation";
        return;
    }
    
    auto cuContext = renderContext_->getCudaContext();
    
    // Get hit group SBT size from scene
    size_t hitGroupSbtSize = 0;
    scene_.generateShaderBindingTableLayout(&hitGroupSbtSize);
    LOG(DBUG) << "Scene hit group SBT size: " << hitGroupSbtSize << " bytes";
    
    // Create SBT for G-buffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline)
    {
        auto& p = gbufferPipeline_->optixPipeline;
        size_t sbtSize;
        p.generateShaderBindingTableLayout(&sbtSize);
        
        LOG(DBUG) << "G-buffer pipeline SBT size: " << sbtSize << " bytes";
        
        if (sbtSize > 0)
        {
            gbufferPipeline_->sbt.initialize(
                cuContext, cudau::BufferType::Device, sbtSize, 1);
            gbufferPipeline_->sbt.setMappedMemoryPersistent(true);
            p.setShaderBindingTable(gbufferPipeline_->sbt, gbufferPipeline_->sbt.getMappedPointer());
        }
        
        // Set hit group SBT for G-buffer pipeline
        if (!gbufferPipeline_->hitGroupSbt.isInitialized())
        {
            if (hitGroupSbtSize > 0)
            {
                gbufferPipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
                gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
            }
        }
        p.setHitGroupShaderBindingTable(gbufferPipeline_->hitGroupSbt, gbufferPipeline_->hitGroupSbt.getMappedPointer());
    }
    
    // Create SBT for path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
    {
        auto& p = pathTracePipeline_->optixPipeline;
        size_t sbtSize;
        p.generateShaderBindingTableLayout(&sbtSize);
        
        LOG(DBUG) << "Path tracing pipeline SBT size: " << sbtSize << " bytes";
        
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
        }
        p.setHitGroupShaderBindingTable(pathTracePipeline_->hitGroupSbt, pathTracePipeline_->hitGroupSbt.getMappedPointer());
    }
    
    LOG(DBUG) << "Shader binding tables created successfully";
}

void ShockerEngine::updateSBTs()
{
    LOG(DBUG) << "Updating shader binding tables";
    
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
    
    // Update hit group SBT for G-buffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline && hitGroupSbtSize > 0)
    {
        // Resize if needed
        if (gbufferPipeline_->hitGroupSbt.isInitialized())
        {
            size_t currentSize = gbufferPipeline_->hitGroupSbt.sizeInBytes();
            if (currentSize < hitGroupSbtSize)
            {
                LOG(DBUG) << "Resizing G-buffer pipeline hit group SBT from " << currentSize << " to " << hitGroupSbtSize << " bytes";
                gbufferPipeline_->hitGroupSbt.resize(1, hitGroupSbtSize);
            }
        }
        else
        {
            gbufferPipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
            gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
        }
        gbufferPipeline_->optixPipeline.setHitGroupShaderBindingTable(
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

void ShockerEngine::linkPipelines()
{
    LOG(DBUG) << "Linking pipelines";
    
    // Link G-buffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline) {
        constexpr uint32_t maxTraceDepth = 1;
        gbufferPipeline_->optixPipeline.link(maxTraceDepth);
        
        // Set stack sizes
        size_t ccStackSize = 0;
        // Calculate stack size for pipeline
        // This is simplified - actual implementation needs proper stack calculation
        gbufferPipeline_->optixPipeline.setStackSize(ccStackSize, 0, 2048, maxTraceDepth);
        
        LOG(DBUG) << "G-buffer pipeline linked with CC stack size: " << ccStackSize;
    }
    
    // Link path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline) {
        constexpr uint32_t maxTraceDepth = 8;
        pathTracePipeline_->optixPipeline.link(maxTraceDepth);
        
        // Set stack sizes
        size_t ccStackSize = 0;
        // Calculate stack size for pipeline  
        // This is simplified - actual implementation needs proper stack calculation
        pathTracePipeline_->optixPipeline.setStackSize(ccStackSize, 0, 4096, maxTraceDepth);
        
        LOG(DBUG) << "Path tracing pipeline linked with CC stack size: " << ccStackSize;
    }
}

void ShockerEngine::updateMaterialHitGroups(ShockerModelPtr model)
{
    LOG(DBUG) << "Updating material hit groups";
    
    // TODO: Set hit groups on model's materials
    // This will be implemented with the full ShockerModel integration
}

void ShockerEngine::updateLaunchParameters(const mace::InputEvent& input)
{
    // Update per-frame parameters
    // Get traversable handle from scene (method depends on optixu::Scene API)
    perFramePlp_.travHandle = 0; // TODO: Get from scene when API is known
    perFramePlp_.numAccumFrames = restartRender_ ? 0 : perFramePlp_.numAccumFrames + 1;
    perFramePlp_.frameIndex = frameCounter_;
    perFramePlp_.camera = lastCamera_;
    perFramePlp_.prevCamera = prevCamera_;
    perFramePlp_.mousePosition = make_int2(0, 0); // TODO: get mouse position from input
    perFramePlp_.maxPathLength = 8;
    perFramePlp_.bufferIndex = frameCounter_ & 1;
    perFramePlp_.enableJittering = true;
    perFramePlp_.enableEnvLight = true;
    perFramePlp_.enableDenoiser = (denoiserHandler_ != nullptr);
    perFramePlp_.renderMode = static_cast<uint32_t>(renderMode_);
    
    // Update static parameters if needed
    if (restartRender_) {
        staticPlp_.imageSize = make_int2(renderHandler_->getWidth(), renderHandler_->getHeight());
        // RNG buffer is already a BlockBuffer2D, just get the native version
        staticPlp_.rngBuffer = rngBuffer_.getBlockBuffer2D();
        
        // Setup G-buffers (direct assignment of surface objects)
        for (int i = 0; i < 2; ++i) {
            staticPlp_.GBuffer0[i] = gBuffer0_[i].getSurfaceObject(0);
            staticPlp_.GBuffer1[i] = gBuffer1_[i].getSurfaceObject(0);
        }
        
        // Update accumulation buffer references from render handler
        if (renderHandler_) {
            // Direct assignment of surface objects from render handler
            staticPlp_.beautyAccumBuffer = renderHandler_->getBeautyAccumSurfaceObject();
            staticPlp_.albedoAccumBuffer = renderHandler_->getAlbedoAccumSurfaceObject();
            staticPlp_.normalAccumBuffer = renderHandler_->getNormalAccumSurfaceObject();
            staticPlp_.motionAccumBuffer = renderHandler_->getMotionAccumSurfaceObject();
        }
        
        // TODO: Update material and instance buffers from handlers
        
        restartRender_ = false;
    }
    
    // Upload to device
    if (staticPlpBuffer_.isInitialized() && perFramePlpBuffer_.isInitialized()) {
        staticPlpBuffer_.write(&staticPlp_, 1);
        perFramePlpBuffer_.write(&perFramePlp_, 1);
    }
}

void ShockerEngine::allocateLaunchParameters()
{
    LOG(DBUG) << "Allocating launch parameters";
    
    CUcontext cuContext = renderContext_->getCudaContext();
    
    // Allocate buffers for split launch parameters
    staticPlpBuffer_.initialize(cuContext, cudau::BufferType::Device, 
                                sizeof(shocker_shared::StaticPipelineLaunchParameters), 1);
    perFramePlpBuffer_.initialize(cuContext, cudau::BufferType::Device,
                                  sizeof(shocker_shared::PerFramePipelineLaunchParameters), 1);
    
    // Setup pointers in main launch parameter structure
    plp_.s = reinterpret_cast<shocker_shared::StaticPipelineLaunchParameters*>(
        staticPlpBuffer_.getDevicePointer());
    plp_.f = reinterpret_cast<shocker_shared::PerFramePipelineLaunchParameters*>(
        perFramePlpBuffer_.getDevicePointer());
    
    // Get device pointer for launch
    plpOnDevice_ = reinterpret_cast<CUdeviceptr>(&plp_);
}

void ShockerEngine::updateCameraBody(const mace::InputEvent& input)
{
    // Simple camera update based on input
    // TODO: Implement proper camera controls
    
    // TODO: check for camera changes properly
    if (false) {
        cameraChanged_ = true;
        restartRender_ = true;
    }
}

void ShockerEngine::updateCameraSensor()
{
    // Update camera sensor parameters
    if (cameraChanged_) {
        // Update aspect ratio if window resized
        if (renderHandler_) {
            lastCamera_.aspect = static_cast<float>(renderHandler_->getWidth()) / 
                                renderHandler_->getHeight();
        }
        cameraChanged_ = false;
    }
}

void ShockerEngine::renderGBuffer()
{
    if (!gbufferPipeline_ || !gbufferPipeline_->optixPipeline) {
        LOG(WARNING) << "G-buffer pipeline not ready";
        return;
    }
    
    CUstream stream = renderContext_->getCudaStream();
    uint32_t width = renderHandler_->getWidth();
    uint32_t height = renderHandler_->getHeight();
    
    // Launch G-buffer pipeline
    gbufferPipeline_->setEntryPoint(GBufferEntryPoint::setupGBuffers);
    
    // TODO: Launch when pipeline is fully configured
    // gbufferPipeline_->optixPipeline.launch(stream, plpOnDevice_, width, height, 1);
    
    // Visualize G-buffer based on debug mode
    if (renderMode_ != RenderMode::GBufferPreview) {
        // TODO: Copy specific G-buffer channel to display
    }
}

void ShockerEngine::renderPathTracing()
{
    if (!pathTracePipeline_ || !pathTracePipeline_->optixPipeline) {
        LOG(WARNING) << "Path tracing pipeline not ready";
        return;
    }
    
    CUstream stream = renderContext_->getCudaStream();
    uint32_t width = renderHandler_->getWidth();
    uint32_t height = renderHandler_->getHeight();
    
    // Clear accumulation buffers if restarting
    if (restartRender_) {
        renderHandler_->clearAccumBuffers(stream);
    }
    
    // Launch path tracing pipeline
    pathTracePipeline_->setEntryPoint(PathTracingEntryPoint::pathTrace);
    
    // TODO: Launch when pipeline is fully configured
    // pathTracePipeline_->optixPipeline.launch(stream, plpOnDevice_, width, height, 1);
    
    // Copy accumulation buffers to linear buffers for display
    renderHandler_->copyAccumToLinearBuffers(stream);
    
    // Apply denoising if enabled
    if (denoiserHandler_ && perFramePlp_.enableDenoiser) {
        bool isNewSequence = (perFramePlp_.numAccumFrames == 0);
        renderHandler_->denoise(stream, isNewSequence, denoiserHandler_.get());
    }
}