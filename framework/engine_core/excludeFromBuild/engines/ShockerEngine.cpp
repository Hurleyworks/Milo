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
    LOG(DBUG, "ShockerEngine constructor");
}

ShockerEngine::~ShockerEngine()
{
    LOG(DBUG) << "ShockerEngine destructor";
    cleanup();
}

void ShockerEngine::initialize(RenderContext* ctx)
{
    LOG(DBUG, "Initializing ShockerEngine");
    
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
    modelHandler_ = std::make_shared<ShockerModelHandler>(ctxPtr);
    renderHandler_ = ShockerRenderHandler::create(ctxPtr);
    denoiserHandler_ = ShockerDenoiserHandler::create(ctxPtr);
    areaLightHandler_ = std::make_shared<AreaLightHandler>();
    
    if (!sceneHandler_ || !materialHandler_ || !modelHandler_ || !renderHandler_) {
        LOG(WARNING, "Failed to create handlers");
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
    
    // Initialize RNG buffer
    if (renderWidth_ > 0 && renderHeight_ > 0)
    {
        rngBuffer_.initialize(renderWidth_, renderHeight_);
        uint64_t seed = 12345;
        rngBuffer_.map();
        for (int y = 0; y < renderHeight_; ++y) {
            for (int x = 0; x < renderWidth_; ++x) {
                shared::PCG32RNG& rng = rngBuffer_(x, y);
                rng.setState(seed + (y * renderWidth_ + x) * 1234567);
            }
        }
        rngBuffer_.unmap();
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
    LOG(DBUG, "ShockerEngine initialized successfully");
}

void ShockerEngine::cleanup()
{
    if (!isInitialized_) {
        return;
    }
    
    LOG(DBUG, "Cleaning up ShockerEngine");
    
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
    
    // Clean up G-buffer textures
    for (auto& tex : gbufferTextures_) {
        if (tex.isInitialized()) {
            tex.finalize();
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
    LOG(DBUG, "ShockerEngine cleanup complete");
}

void ShockerEngine::addGeometry(sabi::RenderableNode node)
{
    if (!isInitialized_) {
        LOG(WARNING, "ShockerEngine not initialized");
        return;
    }
    
    LOG(DBUG, "Adding geometry to ShockerEngine");
    
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
    
    LOG(DBUG, "Clearing ShockerEngine scene");
    
    // Clear models from handlers
    if (modelHandler_) {
        modelHandler_->clearModels();
    }
    
    // Reset scene
    if (scene_) {
        scene_.destroy();
        scene_ = context_->createScene();
    }
    
    restartRender_ = true;
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
    LOG(DBUG, "Environment changed in ShockerEngine");
    environmentDirty_ = true;
    restartRender_ = true;
}

void ShockerEngine::setupPipelines()
{
    LOG(DBUG, "Setting up ShockerEngine pipelines");
    
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
    LOG(DBUG, "Creating G-buffer pipeline");
    
    // TODO: Implementation will be added when we have PTX files
    // For now, this is a stub
    
    // Pipeline configuration
    engine_core::PipelineConfig config;
    config.maxTraceDepth = 1;  // Single bounce for G-buffer
    config.numPayloadDwords = 8;  // For G-buffer payload
    config.numAttributeDwords = 2;
    
    if (gbufferPipeline_) {
        gbufferPipeline_->initialize(context_.get(), config);
    }
}

void ShockerEngine::createPathTracingPipeline()
{
    LOG(DBUG, "Creating path tracing pipeline");
    
    // TODO: Implementation will be added when we have PTX files
    // For now, this is a stub
    
    // Pipeline configuration
    engine_core::PipelineConfig config;
    config.maxTraceDepth = 8;  // Multiple bounces for path tracing
    config.numPayloadDwords = 16;  // For path trace payload
    config.numAttributeDwords = 2;
    
    if (pathTracePipeline_) {
        pathTracePipeline_->initialize(context_.get(), config);
    }
}

void ShockerEngine::createSBTs()
{
    LOG(DBUG, "Creating shader binding tables");
    
    // TODO: Create SBTs for both pipelines
    // This will be implemented when we have the programs created
}

void ShockerEngine::updateSBTs()
{
    LOG(DBUG, "Updating shader binding tables");
    
    // TODO: Update SBTs after scene changes
}

void ShockerEngine::linkPipelines()
{
    LOG(DBUG, "Linking pipelines");
    
    // TODO: Link OptiX pipelines and calculate stack sizes
    // This will be implemented when pipelines are fully created
}

void ShockerEngine::updateMaterialHitGroups(ShockerModelPtr model)
{
    LOG(DBUG, "Updating material hit groups");
    
    // TODO: Set hit groups on model's materials
    // This will be implemented with the full ShockerModel integration
}

void ShockerEngine::updateLaunchParameters(const mace::InputEvent& input)
{
    // Update per-frame parameters
    perFramePlp_.travHandle = scene_.getTraversableHandle();
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
        staticPlp_.rngBuffer = rngBuffer_.getBlockBuffer2D();
        
        // Update accumulation buffer references from render handler
        if (renderHandler_) {
            staticPlp_.beautyAccumBuffer = optixu::NativeBlockBuffer2D<float4>(
                renderHandler_->getBeautyAccumSurfaceObject(),
                cudau::ArrayElementType::Float32, 4,
                renderHandler_->getWidth(), renderHandler_->getHeight());
            staticPlp_.albedoAccumBuffer = optixu::NativeBlockBuffer2D<float4>(
                renderHandler_->getAlbedoAccumSurfaceObject(),
                cudau::ArrayElementType::Float32, 4,
                renderHandler_->getWidth(), renderHandler_->getHeight());
            staticPlp_.normalAccumBuffer = optixu::NativeBlockBuffer2D<float4>(
                renderHandler_->getNormalAccumSurfaceObject(),
                cudau::ArrayElementType::Float32, 4,
                renderHandler_->getWidth(), renderHandler_->getHeight());
            staticPlp_.motionAccumBuffer = optixu::NativeBlockBuffer2D<float2>(
                renderHandler_->getMotionAccumSurfaceObject(),
                cudau::ArrayElementType::Float32, 2,
                renderHandler_->getWidth(), renderHandler_->getHeight());
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
    LOG(DBUG, "Allocating launch parameters");
    
    CUcontext cuContext = renderContext_->getCudaContext();
    
    // Allocate buffers for split launch parameters
    staticPlpBuffer_.initialize(cuContext, cudau::BufferType::Device, 
                                sizeof(shocker::ShockerStaticPipelineLaunchParameters), 1);
    perFramePlpBuffer_.initialize(cuContext, cudau::BufferType::Device,
                                  sizeof(shocker::ShockerPerFramePipelineLaunchParameters), 1);
    
    // Setup pointers in main launch parameter structure
    plp_.s = reinterpret_cast<shocker::ShockerStaticPipelineLaunchParameters*>(
        staticPlpBuffer_.getDevicePointer());
    plp_.f = reinterpret_cast<shocker::ShockerPerFramePipelineLaunchParameters*>(
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
        LOG(WARNING, "G-buffer pipeline not ready");
        return;
    }
    
    CUstream stream = renderContext_->getStream();
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
        LOG(WARNING, "Path tracing pipeline not ready");
        return;
    }
    
    CUstream stream = renderContext_->getStream();
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