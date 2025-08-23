#include "DenoiserHandler.h"
#include "../RenderContext.h"

// Factory method
DenoiserHandlerPtr DenoiserHandler::create(RenderContextPtr renderContext)
{
    return std::make_shared<DenoiserHandler>(renderContext);
}

// Constructor
DenoiserHandler::DenoiserHandler(RenderContextPtr renderContext)
    : renderContext_(renderContext)
{
}

// Destructor
DenoiserHandler::~DenoiserHandler()
{
    finalize();
}

// Initialize denoiser with dimensions and configuration
bool DenoiserHandler::initialize(uint32_t width, uint32_t height, const DenoiserConfig& config)
{
    if (initialized_)
    {
        LOG(WARNING) << "DenoiserHandler already initialized";
        return true;
    }
    
    if (!renderContext_ || !renderContext_->isInitialized())
    {
        LOG(WARNING) << "DenoiserHandler requires initialized RenderContext";
        return false;
    }
    
    width_ = width;
    height_ = height;
    config_ = config;
    
    // Validate and adjust configuration
    validateConfig();
    
    // Set denoised dimensions (2x for upscaling modes)
    denoisedWidth_ = isUpscaleModel(config_.model) ? width * 2 : width;
    denoisedHeight_ = isUpscaleModel(config_.model) ? height * 2 : height;
    
    try
    {
        createDenoiser();
        prepareDenoiser();
        allocateBuffers();
        
        // Setup state once after initialization
        CUstream stream = renderContext_->getCudaStream();
        denoiser_.setupState(stream, stateBuffer_, scratchBuffer_);
        
        initialized_ = true;
        
        LOG(INFO) << "DenoiserHandler initialized successfully"
                  << " (input: " << width_ << "x" << height_
                  << ", output: " << denoisedWidth_ << "x" << denoisedHeight_ << ")"
                  << ", model: " << static_cast<int>(config_.model)
                  << ", temporal: " << isTemporalModel(config_.model)
                  << ", upscale: " << isUpscaleModel(config_.model);
        
        return true;
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to initialize DenoiserHandler: " << e.what();
        deallocateBuffers();
        destroyDenoiser();
        return false;
    }
}

// Finalize and release resources
void DenoiserHandler::finalize()
{
    if (!initialized_)
        return;
    
    deallocateBuffers();
    destroyDenoiser();
    
    initialized_ = false;
    
    LOG(INFO) << "DenoiserHandler finalized";
}

// Resize denoiser for new dimensions
void DenoiserHandler::resize(uint32_t width, uint32_t height)
{
    if (!initialized_)
    {
        LOG(WARNING) << "Cannot resize - DenoiserHandler not initialized";
        return;
    }
    
    if (width_ == width && height_ == height)
    {
        LOG(INFO) << "DenoiserHandler resize - dimensions unchanged";
        return;
    }
    
    LOG(INFO) << "Resizing DenoiserHandler from " << width_ << "x" << height_
              << " to " << width << "x" << height;
    
    // Update dimensions
    width_ = width;
    height_ = height;
    denoisedWidth_ = isUpscaleModel(config_.model) ? width * 2 : width;
    denoisedHeight_ = isUpscaleModel(config_.model) ? height * 2 : height;
    
    // Recreate denoiser with new dimensions
    deallocateBuffers();
    destroyDenoiser();
    createDenoiser();
    prepareDenoiser();
    allocateBuffers();
    
    // Setup state once after resize
    CUstream stream = renderContext_->getCudaStream();
    denoiser_.setupState(stream, stateBuffer_, scratchBuffer_);
}

// Update configuration
void DenoiserHandler::updateConfig(const DenoiserConfig& config)
{
    if (!initialized_)
    {
        config_ = config;
        return;
    }
    
    // Check if model changed
    bool modelChanged = (config.model != config_.model);
    
    config_ = config;
    validateConfig();
    
    if (modelChanged)
    {
        LOG(INFO) << "DenoiserHandler model changed, recreating denoiser";
        
        // Update denoised dimensions
        denoisedWidth_ = isUpscaleModel(config_.model) ? width_ * 2 : width_;
        denoisedHeight_ = isUpscaleModel(config_.model) ? height_ * 2 : height_;
        
        // Recreate denoiser with new model
        deallocateBuffers();
        destroyDenoiser();
        createDenoiser();
        prepareDenoiser();
        allocateBuffers();
        
        // Setup state once after model change
        CUstream stream = renderContext_->getCudaStream();
        denoiser_.setupState(stream, stateBuffer_, scratchBuffer_);
    }
}

// Set denoiser model
void DenoiserHandler::setModel(DenoiserConfig::Model model)
{
    DenoiserConfig newConfig = config_;
    newConfig.model = model;
    updateConfig(newConfig);
}

// Enable/disable temporal mode
void DenoiserHandler::enableTemporalMode(bool enable)
{
    DenoiserConfig newConfig = config_;
    
    if (enable)
    {
        // Switch to temporal variant
        if (config_.model == DenoiserConfig::Model::HDR)
            newConfig.model = DenoiserConfig::Model::Temporal;
        else if (config_.model == DenoiserConfig::Model::AOV)
            newConfig.model = DenoiserConfig::Model::TemporalAOV;
        else if (config_.model == DenoiserConfig::Model::Upscale2X)
            newConfig.model = DenoiserConfig::Model::TemporalUpscale2X;
    }
    else
    {
        // Switch to spatial variant
        if (config_.model == DenoiserConfig::Model::Temporal)
            newConfig.model = DenoiserConfig::Model::HDR;
        else if (config_.model == DenoiserConfig::Model::TemporalAOV)
            newConfig.model = DenoiserConfig::Model::AOV;
        else if (config_.model == DenoiserConfig::Model::TemporalUpscale2X)
            newConfig.model = DenoiserConfig::Model::Upscale2X;
    }
    
    updateConfig(newConfig);
}

// Enable/disable kernel prediction
void DenoiserHandler::enableKernelPrediction(bool enable)
{
    DenoiserConfig newConfig = config_;
    newConfig.useKernelPrediction = enable;
    
    if (enable)
    {
        // Switch to kernel prediction variant
        if (config_.model == DenoiserConfig::Model::HDR)
            newConfig.model = DenoiserConfig::Model::AOV;
        else if (config_.model == DenoiserConfig::Model::Temporal)
            newConfig.model = DenoiserConfig::Model::TemporalAOV;
    }
    else
    {
        // Switch to non-kernel prediction variant
        if (config_.model == DenoiserConfig::Model::AOV)
            newConfig.model = DenoiserConfig::Model::HDR;
        else if (config_.model == DenoiserConfig::Model::TemporalAOV)
            newConfig.model = DenoiserConfig::Model::Temporal;
    }
    
    updateConfig(newConfig);
}

// Enable/disable upscaling
void DenoiserHandler::enableUpscaling(bool enable)
{
    DenoiserConfig newConfig = config_;
    newConfig.performUpscale = enable;
    
    if (enable)
    {
        // Switch to upscale variant
        if (config_.model == DenoiserConfig::Model::HDR ||
            config_.model == DenoiserConfig::Model::AOV)
            newConfig.model = DenoiserConfig::Model::Upscale2X;
        else if (config_.model == DenoiserConfig::Model::Temporal ||
                 config_.model == DenoiserConfig::Model::TemporalAOV)
            newConfig.model = DenoiserConfig::Model::TemporalUpscale2X;
    }
    else
    {
        // Switch to non-upscale variant
        if (config_.model == DenoiserConfig::Model::Upscale2X)
            newConfig.model = DenoiserConfig::Model::HDR;
        else if (config_.model == DenoiserConfig::Model::TemporalUpscale2X)
            newConfig.model = DenoiserConfig::Model::Temporal;
    }
    
    updateConfig(newConfig);
}


// Compute HDR normalizer
void DenoiserHandler::computeNormalizer(
    CUstream stream,
    const DenoiserInputBuffers& inputs)
{
    if (!initialized_)
    {
        LOG(WARNING) << "Cannot compute normalizer - DenoiserHandler not initialized";
        return;
    }
    
    if (!inputs.validate(config_))
    {
        LOG(WARNING) << "Invalid input buffers for denoiser configuration";
        return;
    }
    
    try
    {
        denoiser_.computeNormalizer(
            stream,
            *inputs.noisyBeauty, inputs.beautyFormat,
            scratchBuffer_, hdrNormalizer_.getCUdeviceptr());
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to compute HDR normalizer: " << e.what();
    }
}

// Execute denoising
void DenoiserHandler::denoise(
    CUstream stream,
    const DenoiserInputBuffers& inputs,
    const DenoiserOutputBuffers& outputs,
    bool isFirstFrame,
    float blendFactor)
{
    if (!initialized_)
    {
        LOG(WARNING) << "Cannot denoise - DenoiserHandler not initialized";
        return;
    }
    
    if (!inputs.validate(config_))
    {
        LOG(WARNING) << "Invalid input buffers for denoiser configuration";
        return;
    }
    
    if (!outputs.denoisedBeauty)
    {
        LOG(WARNING) << "No output buffer provided for denoising";
        return;
    }
    
    try
    {
        // Setup OptiX denoiser input structure
        optixu::DenoiserInputBuffers optixInputs = {};  // Zero-initialize all fields
        optixInputs.beautyFormat = inputs.beautyFormat;
        optixInputs.albedoFormat = inputs.albedoFormat;
        optixInputs.normalFormat = inputs.normalFormat;
        optixInputs.flowFormat = inputs.flowFormat;
        
        // Explicitly set AOV-related fields to nullptr (critical for HDR model)
        optixInputs.noisyAovs = nullptr;
        optixInputs.previousDenoisedAovs = nullptr;
        optixInputs.aovFormats = nullptr;
        optixInputs.aovTypes = nullptr;
        optixInputs.numAovs = 0;
        
        optixInputs.noisyBeauty = *inputs.noisyBeauty;
        
        if (config_.useAlbedo && inputs.albedo)
            optixInputs.albedo = *inputs.albedo;
        
        if (config_.useNormal && inputs.normal)
            optixInputs.normal = *inputs.normal;
        
        if (config_.useFlow && inputs.flow && isTemporalModel(config_.model))
            optixInputs.flow = *inputs.flow;
        
        // Set temporal buffers
        if (isTemporalModel(config_.model) && inputs.previousDenoisedBeauty && !isFirstFrame)
        {
            optixInputs.previousDenoisedBeauty = *inputs.previousDenoisedBeauty;
        }
        else if (isTemporalModel(config_.model))
        {
            // Use noisy beauty as previous for first frame
            optixInputs.previousDenoisedBeauty = *inputs.noisyBeauty;
        }
        
        // Set internal guide layer for kernel prediction
        if (isKernelPredictionModel(config_.model) && isTemporalModel(config_.model))
        {
            if (!isFirstFrame)
            {
                optixInputs.previousInternalGuideLayer = inputs.previousInternalGuideLayer;
            }
            
            // outputs.internalGuideLayerForNextFrame will be filled by the denoiser
        }
        
        // Refresh tasks before invoking - required as denoiser state can be invalidated
        denoiser_.getTasks(denoisingTasks_.data());
        
        // Execute denoising for all tasks
        for (const auto& task : denoisingTasks_)
        {
            // For HDR model with guide layers, we pass nullptr for AOV outputs
            // The production code shows this pattern works correctly
            denoiser_.invoke(
                stream, task,
                optixInputs, optixu::IsFirstFrame(isFirstFrame),
                hdrNormalizer_.getCUdeviceptr(), blendFactor,
                *outputs.denoisedBeauty,  // Pass the dereferenced buffer directly
                nullptr,  // No AOV outputs for HDR model
                optixu::BufferView());  // Empty BufferView for internal guide layer
        }
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to execute denoising: " << e.what();
    }
}

// Combined compute normalizer and denoise
void DenoiserHandler::computeAndDenoise(
    CUstream stream,
    const DenoiserInputBuffers& inputs,
    const DenoiserOutputBuffers& outputs,
    bool isFirstFrame,
    float blendFactor)
{
    computeNormalizer(stream, inputs);
    denoise(stream, inputs, outputs, isFirstFrame, blendFactor);
}

// Tiled denoising for large framebuffers
void DenoiserHandler::denoiseTiled(
    CUstream stream,
    const DenoiserInputBuffers& inputs,
    const DenoiserOutputBuffers& outputs,
    bool isFirstFrame)
{
    if (!config_.useTiling)
    {
        // Fall back to regular denoising
        denoise(stream, inputs, outputs, isFirstFrame);
        return;
    }
    
    // Tiled denoising would iterate through tiles
    // This is a simplified version - full implementation would handle tile overlaps
    LOG(INFO) << "Executing tiled denoising with " << denoisingTasks_.size() << " tiles";
    
    computeNormalizer(stream, inputs);
    
    for (const auto& task : denoisingTasks_)
    {
        // Each task represents a tile
        // In a full implementation, we'd offset buffer pointers for each tile
        denoise(stream, inputs, outputs, isFirstFrame);
    }
}

// Get internal guide layer buffer
cudau::Buffer& DenoiserHandler::getInternalGuideLayer(uint32_t index)
{
    if (index >= 2)
        throw std::out_of_range("Internal guide layer index out of range");
    
    return internalGuideLayers_[index];
}

// Create OptiX denoiser
void DenoiserHandler::createDenoiser()
{
    if (!renderContext_)
        throw std::runtime_error("No RenderContext available");
    
    modelKind_ = getOptixModelKind(config_.model);
    
    // Determine guide layer usage
    optixu::GuideAlbedo useAlbedo = config_.useAlbedo ? 
        optixu::GuideAlbedo::Yes : optixu::GuideAlbedo::No;
    optixu::GuideNormal useNormal = config_.useNormal ? 
        optixu::GuideNormal::Yes : optixu::GuideNormal::No;
    
    // Create denoiser
    denoiser_ = renderContext_->getOptiXContext().createDenoiser(
        modelKind_, useAlbedo, useNormal, config_.alphaMode);
    
    LOG(DBUG) << "Created OptiX denoiser with model: " << static_cast<int>(modelKind_)
              << ", albedo: " << (config_.useAlbedo ? "yes" : "no")
              << ", normal: " << (config_.useNormal ? "yes" : "no");
}

// Destroy OptiX denoiser
void DenoiserHandler::destroyDenoiser()
{
    denoisingTasks_.clear();
    
    if (denoiser_)
    {
        denoiser_.destroy();
        LOG(INFO) << "Destroyed OptiX denoiser";
    }
}

// Prepare denoiser and get buffer sizes
void DenoiserHandler::prepareDenoiser()
{
    if (!renderContext_)
        throw std::runtime_error("No RenderContext available");
    
    // Prepare denoiser and get buffer sizes
    optixu::DenoiserSizes denoiserSizes;
    uint32_t numTasks;
    
    uint32_t tileWidth = config_.useTiling ? config_.tileWidth : 0;
    uint32_t tileHeight = config_.useTiling ? config_.tileHeight : 0;
    
    denoiser_.prepare(
        width_, height_, tileWidth, tileHeight,
        &denoiserSizes, &numTasks);
    
    // Store buffer sizes
    stateBufferSize_ = denoiserSizes.stateSize;
    scratchBufferSize_ = denoiserSizes.scratchSize;
    normalizerScratchSize_ = denoiserSizes.scratchSizeForComputeNormalizer;
    internalGuideLayerPixelSize_ = denoiserSizes.internalGuideLayerPixelSize;
    
    // Log buffer sizes
    LOG(DBUG) << "Denoiser buffer sizes:"
              << " State: " << (stateBufferSize_ / (1024 * 1024)) << " MB"
              << ", Scratch: " << (scratchBufferSize_ / (1024 * 1024)) << " MB"
              << ", Normalizer Scratch: " << (normalizerScratchSize_ / (1024 * 1024)) << " MB"
              << ", Internal Guide Layer Pixel Size: " << internalGuideLayerPixelSize_ << " bytes"
              << ", Number of tasks: " << numTasks;
    
    // Setup denoising tasks
    denoisingTasks_.resize(numTasks);
    denoiser_.getTasks(denoisingTasks_.data());
}

// Allocate denoiser buffers
void DenoiserHandler::allocateBuffers()
{
    CUcontext cuContext = renderContext_->getCudaContext();
    
    // Allocate state buffer
    stateBuffer_.initialize(cuContext, cudau::BufferType::Device, stateBufferSize_, 1);
    
    // Allocate scratch buffer (use max of denoise and normalizer scratch sizes)
    size_t maxScratchSize = std::max(scratchBufferSize_, normalizerScratchSize_);
    scratchBuffer_.initialize(cuContext, cudau::BufferType::Device, maxScratchSize, 1);
    
    // Allocate HDR normalizer buffer
    optixu::DenoiserSizes denoiserSizes;
    uint32_t numTasks;
    denoiser_.prepare(width_, height_, 0, 0, &denoiserSizes, &numTasks);
    hdrNormalizer_.initialize(cuContext, cudau::BufferType::Device, denoiserSizes.normalizerSize, 1);
    
    // Allocate internal guide layers for temporal + kernel prediction
    if (internalGuideLayerPixelSize_ > 0 && isKernelPredictionModel(config_.model))
    {
        internalGuideLayerSize_ = denoisedWidth_ * denoisedHeight_ * internalGuideLayerPixelSize_;
        
        for (int i = 0; i < 2; ++i)
        {
            internalGuideLayers_[i].initialize(
                cuContext, cudau::BufferType::Device,
                denoisedWidth_ * denoisedHeight_, internalGuideLayerPixelSize_);
        }
        
        LOG(INFO) << "Allocated internal guide layers: " 
                  << (internalGuideLayerSize_ * 2 / (1024 * 1024)) << " MB total";
    }
    
    LOG(INFO) << "Allocated denoiser buffers:"
              << " State: " << (stateBufferSize_ / (1024 * 1024)) << " MB"
              << ", Scratch: " << (maxScratchSize / (1024 * 1024)) << " MB"
              << ", HDR Normalizer: " << (denoiserSizes.normalizerSize / 1024) << " KB";
}

// Deallocate denoiser buffers
void DenoiserHandler::deallocateBuffers()
{
    // Clean up in reverse order
    for (int i = 1; i >= 0; --i)
    {
        if (internalGuideLayers_[i].isInitialized())
            internalGuideLayers_[i].finalize();
    }
    
    if (hdrNormalizer_.isInitialized())
        hdrNormalizer_.finalize();
    
    if (scratchBuffer_.isInitialized())
        scratchBuffer_.finalize();
    
    if (stateBuffer_.isInitialized())
        stateBuffer_.finalize();
    
    LOG(INFO) << "Deallocated denoiser buffers";
}

// Validate configuration consistency
void DenoiserHandler::validateConfig()
{
    // Ensure upscale models have upscaling enabled
    if (isUpscaleModel(config_.model))
    {
        config_.performUpscale = true;
    }
    
    // Ensure kernel prediction models have the flag set
    if (isKernelPredictionModel(config_.model))
    {
        config_.useKernelPrediction = true;
    }
    
    // Temporal models need flow vectors
    if (isTemporalModel(config_.model))
    {
        config_.useFlow = true;
    }
    
    // Validate tiling parameters
    if (config_.useTiling)
    {
        if (config_.tileWidth == 0 || config_.tileHeight == 0)
        {
            LOG(WARNING) << "Invalid tile dimensions, disabling tiling";
            config_.useTiling = false;
        }
        else
        {
            // Ensure tile dimensions are reasonable
            config_.tileWidth = std::min(config_.tileWidth, width_);
            config_.tileHeight = std::min(config_.tileHeight, height_);
        }
    }
}

// Validate input buffers
bool DenoiserInputBuffers::validate(const DenoiserConfig& config) const
{
    // Beauty buffer is always required
    if (!noisyBeauty)
    {
        LOG(WARNING) << "Noisy beauty buffer is required for denoising";
        return false;
    }
    
    // Check guide layers based on configuration
    if (config.useAlbedo && !albedo)
    {
        LOG(WARNING) << "Albedo buffer required but not provided";
        return false;
    }
    
    if (config.useNormal && !normal)
    {
        LOG(WARNING) << "Normal buffer required but not provided";
        return false;
    }
    
    // Temporal models require flow vectors
    if (config.model == DenoiserConfig::Model::Temporal ||
        config.model == DenoiserConfig::Model::TemporalAOV ||
        config.model == DenoiserConfig::Model::TemporalUpscale2X)
    {
        if (config.useFlow && !flow)
        {
            LOG(WARNING) << "Flow buffer required for temporal model but not provided";
            return false;
        }
    }
    
    return true;
}

// High-level denoising that automatically handles buffer setup from ScreenBufferHandler
void DenoiserHandler::denoiseFrame(CUstream stream, bool isFirstFrame, float blendFactor)
{
    if (!initialized_)
    {
        LOG(WARNING) << "DenoiserHandler not initialized";
        return;
    }
    
    if (!renderContext_)
    {
        LOG(WARNING) << "RenderContext not available";
        return;
    }
    
    // Get the ScreenBufferHandler from the centralized handlers
    auto& handlers = renderContext_->getHandlers();
    if (!handlers.screenBufferHandler)
    {
        LOG(WARNING) << "ScreenBufferHandler not available";
        return;
    }
    
    auto screenBufferHandler = handlers.screenBufferHandler;
    if (!screenBufferHandler->isInitialized())
    {
        LOG(WARNING) << "ScreenBufferHandler not initialized";
        return;
    }
    
    // Set up input buffers based on denoiser configuration
    DenoiserInputBuffers inputs;
    inputs.noisyBeauty = &screenBufferHandler->getLinearBeautyBuffer();
    inputs.beautyFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    
    // Add guide layers based on configuration
    if (config_.useAlbedo)
    {
        inputs.albedo = &screenBufferHandler->getLinearAlbedoBuffer();
        inputs.albedoFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    }
    
    if (config_.useNormal)
    {
        inputs.normal = &screenBufferHandler->getLinearNormalBuffer();
        inputs.normalFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    }
    
    // Handle temporal/flow based on model type
    if (isTemporalModel(config_.model) && config_.useFlow)
    {
        // For temporal models, flow vectors would be set here if available
        // inputs.flow = &screenBufferHandler->getFlowBuffer();
        // inputs.flowFormat = OPTIX_PIXEL_FORMAT_FLOAT2;
        
        // For now, temporal models without flow will fallback to spatial
        inputs.flow = nullptr;
        inputs.previousDenoisedBeauty = nullptr;
        
        LOG(DBUG) << "Temporal model configured but flow vectors not available, using spatial denoising";
    }
    
    // Set up output buffer
    DenoiserOutputBuffers outputs;
    outputs.denoisedBeauty = &screenBufferHandler->getLinearDenoisedBeautyBuffer();
    
    // AOV outputs for kernel prediction modes (if configured)
    if (config_.useKernelPrediction)
    {
        // These would be set if we had separate denoised AOV buffers
        outputs.denoisedAlbedo = nullptr;
        outputs.denoisedNormal = nullptr;
    }
    
    // Execute combined normalizer computation and denoising
    computeAndDenoise(stream, inputs, outputs, isFirstFrame, blendFactor);
}