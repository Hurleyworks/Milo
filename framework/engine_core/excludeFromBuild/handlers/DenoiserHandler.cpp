#include "DenoiserHandler.h"
#include "../RenderContext.h"

// Factory method following render_core pattern
DenoiserHandlerPtr DenoiserHandler::create(RenderContextPtr renderContext)
{
    return std::make_shared<DenoiserHandler>(renderContext);
}

DenoiserHandler::DenoiserHandler(RenderContextPtr renderContext)
    : renderContext_(renderContext)
{
}

DenoiserHandler::~DenoiserHandler()
{
    finalize();
}

bool DenoiserHandler::initialize(uint32_t width, uint32_t height, bool useTemporalDenoiser)
{
    if (initialized_)
    {
        LOG(WARNING) << "DenoiserHandler already initialized";
        return true;
    }
    
    width_ = width;
    height_ = height;
    useTemporalDenoiser_ = useTemporalDenoiser;
    
    try
    {
        createDenoiser();
        prepareDenoiser();
        initialized_ = true;
        
        LOG(INFO) << "DenoiserHandler initialized successfully (" << width << "x" << height << ")";
        return true;
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to initialize DenoiserHandler: " << e.what();
        destroyDenoiser();
        return false;
    }
}

void DenoiserHandler::finalize()
{
    if (!initialized_)
        return;
        
    destroyDenoiser();
    initialized_ = false;
    
    LOG(INFO) << "DenoiserHandler finalized";
}

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
    
    // Recreate denoiser with new dimensions
    destroyDenoiser();
    createDenoiser();
    prepareDenoiser();
}

void DenoiserHandler::updateDenoiserType(bool useTemporalDenoiser)
{
    if (useTemporalDenoiser_ == useTemporalDenoiser)
        return;
        
    LOG(INFO) << "Updating denoiser type to " << (useTemporalDenoiser ? "temporal" : "spatial");
    
    useTemporalDenoiser_ = useTemporalDenoiser;
    
    if (initialized_)
    {
        // Recreate denoiser with new type
        destroyDenoiser();
        createDenoiser();
        prepareDenoiser();
    }
}

void DenoiserHandler::setupState(CUstream stream)
{
    if (!initialized_)
        return;
        
    try
    {
        denoiser_.setupState(stream, stateBuffer_, scratchBuffer_);
        CUDADRV_CHECK(cuStreamSynchronize(stream));
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to setup DenoiserHandler state: " << e.what();
    }
}

void DenoiserHandler::createDenoiser()
{
    if (!renderContext_)
        throw std::runtime_error("No RenderContext available");
    
    // Create denoiser following Shocker's approach
    OptixDenoiserModelKind modelKind = useTemporalDenoiser_ ? 
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
        
    denoiser_ = renderContext_->getOptiXContext().createDenoiser(
        modelKind, useAlbedo_, useNormal_, OPTIX_DENOISER_ALPHA_MODE_COPY);
    
    LOG(INFO) << "Created " << (useTemporalDenoiser_ ? "temporal" : "HDR") << " denoiser";
}

void DenoiserHandler::destroyDenoiser()
{
    // Clean up in reverse order
    denoisingTasks_.clear();
    
    if (scratchBuffer_.isInitialized())
        scratchBuffer_.finalize();
    if (stateBuffer_.isInitialized())
        stateBuffer_.finalize();
        
    // Denoiser destroy is handled by its destructor
    // denoiser_.destroy();
}

void DenoiserHandler::prepareDenoiser()
{
    // TODO: Check if denoiser is valid
    // if (!denoiser_)
    //     throw std::runtime_error("Denoiser not created");
    if (!renderContext_)
        throw std::runtime_error("No RenderContext available");
    
    // Prepare denoiser and get buffer sizes
    optixu::DenoiserSizes denoiserSizes;
    uint32_t numTasks;
    denoiser_.prepare(
        width_, height_, tileWidth_, tileHeight_,
        &denoiserSizes, &numTasks);
    
    // Log buffer sizes
    LOG(INFO) << "Denoiser State Buffer: " << (denoiserSizes.stateSize / (1024 * 1024)) << " MB";
    LOG(INFO) << "Denoiser Scratch Buffer: " << (denoiserSizes.scratchSize / (1024 * 1024)) << " MB";
    LOG(INFO) << "Compute Normalizer Scratch Buffer: " << (denoiserSizes.scratchSizeForComputeNormalizer / (1024 * 1024)) << " MB";
    LOG(INFO) << "Number of denoising tasks: " << numTasks;
    
    // Initialize buffers
    CUcontext cuContext = renderContext_->getCudaContext();
    stateBuffer_.initialize(cuContext, cudau::BufferType::Device, denoiserSizes.stateSize, 1);
    scratchBuffer_.initialize(
        cuContext, cudau::BufferType::Device,
        std::max(denoiserSizes.scratchSize, denoiserSizes.scratchSizeForComputeNormalizer), 1);
    
    // Setup denoising tasks
    denoisingTasks_.resize(numTasks);
    denoiser_.getTasks(denoisingTasks_.data());
}