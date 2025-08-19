#include "ShockerDenoiserHandler.h"
#include "../../../RenderContext.h"

// Factory method following render_core pattern
ShockerDenoiserHandlerPtr ShockerDenoiserHandler::create(RenderContextPtr renderContext)
{
    return std::make_shared<ShockerDenoiserHandler>(renderContext);
}

ShockerDenoiserHandler::ShockerDenoiserHandler(RenderContextPtr renderContext)
    : renderContext_(renderContext)
{
}

ShockerDenoiserHandler::~ShockerDenoiserHandler()
{
    finalize();
}

bool ShockerDenoiserHandler::initialize(uint32_t width, uint32_t height, bool useTemporalDenoiser)
{
    if (initialized_)
    {
        LOG(WARNING) << "ShockerDenoiserHandler already initialized";
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
        
        LOG(INFO) << "ShockerDenoiserHandler initialized successfully (" << width << "x" << height << ")";
        return true;
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to initialize ShockerDenoiserHandler: " << e.what();
        destroyDenoiser();
        return false;
    }
}

void ShockerDenoiserHandler::finalize()
{
    if (!initialized_)
        return;
        
    destroyDenoiser();
    initialized_ = false;
    
    LOG(INFO) << "ShockerDenoiserHandler finalized";
}

void ShockerDenoiserHandler::resize(uint32_t width, uint32_t height)
{
    if (!initialized_)
    {
        LOG(WARNING) << "Cannot resize - ShockerDenoiserHandler not initialized";
        return;
    }
    
    if (width_ == width && height_ == height)
    {
        LOG(INFO) << "ShockerDenoiserHandler resize - dimensions unchanged";
        return;
    }
    
    LOG(INFO) << "Resizing ShockerDenoiserHandler from " << width_ << "x" << height_ 
              << " to " << width << "x" << height;
    
    // Update dimensions
    width_ = width;
    height_ = height;
    
    // Recreate denoiser with new dimensions
    destroyDenoiser();
    createDenoiser();
    prepareDenoiser();
}

void ShockerDenoiserHandler::updateDenoiserType(bool useTemporalDenoiser)
{
    if (useTemporalDenoiser_ == useTemporalDenoiser)
        return;
        
    LOG(INFO) << "Updating ShockerDenoiserHandler type to " << (useTemporalDenoiser ? "temporal" : "spatial");
    
    useTemporalDenoiser_ = useTemporalDenoiser;
    
    if (initialized_)
    {
        // Recreate denoiser with new type
        destroyDenoiser();
        createDenoiser();
        prepareDenoiser();
    }
}

void ShockerDenoiserHandler::setupState(CUstream stream)
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
        LOG(WARNING) << "Failed to setup ShockerDenoiserHandler state: " << e.what();
    }
}

void ShockerDenoiserHandler::createDenoiser()
{
    if (!renderContext_)
        throw std::runtime_error("No RenderContext available");
    
    // Create denoiser specifically for Shocker's needs
    // Shocker supports temporal denoising with flow vectors
    OptixDenoiserModelKind modelKind = useTemporalDenoiser_ ? 
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
        
    denoiser_ = renderContext_->getOptiXContext().createDenoiser(
        modelKind, useAlbedo_, useNormal_, OPTIX_DENOISER_ALPHA_MODE_COPY);
    
    LOG(INFO) << "Created Shocker " << (useTemporalDenoiser_ ? "temporal" : "HDR") << " denoiser";
}

void ShockerDenoiserHandler::destroyDenoiser()
{
    // Clean up in reverse order
    denoisingTasks_.clear();
    
    if (scratchBuffer_.isInitialized())
        scratchBuffer_.finalize();
    if (stateBuffer_.isInitialized())
        stateBuffer_.finalize();
        
    // Denoiser destroy is handled by its destructor
}

void ShockerDenoiserHandler::prepareDenoiser()
{
    if (!renderContext_)
        throw std::runtime_error("No RenderContext available");
    
    // Prepare denoiser and get buffer sizes
    optixu::DenoiserSizes denoiserSizes;
    uint32_t numTasks;
    denoiser_.prepare(
        width_, height_, tileWidth_, tileHeight_,
        &denoiserSizes, &numTasks);
    
    // Log buffer sizes with Shocker prefix
    LOG(INFO) << "Shocker Denoiser State Buffer: " << (denoiserSizes.stateSize / (1024 * 1024)) << " MB";
    LOG(INFO) << "Shocker Denoiser Scratch Buffer: " << (denoiserSizes.scratchSize / (1024 * 1024)) << " MB";
    LOG(INFO) << "Shocker Compute Normalizer Scratch Buffer: " << (denoiserSizes.scratchSizeForComputeNormalizer / (1024 * 1024)) << " MB";
    LOG(INFO) << "Shocker Number of denoising tasks: " << numTasks;
    
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