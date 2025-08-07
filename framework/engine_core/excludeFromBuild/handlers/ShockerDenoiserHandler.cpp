#include "ShockerDenoiserHandler.h"
#include "../RenderContext.h"

// Factory method
ShockerDenoiserHandlerPtr ShockerDenoiserHandler::create(RenderContextPtr renderContext)
{
    return std::make_shared<ShockerDenoiserHandler>(renderContext);
}

// Constructor
ShockerDenoiserHandler::ShockerDenoiserHandler(RenderContextPtr renderContext) :
    renderContext_(renderContext)
{
    if (renderContext_)
    {
        optixContext_ = renderContext_->getOptiXContext();
        cudaContext_ = renderContext_->getCudaContext();
    }
}

// Destructor
ShockerDenoiserHandler::~ShockerDenoiserHandler()
{
    finalize();
}

// Initialize denoiser with specified dimensions and model type
bool ShockerDenoiserHandler::initialize(uint32_t width, uint32_t height, bool useTemporalDenoiser)
{
    if (initialized_)
    {
        LOG(DBUG) << "ShockerDenoiserHandler already initialized";
        return true;
    }

    if (!renderContext_ || !renderContext_->isInitialized())
    {
        LOG(WARNING) << "ShockerDenoiserHandler: RenderContext not initialized";
        return false;
    }

    width_ = width;
    height_ = height;
    isTemporalDenoiser_ = useTemporalDenoiser;

    try
    {
        if (!createDenoiser(useTemporalDenoiser))
        {
            LOG(WARNING) << "Failed to create Shocker denoiser";
            return false;
        }

        if (!setupBuffersAndTasks(width, height))
        {
            LOG(WARNING) << "Failed to setup Shocker denoiser buffers and tasks";
            denoiser_.destroy();
            return false;
        }

        needsStateSetup_ = true;
        initialized_ = true;
        LOG(INFO) << "ShockerDenoiserHandler initialized successfully (" << width << "x" << height 
                  << ", " << (useTemporalDenoiser ? "temporal" : "HDR") << ")";
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to initialize Shocker denoiser: " << ex.what();
        finalize();
        return false;
    }
}

// Clean up denoiser resources
void ShockerDenoiserHandler::finalize()
{
    if (!initialized_)
    {
        return;
    }

    tasks_.clear();
    
    if (scratchBuffer_.isInitialized())
    {
        scratchBuffer_.finalize();
    }
    
    if (stateBuffer_.isInitialized())
    {
        stateBuffer_.finalize();
    }
    
    denoiser_.destroy();

    initialized_ = false;
    width_ = 0;
    height_ = 0;
    needsStateSetup_ = false;

    LOG(DBUG) << "ShockerDenoiserHandler finalized";
}

// Resize denoiser for new dimensions
void ShockerDenoiserHandler::resize(uint32_t width, uint32_t height)
{
    if (!initialized_)
    {
        LOG(WARNING) << "ShockerDenoiserHandler not initialized";
        return;
    }

    // Skip if dimensions haven't changed
    if (width_ == width && height_ == height)
    {
        return;
    }

    width_ = width;
    height_ = height;

    if (!setupBuffersAndTasks(width, height))
    {
        LOG(WARNING) << "Failed to resize Shocker denoiser buffers and tasks";
        return;
    }

    needsStateSetup_ = true;
    LOG(DBUG) << "ShockerDenoiserHandler resized to " << width << "x" << height;
}

// Switch between temporal and HDR denoising models
void ShockerDenoiserHandler::updateDenoiserType(bool useTemporalDenoiser)
{
    if (!initialized_)
    {
        LOG(WARNING) << "ShockerDenoiserHandler not initialized";
        return;
    }

    if (isTemporalDenoiser_ == useTemporalDenoiser)
    {
        return;
    }

    isTemporalDenoiser_ = useTemporalDenoiser;

    // Destroy current denoiser
    denoiser_.destroy();

    // Create new denoiser with different model
    if (!createDenoiser(useTemporalDenoiser))
    {
        LOG(WARNING) << "Failed to create new Shocker denoiser type";
        return;
    }

    // Re-setup with current dimensions
    if (!setupBuffersAndTasks(width_, height_))
    {
        LOG(WARNING) << "Failed to setup buffers for new Shocker denoiser type";
        return;
    }

    needsStateSetup_ = true;
    LOG(INFO) << "ShockerDenoiserHandler type updated to " << (useTemporalDenoiser ? "temporal" : "HDR");
}

// Setup denoiser state with provided stream
void ShockerDenoiserHandler::setupState(CUstream stream)
{
    if (!initialized_)
    {
        LOG(WARNING) << "ShockerDenoiserHandler not initialized";
        return;
    }

    if (!needsStateSetup_)
    {
        return;
    }

    denoiser_.setupState(stream, stateBuffer_, scratchBuffer_);
    needsStateSetup_ = false;
}

// Create denoiser with specified model type
bool ShockerDenoiserHandler::createDenoiser(bool useTemporalDenoiser)
{
    OptixDenoiserModelKind modelKind = useTemporalDenoiser 
        ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL 
        : OPTIX_DENOISER_MODEL_KIND_HDR;

    // Create denoiser with albedo and normal guides for better quality
    denoiser_ = optixContext_.createDenoiser(
        modelKind,
        optixu::GuideAlbedo::Yes,
        optixu::GuideNormal::Yes,
        OPTIX_DENOISER_ALPHA_MODE_COPY);

    return true;
}

// Setup buffers and tasks for denoising
bool ShockerDenoiserHandler::setupBuffersAndTasks(uint32_t width, uint32_t height)
{
    // Configure for non-tiled denoising (same as sample code)
    constexpr uint32_t tileWidth = 0;
    constexpr uint32_t tileHeight = 0;

    optixu::DenoiserSizes denoiserSizes;
    uint32_t numTasks;
    denoiser_.prepare(width, height, tileWidth, tileHeight, &denoiserSizes, &numTasks);

    LOG(DBUG) << "Shocker Denoiser State Buffer: " << denoiserSizes.stateSize << " bytes";
    LOG(DBUG) << "Shocker Denoiser Scratch Buffer: " << denoiserSizes.scratchSize << " bytes";
    LOG(DBUG) << "Shocker Compute Intensity Scratch Buffer: " << denoiserSizes.scratchSizeForComputeNormalizer << " bytes";

    // Buffer type for device memory allocation
    constexpr cudau::BufferType bufferType = cudau::BufferType::Device;

    // Initialize or resize state buffer
    if (!stateBuffer_.isInitialized())
    {
        stateBuffer_.initialize(cudaContext_, bufferType, denoiserSizes.stateSize, 1);
    }
    else
    {
        stateBuffer_.resize(denoiserSizes.stateSize, 1);
    }

    // Initialize or resize scratch buffer (use max of scratch and compute intensity scratch)
    size_t scratchSize = std::max(denoiserSizes.scratchSize, denoiserSizes.scratchSizeForComputeNormalizer);
    
    if (!scratchBuffer_.isInitialized())
    {
        scratchBuffer_.initialize(cudaContext_, bufferType, scratchSize, 1);
    }
    else
    {
        scratchBuffer_.resize(scratchSize, 1);
    }

    // Setup denoising tasks
    tasks_.resize(numTasks);
    denoiser_.getTasks(tasks_.data());

    return true;
}