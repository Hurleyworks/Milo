#include "DenoiserHandler.h"
#include "../RenderContext.h"
#include "../GPUContext.h"

namespace dog
{

DenoiserHandler::DenoiserHandler(RenderContextPtr ctx)
    : render_context_(ctx)
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

    if (!render_context_)
    {
        LOG(WARNING) << "Invalid render context";
        return false;
    }

    width_ = width;
    height_ = height;
    is_temporal_ = useTemporalDenoiser;

    try
    {
        if (!createDenoiser(useTemporalDenoiser))
        {
            LOG(WARNING) << "Failed to create denoiser";
            return false;
        }

        if (!setupBuffersAndTasks(width, height))
        {
            LOG(WARNING) << "Failed to setup denoiser buffers and tasks";
            denoiser_.destroy();
            return false;
        }

        // Allocate intensity buffer for HDR denoising
        auto cuda_context = render_context_->getGPUContext().getCudaContext();
        intensity_buffer_.initialize(cuda_context, cudau::BufferType::Device, sizeof(float), 1);

        needs_state_setup_ = true;
        initialized_ = true;
        
        LOG(INFO) << "DenoiserHandler initialized successfully (" 
                  << width << "x" << height << ", " 
                  << (useTemporalDenoiser ? "temporal" : "HDR") << ")";
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to initialize denoiser: " << ex.what();
        finalize();
        return false;
    }
}

void DenoiserHandler::finalize()
{
    if (!initialized_)
    {
        return;
    }

    tasks_.clear();
    intensity_buffer_.finalize();
    scratch_buffer_.finalize();
    state_buffer_.finalize();
    denoiser_.destroy();

    initialized_ = false;
    width_ = 0;
    height_ = 0;
    needs_state_setup_ = false;

    LOG(INFO) << "DenoiserHandler finalized";
}

void DenoiserHandler::resize(uint32_t width, uint32_t height)
{
    if (!initialized_)
    {
        LOG(WARNING) << "DenoiserHandler not initialized";
        return;
    }

    width_ = width;
    height_ = height;

    if (!setupBuffersAndTasks(width, height))
    {
        LOG(WARNING) << "Failed to resize denoiser buffers and tasks";
        return;
    }

    needs_state_setup_ = true;
    LOG(INFO) << "DenoiserHandler resized to " << width << "x" << height;
}

void DenoiserHandler::updateDenoiserType(bool useTemporalDenoiser)
{
    if (!initialized_)
    {
        LOG(WARNING) << "DenoiserHandler not initialized";
        return;
    }

    if (is_temporal_ == useTemporalDenoiser)
    {
        return;
    }

    is_temporal_ = useTemporalDenoiser;

    // Destroy current denoiser
    denoiser_.destroy();

    // Create new denoiser with different model
    if (!createDenoiser(useTemporalDenoiser))
    {
        LOG(WARNING) << "Failed to create new denoiser type";
        return;
    }

    // Re-setup with current dimensions
    if (!setupBuffersAndTasks(width_, height_))
    {
        LOG(WARNING) << "Failed to setup buffers for new denoiser type";
        return;
    }

    needs_state_setup_ = true;
    LOG(INFO) << "DenoiserHandler type updated to " << (useTemporalDenoiser ? "temporal" : "HDR");
}

void DenoiserHandler::setupState(CUstream stream)
{
    if (!initialized_)
    {
        LOG(WARNING) << "DenoiserHandler not initialized";
        return;
    }

    denoiser_.setupState(stream, state_buffer_, scratch_buffer_);
    needs_state_setup_ = false;
}

float DenoiserHandler::computeIntensity(CUstream stream,
                                        const cudau::TypedBuffer<float4>& beautyBuffer,
                                        uint32_t width, uint32_t height)
{
    if (!initialized_)
    {
        LOG(WARNING) << "DenoiserHandler not initialized";
        return 1.0f;
    }

    // Compute normalizer (intensity for HDR)
    denoiser_.computeNormalizer(
        stream,
        beautyBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
        scratch_buffer_, reinterpret_cast<CUdeviceptr>(intensity_buffer_.getDevicePointer()));

    // Read back intensity value
    float intensity = 1.0f;
    CUDADRV_CHECK(cuMemcpyDtoH(&intensity, reinterpret_cast<CUdeviceptr>(intensity_buffer_.getDevicePointer()), sizeof(float)));

    return intensity;
}

void DenoiserHandler::denoise(CUstream stream,
                              const cudau::TypedBuffer<float4>& beautyBuffer,
                              const cudau::TypedBuffer<float4>& albedoBuffer,
                              const cudau::TypedBuffer<float4>& normalBuffer,
                              const cudau::TypedBuffer<float2>& flowBuffer,
                              cudau::TypedBuffer<float4>& denoisedBuffer,
                              float blendFactor,
                              bool useTemporalMode)
{
    if (!initialized_)
    {
        LOG(WARNING) << "DenoiserHandler not initialized";
        return;
    }

    if (needs_state_setup_)
    {
        LOG(WARNING) << "Denoiser state needs setup - call setupState first";
        return;
    }

    // For non-tiled denoising, we use a single task
    if (tasks_.empty())
    {
        LOG(WARNING) << "No denoising tasks available";
        return;
    }

    const optixu::DenoisingTask& task = tasks_[0];

    // Setup input buffers
    optixu::DenoiserInputBuffers inputBuffers = {};
    inputBuffers.noisyBeauty = beautyBuffer;
    inputBuffers.beautyFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    
    if (albedoBuffer.isInitialized())
    {
        inputBuffers.albedo = albedoBuffer;
        inputBuffers.albedoFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    }
    
    if (normalBuffer.isInitialized())
    {
        inputBuffers.normal = normalBuffer;
        inputBuffers.normalFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    }
    
    if (is_temporal_ && useTemporalMode && flowBuffer.isInitialized())
    {
        inputBuffers.flow = flowBuffer;
        inputBuffers.flowFormat = OPTIX_PIXEL_FORMAT_FLOAT2;
    }

    // Compute normalizer for HDR denoising
    CUdeviceptr normalizer = 0;
    if (!is_temporal_)
    {
        normalizer = reinterpret_cast<CUdeviceptr>(intensity_buffer_.getDevicePointer());
        denoiser_.computeNormalizer(
            stream,
            beautyBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
            scratch_buffer_, normalizer);
    }

    // Execute denoising
    optixu::IsFirstFrame isFirstFrame = useTemporalMode ? optixu::IsFirstFrame::No : optixu::IsFirstFrame::Yes;
    
    denoiser_.invoke(
        stream, task,
        inputBuffers, isFirstFrame,
        normalizer, blendFactor,
        denoisedBuffer, nullptr,  // No additional AOV outputs
        optixu::BufferView());     // No internal guide layer for next frame
}

bool DenoiserHandler::createDenoiser(bool useTemporalDenoiser)
{
    if (!render_context_)
    {
        LOG(WARNING) << "Invalid render context";
        return false;
    }

    auto optix_context = render_context_->getGPUContext().getOptixContext();
    
    OptixDenoiserModelKind modelKind = useTemporalDenoiser ? 
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;

    denoiser_ = optix_context.createDenoiser(
        modelKind,
        optixu::GuideAlbedo::Yes,
        optixu::GuideNormal::Yes,
        OPTIX_DENOISER_ALPHA_MODE_COPY);

    return true;
}

bool DenoiserHandler::setupBuffersAndTasks(uint32_t width, uint32_t height)
{
    if (!render_context_)
    {
        LOG(WARNING) << "Invalid render context";
        return false;
    }

    auto cuda_context = render_context_->getGPUContext().getCudaContext();

    // Configure for non-tiled denoising
    constexpr uint32_t tileWidth = 0;
    constexpr uint32_t tileHeight = 0;

    optixu::DenoiserSizes denoiserSizes;
    uint32_t numTasks;
    denoiser_.prepare(width, height, tileWidth, tileHeight, &denoiserSizes, &numTasks);

    LOG(DBUG) << "Denoiser State Buffer: " << denoiserSizes.stateSize << " bytes";
    LOG(DBUG) << "Denoiser Scratch Buffer: " << denoiserSizes.scratchSize << " bytes";
    LOG(DBUG) << "Compute Intensity Scratch Buffer: " << denoiserSizes.scratchSizeForComputeNormalizer << " bytes";

    // Initialize or resize buffers
    if (!state_buffer_.isInitialized())
    {
        state_buffer_.initialize(cuda_context, cudau::BufferType::Device, 
                                denoiserSizes.stateSize, 1);
    }
    else
    {
        state_buffer_.resize(denoiserSizes.stateSize, 1);
    }

    size_t scratchSize = std::max(denoiserSizes.scratchSize, 
                                  denoiserSizes.scratchSizeForComputeNormalizer);
    if (!scratch_buffer_.isInitialized())
    {
        scratch_buffer_.initialize(cuda_context, cudau::BufferType::Device, scratchSize, 1);
    }
    else
    {
        scratch_buffer_.resize(scratchSize, 1);
    }

    // Setup tasks
    tasks_.resize(numTasks);
    denoiser_.getTasks(tasks_.data());

    return true;
}

// Note: prepareDenoiserInputs is no longer needed with the new API structure
// Input buffers are prepared directly in the denoise() method

size_t DenoiserHandler::getStateBufferSize() const
{
    return state_buffer_.isInitialized() ? state_buffer_.sizeInBytes() : 0;
}

size_t DenoiserHandler::getScratchBufferSize() const
{
    return scratch_buffer_.isInitialized() ? scratch_buffer_.sizeInBytes() : 0;
}

size_t DenoiserHandler::getTotalMemoryUsage() const
{
    return getStateBufferSize() + getScratchBufferSize() + 
           (intensity_buffer_.isInitialized() ? intensity_buffer_.sizeInBytes() : 0);
}

} // namespace dog