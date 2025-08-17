#include "ScreenBufferHandler.h"
#include "../RenderContext.h"
#include <g3log/g3log.hpp>
#include <random>

namespace dog
{

ScreenBufferHandler::ScreenBufferHandler(RenderContextPtr ctx)
    : ctx_(ctx)
{
}

ScreenBufferHandler::~ScreenBufferHandler()
{
    finalize();
}

bool ScreenBufferHandler::initialize(uint32_t width, uint32_t height)
{
    if (initialized_)
    {
        LOG(WARNING) << "ScreenBufferHandler already initialized";
        return true;
    }

    if (!ctx_ || !ctx_->getCudaContext())
    {
        LOG(WARNING) << "ScreenBufferHandler: Invalid render context";
        return false;
    }

    width_ = width;
    height_ = height;

    try
    {
        // Initialize G-buffers
        gbuffers_.initialize(ctx_->getCudaContext(), width, height);

        // Initialize accumulation buffers
        accumulation_buffers_.initialize(ctx_->getCudaContext(), width, height);

        // Initialize linear buffers
        linear_buffers_.initialize(ctx_->getCudaContext(), width, height);

        // Initialize RNG buffer
        if (!initializeRngBuffer(width, height))
        {
            LOG(WARNING) << "Failed to initialize RNG buffer";
            finalize();
            return false;
        }

        initialized_ = true;
        LOG(INFO) << "ScreenBufferHandler initialized successfully (" << width << "x" << height << ")";
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to initialize screen buffers: " << ex.what();
        finalize();
        return false;
    }
}

void ScreenBufferHandler::finalize()
{
    if (!initialized_)
    {
        return;
    }

    finalizeRngBuffer();
    linear_buffers_.finalize();
    accumulation_buffers_.finalize();
    gbuffers_.finalize();

    initialized_ = false;
    width_ = 0;
    height_ = 0;

    LOG(DBUG) << "ScreenBufferHandler finalized";
}

void ScreenBufferHandler::resize(uint32_t width, uint32_t height)
{
    if (!initialized_)
    {
        LOG(WARNING) << "ScreenBufferHandler not initialized";
        return;
    }

    width_ = width;
    height_ = height;

    // Resize G-buffers
    gbuffers_.resize(width, height);

    // Resize accumulation buffers
    accumulation_buffers_.resize(width, height);

    // Resize linear buffers
    linear_buffers_.resize(width, height);

    // Resize RNG buffer (requires finalize and reinitialize)
    resizeRngBuffer(width, height);

    LOG(INFO) << "ScreenBufferHandler resized to " << width << "x" << height;
}

bool ScreenBufferHandler::initializeRngBuffer(uint32_t width, uint32_t height)
{
    if (!ctx_ || !ctx_->getCudaContext())
    {
        return false;
    }

    // For RNG buffer, we'll use a simple uint32_t array for now
    // The actual RNG structure will be defined when we create the shared header
    rng_buffer_.initialize2D(
        ctx_->getCudaContext(), cudau::ArrayElementType::UInt32, 2,  // 2 uint32s for simple RNG state
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        width, height, 1);

    // Initialize RNG states with random seeds
    auto rngs = rng_buffer_.map<uint32_t>();
    std::mt19937_64 rngSeed(591842031321323413);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            uint32_t idx = (y * width + x) * 2;
            rngs[idx] = static_cast<uint32_t>(rngSeed());
            rngs[idx + 1] = static_cast<uint32_t>(rngSeed());
        }
    }
    rng_buffer_.unmap();

    return true;
}

void ScreenBufferHandler::finalizeRngBuffer()
{
    rng_buffer_.finalize();
}

void ScreenBufferHandler::resizeRngBuffer(uint32_t width, uint32_t height)
{
    finalizeRngBuffer();
    initializeRngBuffer(width, height);
}

// GBuffers implementation
void ScreenBufferHandler::GBuffers::initialize(CUcontext cuContext, uint32_t width, uint32_t height)
{
    for (int i = 0; i < 2; ++i)
    {
        // For now, use generic sizes - will be updated when we define the actual GBuffer structures
        gBuffer0[i].initialize2D(
            cuContext, cudau::ArrayElementType::UInt32, 4,  // 4 uint32s for GBuffer0
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        gBuffer1[i].initialize2D(
            cuContext, cudau::ArrayElementType::UInt32, 4,  // 4 uint32s for GBuffer1
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
    }
    LOG(DBUG) << "G-buffers initialized";
}

void ScreenBufferHandler::GBuffers::resize(uint32_t width, uint32_t height)
{
    for (int i = 0; i < 2; ++i)
    {
        gBuffer0[i].resize(width, height);
        gBuffer1[i].resize(width, height);
    }
    LOG(DBUG) << "G-buffers resized to " << width << "x" << height;
}

void ScreenBufferHandler::GBuffers::finalize()
{
    for (int i = 1; i >= 0; --i)
    {
        gBuffer1[i].finalize();
        gBuffer0[i].finalize();
    }
    LOG(DBUG) << "G-buffers finalized";
}

// AccumulationBuffers implementation
void ScreenBufferHandler::AccumulationBuffers::initialize(CUcontext cuContext, uint32_t width, uint32_t height)
{
    beautyAccumBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 4,
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        width, height, 1);
    albedoAccumBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 4,
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        width, height, 1);
    normalAccumBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 4,
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        width, height, 1);
    LOG(DBUG) << "Accumulation buffers initialized";
}

void ScreenBufferHandler::AccumulationBuffers::resize(uint32_t width, uint32_t height)
{
    beautyAccumBuffer.resize(width, height);
    albedoAccumBuffer.resize(width, height);
    normalAccumBuffer.resize(width, height);
    LOG(DBUG) << "Accumulation buffers resized to " << width << "x" << height;
}

void ScreenBufferHandler::AccumulationBuffers::finalize()
{
    normalAccumBuffer.finalize();
    albedoAccumBuffer.finalize();
    beautyAccumBuffer.finalize();
    LOG(DBUG) << "Accumulation buffers finalized";
}

// LinearBuffers implementation
void ScreenBufferHandler::LinearBuffers::initialize(CUcontext cuContext, uint32_t width, uint32_t height)
{
    uint32_t bufferSize = width * height;

    // Use default buffer type for now
    linearBeautyBuffer.initialize(cuContext, cudau::BufferType::Device, bufferSize);
    linearAlbedoBuffer.initialize(cuContext, cudau::BufferType::Device, bufferSize);
    linearNormalBuffer.initialize(cuContext, cudau::BufferType::Device, bufferSize);
    linearFlowBuffer.initialize(cuContext, cudau::BufferType::Device, bufferSize);
    linearDenoisedBeautyBuffer.initialize(cuContext, cudau::BufferType::Device, bufferSize);

    LOG(DBUG) << "Linear buffers initialized";
}

void ScreenBufferHandler::LinearBuffers::resize(uint32_t width, uint32_t height)
{
    uint32_t bufferSize = width * height;

    linearBeautyBuffer.resize(bufferSize);
    linearAlbedoBuffer.resize(bufferSize);
    linearNormalBuffer.resize(bufferSize);
    linearFlowBuffer.resize(bufferSize);
    linearDenoisedBeautyBuffer.resize(bufferSize);

    LOG(DBUG) << "Linear buffers resized to " << width << "x" << height;
}

void ScreenBufferHandler::LinearBuffers::finalize()
{
    linearDenoisedBeautyBuffer.finalize();
    linearFlowBuffer.finalize();
    linearNormalBuffer.finalize();
    linearAlbedoBuffer.finalize();
    linearBeautyBuffer.finalize();
    LOG(DBUG) << "Linear buffers finalized";
}

} // namespace dog