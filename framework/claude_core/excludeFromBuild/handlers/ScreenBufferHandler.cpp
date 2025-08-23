#include "ScreenBufferHandler.h"
#include "../tools/PTXManager.h"
#include "../common/common_shared.h"

ScreenBufferHandler::ScreenBufferHandler(RenderContextPtr ctx)
    : renderContext_(ctx)
{
    if (renderContext_ && renderContext_->isInitialized())
    {
        cudaContext_ = renderContext_->getCudaContext();
    }
}

ScreenBufferHandler::~ScreenBufferHandler()
{
    finalize();
}

bool ScreenBufferHandler::initialize(uint32_t width, uint32_t height)
{
    if (!validateDimensions(width, height))
    {
        LOG(WARNING) << "Invalid dimensions: " << width << "x" << height;
        return false;
    }

    if (initialized_ && width_ == width && height_ == height)
    {
        LOG(DBUG) << "ScreenBufferHandler already initialized with same parameters";
        return true;
    }

    // Clean up if switching dimensions
    if (initialized_)
    {
        finalize();
    }

    if (!renderContext_ || !renderContext_->isInitialized())
    {
        LOG(WARNING) << "RenderContext not initialized";
        return false;
    }

    cudaContext_ = renderContext_->getCudaContext();
    width_ = width;
    height_ = height;
    currentFrameIndex_ = 0;

    try
    {
        LOG(INFO) << "Initializing ScreenBufferHandler (" << width << "x" << height << ")";

        // Initialize G-buffers
        gbuffers_.initialize(cudaContext_, width, height);

        // Initialize accumulation buffers
        accumBuffers_.initialize(cudaContext_, width, height);

        // Initialize linear buffers
        linearBuffers_.initialize(cudaContext_, width, height);

        // Initialize RNG buffer
        if (!initializeRngBuffer(cudaContext_, width, height))
        {
            LOG(WARNING) << "Failed to initialize RNG buffer";
            finalize();
            return false;
        }

        // Initialize pick info buffers (double buffered)
        for (int i = 0; i < 2; ++i)
        {
            pickInfoBuffers_[i].initialize(cudaContext_, cudau::BufferType::Device, 1);
        }

        // Load CUDA kernels for buffer operations
        if (!loadKernels())
        {
            LOG(WARNING) << "Failed to load CUDA kernels";
            finalize();
            return false;
        }

        initialized_ = true;
        LOG(INFO) << "ScreenBufferHandler initialized successfully. GPU memory usage: " 
                  << (getTotalGPUMemoryUsage() / (1024.0 * 1024.0)) << " MB";
        return true;
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Exception during ScreenBufferHandler initialization: " << e.what();
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

    LOG(DBUG) << "Finalizing ScreenBufferHandler";

    cleanupKernels();
    
    // Finalize pick info buffers
    for (int i = 1; i >= 0; --i)
    {
        pickInfoBuffers_[i].finalize();
    }
    
    finalizeRngBuffer();
    linearBuffers_.finalize();
    accumBuffers_.finalize();
    gbuffers_.finalize();

    initialized_ = false;
    width_ = 0;
    height_ = 0;
    currentFrameIndex_ = 0;
    cudaContext_ = nullptr;
    renderContext_ = nullptr;

    LOG(DBUG) << "ScreenBufferHandler finalized";
}

bool ScreenBufferHandler::resize(uint32_t width, uint32_t height)
{
    if (!validateDimensions(width, height))
    {
        LOG(WARNING) << "Invalid resize dimensions: " << width << "x" << height;
        return false;
    }

    if (!initialized_)
    {
        LOG(WARNING) << "ScreenBufferHandler not initialized";
        return false;
    }

    if (width_ == width && height_ == height)
    {
        LOG(DBUG) << "Resize requested with same dimensions, skipping";
        return true;
    }

    LOG(INFO) << "Resizing ScreenBufferHandler from " << width_ << "x" << height_ 
              << " to " << width << "x" << height;

    width_ = width;
    height_ = height;

    try
    {
        // Resize all buffer groups
        gbuffers_.resize(width, height);
        accumBuffers_.resize(width, height);
        linearBuffers_.resize(width, height);

        // RNG buffer needs special handling to preserve states
        resizeRngBuffer(cudaContext_, width, height);
        
        // Pick info buffers don't need resizing as they store single pixel info
        // They remain at size 1

        LOG(INFO) << "ScreenBufferHandler resized successfully. GPU memory usage: " 
                  << (getTotalGPUMemoryUsage() / (1024.0 * 1024.0)) << " MB";
        return true;
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Exception during resize: " << e.what();
        return false;
    }
}

void ScreenBufferHandler::clear()
{
    if (!initialized_)
    {
        return;
    }

    // Clear operations would be done through CUDA kernels when needed
    currentFrameIndex_ = 0;
}

// Surface object accessors are inline in the header

void ScreenBufferHandler::copyAccumToLinearBuffers(CUstream stream)
{
    if (!initialized_ || !kernelCopyAccumToLinear_)
    {
        LOG(DBUG) << "Copy kernel not available, skipping copyAccumToLinearBuffers";
        return;
    }

    // Get surface objects
    CUsurfObject beautySurf = accumBuffers_.beautyAccumBuffer.getSurfaceObject(0);
    CUsurfObject albedoSurf = accumBuffers_.albedoAccumBuffer.getSurfaceObject(0);
    CUsurfObject normalSurf = accumBuffers_.normalAccumBuffer.getSurfaceObject(0);
    CUsurfObject flowSurf = accumBuffers_.flowAccumBuffer.getSurfaceObject(0);
    
    // Get device pointers
    float4* beautyPtr = linearBuffers_.linearBeautyBuffer.getDevicePointer();
    float4* albedoPtr = linearBuffers_.linearAlbedoBuffer.getDevicePointer();
    float4* normalPtr = linearBuffers_.linearNormalBuffer.getDevicePointer();
    float2* flowPtr = linearBuffers_.linearFlowBuffer.getDevicePointer();
    
    // Create imageSize parameter
    uint2 imageSize = make_uint2(width_, height_);

    // Set up kernel parameters matching the copySurfacesToLinear kernel signature
    void* kernelParams[] = {
        &beautySurf,
        &albedoSurf,
        &normalSurf,
        &flowSurf,
        &beautyPtr,
        &albedoPtr,
        &normalPtr,
        &flowPtr,
        &imageSize
    };

    // Launch configuration
    uint32_t blockSizeX = 16;
    uint32_t blockSizeY = 16;
    uint32_t gridSizeX = (width_ + blockSizeX - 1) / blockSizeX;
    uint32_t gridSizeY = (height_ + blockSizeY - 1) / blockSizeY;

    // Launch the kernel
    CUresult result = cuLaunchKernel(
        kernelCopyAccumToLinear_,
        gridSizeX, gridSizeY, 1,     // Grid dimensions
        blockSizeX, blockSizeY, 1,    // Block dimensions
        0,                            // Shared memory size
        stream,                       // Stream
        kernelParams,                 // Kernel parameters
        nullptr                       // Extra options
    );
    
    if (result != CUDA_SUCCESS)
    {
        const char* errorName = nullptr;
        cuGetErrorName(result, &errorName);
        LOG(WARNING) << "Failed to launch copySurfacesToLinear kernel: " << errorName;
    }
}


bool ScreenBufferHandler::initializeRngStates(uint64_t seed)
{
    // Check if RNG buffer is initialized (not the handler itself, as this is called during initialization)
    if (!rngBuffer_.isInitialized())
    {
        return false;
    }

    if (seed == 0)
    {
        std::random_device rd;
        seed = ((uint64_t)rd() << 32) | rd();
    }

    auto rngs = rngBuffer_.map<shared::PCG32RNG>();
    std::mt19937_64 rngSeed(seed);
    
    for (uint32_t y = 0; y < height_; ++y)
    {
        for (uint32_t x = 0; x < width_; ++x)
        {
            shared::PCG32RNG& rng = rngs[y * width_ + x];
            rng.setState(rngSeed());
        }
    }
    
    rngBuffer_.unmap();
    return true;
}

size_t ScreenBufferHandler::getTotalGPUMemoryUsage() const
{
    if (!initialized_)
    {
        return 0;
    }

    size_t total = 0;
    
    // G-buffers (double buffered, 2 buffers with 4 uint32_t elements each)
    total += 2 * width_ * height_ * 4 * sizeof(uint32_t);  // gBuffer0
    total += 2 * width_ * height_ * 4 * sizeof(uint32_t);  // gBuffer1
    
    // Accumulation buffers (4 buffers of float4)
    total += 4 * width_ * height_ * sizeof(float4);  // beauty, albedo, normal, flow
    
    // Linear buffers (4 float4 + 1 float2)
    total += 4 * width_ * height_ * sizeof(float4);
    total += width_ * height_ * sizeof(float2);
    
    // RNG buffer
    total += width_ * height_ * sizeof(shared::PCG32RNG);
    
    // Pick info buffers (double buffered, single pixel each)
    total += 2 * sizeof(shared::PickInfo);
    
    return total;
}

// RNG buffer management
bool ScreenBufferHandler::initializeRngBuffer(CUcontext cuContext, uint32_t width, uint32_t height)
{
    rngBuffer_.initialize2D(
        cuContext, 
        cudau::ArrayElementType::UInt32, 
        (sizeof(shared::PCG32RNG) + 3) / 4,
        cudau::ArraySurface::Enable, 
        cudau::ArrayTextureGather::Disable,
        width, height, 1
    );

    return initializeRngStates();
}

void ScreenBufferHandler::finalizeRngBuffer()
{
    rngBuffer_.finalize();
}

void ScreenBufferHandler::resizeRngBuffer(CUcontext cuContext, uint32_t width, uint32_t height)
{
    // Save current RNG states if possible
    std::vector<shared::PCG32RNG> savedStates;
    if (initialized_ && width_ > 0 && height_ > 0)
    {
        uint32_t oldSize = width_ * height_;
        savedStates.resize(oldSize);
        auto rngs = rngBuffer_.map<shared::PCG32RNG>();
        std::copy(rngs, rngs + oldSize, savedStates.begin());
        rngBuffer_.unmap();
    }

    finalizeRngBuffer();
    initializeRngBuffer(cuContext, width, height);

    // Restore RNG states where possible
    if (!savedStates.empty())
    {
        auto rngs = rngBuffer_.map<shared::PCG32RNG>();
        uint32_t minWidth = std::min(width_, width);
        uint32_t minHeight = std::min(height_, height);
        
        for (uint32_t y = 0; y < minHeight; ++y)
        {
            for (uint32_t x = 0; x < minWidth; ++x)
            {
                rngs[y * width + x] = savedStates[y * width_ + x];
            }
        }
        
        // Initialize new pixels if buffer grew
        if (width > width_ || height > height_)
        {
            std::random_device rd;
            std::mt19937_64 rngSeed(((uint64_t)rd() << 32) | rd());
            
            for (uint32_t y = 0; y < height; ++y)
            {
                for (uint32_t x = 0; x < width; ++x)
                {
                    if (x >= width_ || y >= height_)
                    {
                        rngs[y * width + x].setState(rngSeed());
                    }
                }
            }
        }
        
        rngBuffer_.unmap();
    }
}

bool ScreenBufferHandler::loadKernels()
{
    if (!renderContext_ || !renderContext_->getPTXManager())
    {
        LOG(WARNING) << "PTXManager not available for loading copy kernels";
        return false;
    }
    
    // Load the Shocker copy buffers kernel (temporary until we have a generic one)
    std::vector<char> ptxData = renderContext_->getPTXManager()->getPTXData("optix_shocker_copybuffers");
    
    if (ptxData.empty())
    {
        LOG(WARNING) << "Failed to get PTX data for optix_shocker_copybuffers";
        return false;
    }
    
    LOG(DBUG) << "Loading PTX data for copy buffers (" << ptxData.size() << " bytes)";
    
    CUresult cuResult = cuModuleLoadData(&moduleCopyBuffers_, ptxData.data());
    if (cuResult != CUDA_SUCCESS)
    {
        LOG(WARNING) << "Failed to load copy buffers module: " << cuResult;
        return false;
    }
    
    // Get the kernel function
    cuResult = cuModuleGetFunction(&kernelCopyAccumToLinear_, moduleCopyBuffers_, "copySurfacesToLinear");
    if (cuResult != CUDA_SUCCESS)
    {
        LOG(WARNING) << "Failed to get copySurfacesToLinear kernel: " << cuResult;
        cuModuleUnload(moduleCopyBuffers_);
        moduleCopyBuffers_ = nullptr;
        return false;
    }
    
    LOG(INFO) << "Copy buffer kernels loaded successfully";
    return true;
}

void ScreenBufferHandler::cleanupKernels()
{
    if (kernelCopyAccumToLinear_)
    {
        kernelCopyAccumToLinear_ = nullptr;
    }
    
    if (moduleCopyBuffers_)
    {
        cuModuleUnload(moduleCopyBuffers_);
        moduleCopyBuffers_ = nullptr;
    }
}

bool ScreenBufferHandler::validateDimensions(uint32_t width, uint32_t height) const
{
    // Validate reasonable dimensions
    if (width == 0 || height == 0)
    {
        return false;
    }
    
    // Max texture size for most GPUs
    const uint32_t maxDimension = 16384;
    if (width > maxDimension || height > maxDimension)
    {
        return false;
    }
    
    // Check for reasonable total pixel count (avoid overflow)
    uint64_t pixelCount = (uint64_t)width * (uint64_t)height;
    const uint64_t maxPixels = 256ULL * 1024ULL * 1024ULL;  // 256 megapixels
    if (pixelCount > maxPixels)
    {
        return false;
    }
    
    return true;
}

// GBuffers implementation
void ScreenBufferHandler::GBuffers::initialize(CUcontext cuContext, uint32_t width, uint32_t height)
{
    for (int i = 0; i < 2; ++i)
    {
        gBuffer0[i].initialize2D(
            cuContext, cudau::ArrayElementType::UInt32, 4,  // 16 bytes / 4 = 4 uint32 channels
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        gBuffer1[i].initialize2D(
            cuContext, cudau::ArrayElementType::UInt32, 4,  // 16 bytes / 4 = 4 uint32 channels
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
    flowAccumBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 4,  // float4 for motion vectors (Shocker compatibility)
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        width, height, 1);
    LOG(DBUG) << "Accumulation buffers initialized";
}

void ScreenBufferHandler::AccumulationBuffers::resize(uint32_t width, uint32_t height)
{
    beautyAccumBuffer.resize(width, height);
    albedoAccumBuffer.resize(width, height);
    normalAccumBuffer.resize(width, height);
    flowAccumBuffer.resize(width, height);
    LOG(DBUG) << "Accumulation buffers resized to " << width << "x" << height;
}

void ScreenBufferHandler::AccumulationBuffers::finalize()
{
    flowAccumBuffer.finalize();
    normalAccumBuffer.finalize();
    albedoAccumBuffer.finalize();
    beautyAccumBuffer.finalize();
    LOG(DBUG) << "Accumulation buffers finalized";
}


// LinearBuffers implementation
void ScreenBufferHandler::LinearBuffers::initialize(CUcontext cuContext, uint32_t width, uint32_t height)
{
    uint32_t bufferSize = width * height;
    
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

