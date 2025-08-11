#include "RiPRRenderHandler.h"
#include "../../../RenderContext.h"
#include "../../../tools/PTXManager.h"
#include "../../../tools/GPUTimerManager.h"
#include "RiPRDenoiserHandler.h"

RiPRRenderHandler::RiPRRenderHandler(RenderContextPtr ctx)
    : renderContext_(ctx)
{
    LOG(INFO) << "RiPRRenderHandler created";
}

RiPRRenderHandler::~RiPRRenderHandler()
{
    finalize();
}

bool RiPRRenderHandler::initialize(uint32_t width, uint32_t height)
{
    LOG(INFO) << "Initializing RiPRRenderHandler with dimensions " << width << "x" << height;
    
    if (initialized_ && width_ == width && height_ == height)
    {
        LOG(DBUG) << "RiPRRenderHandler already initialized with same dimensions";
        return true;
    }
    
    // Clean up if already initialized with different dimensions
    if (initialized_)
    {
        finalize();
    }
    
    if (!renderContext_ || !renderContext_->isInitialized())
    {
        LOG(WARNING) << "RenderContext not initialized";
        return false;
    }
    
    width_ = width;
    height_ = height;
    accumulationCount_ = 0;
    
    CUcontext cuContext = renderContext_->getCudaContext();
    
    // Initialize accumulation buffers (2D arrays for RiPR kernels to write to)
    try
    {
        // Beauty accumulation buffer (HDR color)
        beautyAccumBuffer_.initialize2D(
            cuContext, 
            cudau::ArrayElementType::Float32, 4,  // float4
            cudau::ArraySurface::Enable, 
            cudau::ArrayTextureGather::Disable,
            width, height, 1
        );
        
        // Albedo accumulation buffer
        albedoAccumBuffer_.initialize2D(
            cuContext, 
            cudau::ArrayElementType::Float32, 4,  // float4
            cudau::ArraySurface::Enable, 
            cudau::ArrayTextureGather::Disable,
            width, height, 1
        );
        
        // Normal accumulation buffer
        normalAccumBuffer_.initialize2D(
            cuContext, 
            cudau::ArrayElementType::Float32, 4,  // float4
            cudau::ArraySurface::Enable, 
            cudau::ArrayTextureGather::Disable,
            width, height, 1
        );
        
        // Motion accumulation buffer (motion vectors for temporal denoising)
        motionAccumBuffer_.initialize2D(
            cuContext, 
            cudau::ArrayElementType::Float32, 4,  // float4 (using float4 for consistency, even though we only need float2)
            cudau::ArraySurface::Enable, 
            cudau::ArrayTextureGather::Disable,
            width, height, 1
        );
        
        // Initialize linear buffers for display and denoising
        linearBeautyBuffer_.initialize(cuContext, cudau::BufferType::Device, width * height);
        linearAlbedoBuffer_.initialize(cuContext, cudau::BufferType::Device, width * height);
        linearNormalBuffer_.initialize(cuContext, cudau::BufferType::Device, width * height);
        linearMotionBuffer_.initialize(cuContext, cudau::BufferType::Device, width * height);
        
        // Initialize pick info buffers (double buffered)
        for (int i = 0; i < 2; ++i)
        {
            pickInfoBuffers_[i].initialize(cuContext, cudau::BufferType::Device, 1);  // Single element for pick result
        }
        
        // Load CUDA kernels for buffer operations
        loadKernels();
        
        initialized_ = true;
        LOG(INFO) << "RiPRRenderHandler initialized successfully";
        return true;
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to initialize RiPRRenderHandler: " << e.what();
        finalize();
        return false;
    }
}

void RiPRRenderHandler::finalize()
{
    if (!initialized_)
    {
        return;
    }
    
    LOG(DBUG) << "Finalizing RiPRRenderHandler";
    
    // Clean up kernels
    cleanupKernels();
    
    // Finalize buffers
    if (beautyAccumBuffer_.isInitialized())
        beautyAccumBuffer_.finalize();
    if (albedoAccumBuffer_.isInitialized())
        albedoAccumBuffer_.finalize();
    if (normalAccumBuffer_.isInitialized())
        normalAccumBuffer_.finalize();
    if (motionAccumBuffer_.isInitialized())
        motionAccumBuffer_.finalize();
    
    if (linearBeautyBuffer_.isInitialized())
        linearBeautyBuffer_.finalize();
    if (linearAlbedoBuffer_.isInitialized())
        linearAlbedoBuffer_.finalize();
    if (linearNormalBuffer_.isInitialized())
        linearNormalBuffer_.finalize();
    if (linearMotionBuffer_.isInitialized())
        linearMotionBuffer_.finalize();
    
    // Finalize pick info buffers
    for (int i = 0; i < 2; ++i)
    {
        if (pickInfoBuffers_[i].isInitialized())
            pickInfoBuffers_[i].finalize();
    }
    
    width_ = 0;
    height_ = 0;
    accumulationCount_ = 0;
    initialized_ = false;
    
    LOG(DBUG) << "RiPRRenderHandler resources finalized";
}

void RiPRRenderHandler::loadKernels()
{
    LOG(INFO) << "Loading RiPR render kernels...";
    
    if (!renderContext_ || !renderContext_->getPTXManager())
    {
        LOG(WARNING) << "PTXManager not available";
        return;
    }
    
    // Use RiPR-specific copy buffers kernel
    std::vector<char> ptxData = renderContext_->getPTXManager()->getPTXData("optix_ripr_copybuffers");
    
    if (ptxData.empty())
    {
        // Fall back to Milo kernels if RiPR-specific ones aren't available yet
        LOG(WARNING) << "RiPR-specific PTX not found, falling back to Milo kernels";
        ptxData = renderContext_->getPTXManager()->getPTXData("optix_milo_copybuffers");
        
        if (ptxData.empty())
        {
            LOG(WARNING) << "Failed to get PTX data for copy buffers";
            return;
        }
    }
    
    LOG(DBUG) << "Loading PTX data for RiPR copy buffers (" << ptxData.size() << " bytes)";
    
    CUcontext cuContext = renderContext_->getCudaContext();
    
    // Load the module
    CUDADRV_CHECK(cuModuleLoadData(&moduleRiPRCopyBuffers_, ptxData.data()));
    
    // Set up the kernels
    kernelCopySurfacesToLinear_.set(moduleRiPRCopyBuffers_, "copySurfacesToLinear", cudau::dim3(8, 8), 0);
  
    LOG(INFO) << "RiPR copy buffers module loaded successfully";
}

void RiPRRenderHandler::cleanupKernels()
{
    if (moduleRiPRCopyBuffers_)
    {
        CUDADRV_CHECK(cuModuleUnload(moduleRiPRCopyBuffers_));
        moduleRiPRCopyBuffers_ = nullptr;
    }
}

void RiPRRenderHandler::copyAccumToLinearBuffers(CUstream stream)
{
    if (!initialized_ || !moduleRiPRCopyBuffers_)
    {
        LOG(WARNING) << "RiPRRenderHandler not properly initialized";
        return;
    }
    
    // Use the combined kernel to copy all buffers in a single pass
    kernelCopySurfacesToLinear_.launchWithThreadDim(
        stream, cudau::dim3(width_, height_),
        beautyAccumBuffer_.getSurfaceObject(0),
        albedoAccumBuffer_.getSurfaceObject(0),
        normalAccumBuffer_.getSurfaceObject(0),
        motionAccumBuffer_.getSurfaceObject(0),
        linearBeautyBuffer_.getDevicePointer(),
        linearAlbedoBuffer_.getDevicePointer(),
        linearNormalBuffer_.getDevicePointer(),
        linearMotionBuffer_.getDevicePointer(),
        uint2{width_, height_}
    );
}


bool RiPRRenderHandler::denoise(CUstream stream, bool isNewSequence, RiPRDenoiserHandler* denoiserHandler, GPUTimerManager::GPUTimer* timer)
{
    if (!initialized_)
    {
        LOG(WARNING) << "RiPRRenderHandler not initialized";
        return false;
    }
    
    // Check if denoiser handler is provided and initialized
    if (!denoiserHandler || !denoiserHandler->isInitialized())
    {
        // No denoiser available, this is not an error
        return false;
    }
    
    // Start timing if timer provided
    if (timer)
    {
        timer->denoise.start(stream);
    }
    
    // Setup denoiser input buffers
    optixu::DenoiserInputBuffers inputBuffers = {};
    inputBuffers.noisyBeauty = linearBeautyBuffer_;
    inputBuffers.albedo = linearAlbedoBuffer_;
    inputBuffers.normal = linearNormalBuffer_;
    inputBuffers.flow = linearMotionBuffer_;  // Motion vectors for temporal denoising
    
    // For temporal denoising: if new sequence, use noisy beauty as previous
    // Otherwise, the denoised result from previous frame is already in the beauty buffer
    if (denoiserHandler->isTemporalDenoiser())
    {
        inputBuffers.previousDenoisedBeauty = isNewSequence ? linearBeautyBuffer_ : linearBeautyBuffer_;
    }
    
    // Set all formats
    inputBuffers.beautyFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    inputBuffers.albedoFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    inputBuffers.normalFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    inputBuffers.flowFormat = OPTIX_PIXEL_FORMAT_FLOAT2;  // Motion vectors are float2
    
    // Compute HDR normalizer
    CUdeviceptr hdrNormalizer = 0;
    CUDADRV_CHECK(cuMemAlloc(&hdrNormalizer, sizeof(float)));
    
    // Use RAII to ensure cleanup
    struct HDRNormalizerCleanup {
        CUdeviceptr ptr;
        ~HDRNormalizerCleanup() noexcept {
            if (ptr) {
                CUresult result = cuMemFree(ptr);
                if (result != CUDA_SUCCESS) {
                    LOG(WARNING) << "Failed to free HDR normalizer memory: " << result;
                }
            }
        }
    } cleanup{hdrNormalizer};
    
    denoiserHandler->getDenoiser().computeNormalizer(
        stream,
        linearBeautyBuffer_, OPTIX_PIXEL_FORMAT_FLOAT4,
        denoiserHandler->getScratchBuffer(), hdrNormalizer);
    
    // Denoise (output goes back to beauty buffer since we don't have separate denoised buffer)
    const auto& denoisingTasks = denoiserHandler->getTasks();
    for (size_t i = 0; i < denoisingTasks.size(); ++i)
    {
        denoiserHandler->getDenoiser().invoke(
            stream, denoisingTasks[i],
            inputBuffers, optixu::IsFirstFrame(isNewSequence),
            hdrNormalizer, 0.0f,
            linearBeautyBuffer_,  // Output to beauty buffer
            nullptr, optixu::BufferView()); // no AOV outputs, no internal guide layer
    }
    
    // Stop timing if timer provided
    if (timer)
    {
        timer->denoise.stop(stream);
    }
    
   // LOG(DBUG) << "RiPR denoising completed (temporal: " << denoiserHandler->isTemporalDenoiser() 
     //         << ", new sequence: " << isNewSequence << ")";
    
    return true;  // Successfully invoked denoiser
}

bool RiPRRenderHandler::resize(uint32_t newWidth, uint32_t newHeight)
{
    if (newWidth == width_ && newHeight == height_)
    {
        return true;
    }
    
    LOG(INFO) << "Resizing RiPRRenderHandler from " << width_ << "x" << height_ 
              << " to " << newWidth << "x" << newHeight;
    
    // Re-initialize with new dimensions
    return initialize(newWidth, newHeight);
}