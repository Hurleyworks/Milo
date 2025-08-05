#include "MiloRenderHandler.h"
#include "../RenderContext.h"
#include "../tools/PTXManager.h"
#include "../tools/GPUTimerManager.h"
#include "../milo_shared.h"
#include "MiloDenoiserHandler.h"

MiloRenderHandler::MiloRenderHandler(RenderContextPtr ctx)
    : renderContext_(ctx)
{
    LOG(INFO) << "MiloRenderHandler created";
}

MiloRenderHandler::~MiloRenderHandler()
{
    finalize();
}

bool MiloRenderHandler::initialize(uint32_t width, uint32_t height)
{
    LOG(INFO) << "Initializing MiloRenderHandler with dimensions " << width << "x" << height;
    
    if (initialized_ && width_ == width && height_ == height)
    {
        LOG(DBUG) << "MiloRenderHandler already initialized with same dimensions";
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
    
    CUcontext cuContext = renderContext_->getCudaContext();
    
    // Initialize accumulation buffers (2D arrays for Milo kernels to write to)
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
        
        // Flow accumulation buffer (motion vectors)
        flowAccumBuffer_.initialize2D(
            cuContext, 
            cudau::ArrayElementType::Float32, 4,  // float4
            cudau::ArraySurface::Enable, 
            cudau::ArrayTextureGather::Disable,
            width, height, 1
        );
        
        // Initialize linear buffers for display and denoising
        linearBeautyBuffer_.initialize(cuContext, cudau::BufferType::Device, width * height);
        linearAlbedoBuffer_.initialize(cuContext, cudau::BufferType::Device, width * height);
        linearNormalBuffer_.initialize(cuContext, cudau::BufferType::Device, width * height);
        linearFlowBuffer_.initialize(cuContext, cudau::BufferType::Device, width * height);
        
        // Initialize pick info buffers (double buffered)
        for (int i = 0; i < 2; ++i)
        {
            pickInfoBuffers_[i].initialize(cuContext, cudau::BufferType::Device, 1);  // Single element for pick result
        }
        
        // Load CUDA kernels for buffer operations
        loadKernels();
        
        initialized_ = true;
        LOG(INFO) << "MiloRenderHandler initialized successfully";
        return true;
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to initialize MiloRenderHandler: " << e.what();
        finalize();
        return false;
    }
}

void MiloRenderHandler::finalize()
{
    if (!initialized_)
    {
        return;
    }
    
    LOG(DBUG) << "Finalizing MiloRenderHandler";
    
    // Clean up kernels
    cleanupKernels();
    
    // Finalize buffers
    if (beautyAccumBuffer_.isInitialized())
        beautyAccumBuffer_.finalize();
    if (albedoAccumBuffer_.isInitialized())
        albedoAccumBuffer_.finalize();
    if (normalAccumBuffer_.isInitialized())
        normalAccumBuffer_.finalize();
    if (flowAccumBuffer_.isInitialized())
        flowAccumBuffer_.finalize();
    
    if (linearBeautyBuffer_.isInitialized())
        linearBeautyBuffer_.finalize();
    if (linearAlbedoBuffer_.isInitialized())
        linearAlbedoBuffer_.finalize();
    if (linearNormalBuffer_.isInitialized())
        linearNormalBuffer_.finalize();
    if (linearFlowBuffer_.isInitialized())
        linearFlowBuffer_.finalize();
    
    // Finalize pick info buffers
    for (int i = 0; i < 2; ++i)
    {
        if (pickInfoBuffers_[i].isInitialized())
            pickInfoBuffers_[i].finalize();
    }
    
    width_ = 0;
    height_ = 0;
    initialized_ = false;
    
    LOG(DBUG) << "MiloRenderHandler resources finalized";
}

void MiloRenderHandler::loadKernels()
{
    LOG(INFO) << "Loading Milo render kernels...";
    
    if (!renderContext_ || !renderContext_->getPTXManager())
    {
        LOG(WARNING) << "PTXManager not available";
        return;
    }
    
    // Use Milo-specific copy buffers kernel
    std::vector<char> ptxData = renderContext_->getPTXManager()->getPTXData("optix_milo_copybuffers");
    
    if (ptxData.empty())
    {
        LOG(WARNING) << "Failed to get PTX data for optix_milo_copybuffers";
        return;
    }
    
    LOG(DBUG) << "Loading PTX data for Milo copy buffers (" << ptxData.size() << " bytes)";
    
    CUcontext cuContext = renderContext_->getCudaContext();
    
    // Load the module
    CUDADRV_CHECK(cuModuleLoadData(&moduleCopyBuffers_, ptxData.data()));
    
    // Set up the combined kernel
    kernelCopySurfacesToLinear_.set(moduleCopyBuffers_, "copySurfacesToLinear", cudau::dim3(8, 8), 0);
    
    LOG(INFO) << "Milo copy buffers module loaded successfully";
}

void MiloRenderHandler::cleanupKernels()
{
    if (moduleCopyBuffers_)
    {
        CUDADRV_CHECK(cuModuleUnload(moduleCopyBuffers_));
        moduleCopyBuffers_ = nullptr;
    }
}

void MiloRenderHandler::copyAccumToLinearBuffers(CUstream stream)
{
    if (!initialized_ || !moduleCopyBuffers_)
    {
        LOG(WARNING) << "MiloRenderHandler not properly initialized";
        return;
    }
    
    // Use the combined kernel to copy all buffers in a single pass
    kernelCopySurfacesToLinear_.launchWithThreadDim(
        stream, cudau::dim3(width_, height_),
        beautyAccumBuffer_.getSurfaceObject(0),
        albedoAccumBuffer_.getSurfaceObject(0),
        normalAccumBuffer_.getSurfaceObject(0),
        flowAccumBuffer_.getSurfaceObject(0),
        linearBeautyBuffer_.getDevicePointer(),
        linearAlbedoBuffer_.getDevicePointer(),
        linearNormalBuffer_.getDevicePointer(),
        linearFlowBuffer_.getDevicePointer(),
        uint2{width_, height_}
    );
}


void MiloRenderHandler::clearAccumBuffers(CUstream stream)
{
    if (!initialized_)
    {
        LOG(WARNING) << "MiloRenderHandler not initialized";
        return;
    }
    
    // For now, we'll rely on the Milo kernels to handle initialization
    // The clearAccumBuffers kernel is available in optix_milo_copybuffers if needed
    LOG(DBUG) << "Accumulation buffer clearing requested - Milo kernels will handle initialization";
}


bool MiloRenderHandler::denoise(CUstream stream, bool isNewSequence, MiloDenoiserHandler* denoiserHandler, GPUTimerManager::GPUTimer* timer)
{
    if (!initialized_)
    {
        LOG(WARNING) << "MiloRenderHandler not initialized";
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
    inputBuffers.flow = linearFlowBuffer_;  // Motion vectors for temporal denoising
    
    // For temporal denoising: if new sequence, use noisy beauty as previous
    inputBuffers.previousDenoisedBeauty = linearBeautyBuffer_;
    
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
    for (int i = 0; i < denoisingTasks.size(); ++i)
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
    
    return true;  // Successfully invoked denoiser
}

bool MiloRenderHandler::resize(uint32_t newWidth, uint32_t newHeight)
{
    if (newWidth == width_ && newHeight == height_)
    {
        return true;
    }
    
    LOG(INFO) << "Resizing MiloRenderHandler from " << width_ << "x" << height_ 
              << " to " << newWidth << "x" << newHeight;
    
    // Re-initialize with new dimensions
    return initialize(newWidth, newHeight);
}