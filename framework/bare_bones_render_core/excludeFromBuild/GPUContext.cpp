#include "GPUContext.h"
#include <g3log/g3log.hpp>

GPUContext::~GPUContext()
{
    finalize();
}

bool GPUContext::initialize (int deviceIndex)
{
    if (initialized_)
    {
        LOG(WARNING) << "GPUContext already initialized";
        return true;
    }

    device_index_ = deviceIndex;

    if (!initializeCuda (deviceIndex))
    {
        LOG(WARNING) << "Failed to initialize CUDA context";
        return false;
    }

    if (!queryDeviceCapabilities())
    {
        LOG(WARNING) << "Failed to query device capabilities";
        cleanupCuda();
        return false;
    }

    if (!initializeOptix())
    {
        LOG(WARNING) << "Failed to initialize OptiX context";
        cleanupCuda();
        return false;
    }

    initialized_ = true;
    LOG(INFO) << "GPUContext initialized successfully";
    LOG(INFO) << "  Device: " << device_name_;
    LOG(INFO) << "  Compute Capability: " << compute_capability_major_ << "." << compute_capability_minor_;
    LOG(INFO) << "  Total Memory: " << (total_memory_ / (1024 * 1024)) << " MB";
    return true;
}

void GPUContext::finalize()
{
    if (!initialized_)
    {
        return;
    }

    cleanupOptix();
    cleanupCuda();

    initialized_ = false;
    LOG(DBUG) << "GPUContext finalized";
}

bool GPUContext::initializeCuda (int deviceIndex)
{
    // Initialize CUDA driver API
    CUDADRV_CHECK (cuInit (0));
    
    // Get device count and validate index
    int32_t cuda_device_count;
    CUDADRV_CHECK (cuDeviceGetCount (&cuda_device_count));
    
    if (deviceIndex >= cuda_device_count)
    {
        LOG(WARNING) << "Device index " << deviceIndex << " is out of range (available devices: " 
                     << cuda_device_count << ")";
        return false;
    }
    
    // Get device handle
    CUdevice cuda_device;
    CUDADRV_CHECK (cuDeviceGet (&cuda_device, deviceIndex));
    
    // Create context for the specified device
    CUDADRV_CHECK (cuCtxCreate (&cuda_context_, 0, cuda_device));
    CUDADRV_CHECK (cuCtxSetCurrent (cuda_context_));
    
    // Create a default stream
    CUDADRV_CHECK (cuStreamCreate (&cuda_stream_, 0));

    LOG(DBUG) << "CUDA context created successfully for device " << deviceIndex;
    return true;
}

bool GPUContext::queryDeviceCapabilities()
{
    CUdevice cuda_device;
    CUDADRV_CHECK (cuDeviceGet (&cuda_device, device_index_));
    
    // Get device name
    char device_name_buffer[256];
    CUDADRV_CHECK (cuDeviceGetName (device_name_buffer, sizeof (device_name_buffer), cuda_device));
    device_name_ = device_name_buffer;
    
    // Get compute capability
    CUDADRV_CHECK (cuDeviceGetAttribute (&compute_capability_major_,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                         cuda_device));
    CUDADRV_CHECK (cuDeviceGetAttribute (&compute_capability_minor_,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                         cuda_device));
    
    // Get total memory
    CUDADRV_CHECK (cuDeviceTotalMem (&total_memory_, cuda_device));
    
    // Check minimum requirements for OptiX
    if (compute_capability_major_ < 5)
    {
        LOG(WARNING) << "GPU compute capability " << compute_capability_major_ << "." 
                     << compute_capability_minor_ << " is below minimum required 5.0 for OptiX";
        return false;
    }
    
    return true;
}

bool GPUContext::initializeOptix()
{
    // Create OptiX context from CUDA context
    // Enable validation in debug builds
    const bool enable_validation = 
#ifdef _DEBUG
        true;
#else
        false;
#endif
    
    optix_context_ = optixu::Context::create (
        cuda_context_, 
        4,  // Max traversable graph depth
        enable_validation ? optixu::EnableValidation::Yes : optixu::EnableValidation::No);

    LOG(DBUG) << "OptiX context created successfully";
    return true;
}

void GPUContext::cleanupCuda()
{
    if (cuda_stream_)
    {
        CUDADRV_CHECK (cuStreamDestroy (cuda_stream_));
        cuda_stream_ = nullptr;
    }
    
    if (cuda_context_)
    {
        CUDADRV_CHECK (cuCtxDestroy (cuda_context_));
        cuda_context_ = nullptr;
    }
    LOG(DBUG) << "CUDA context destroyed";
}

void GPUContext::cleanupOptix()
{
    if (optix_context_)
    {
        optix_context_.destroy();
    }
    LOG(DBUG) << "OptiX context destroyed";
}