#include "GPUContext.h"
#include "tools/PTXManager.h"

GPUContext::GPUContext()
{
    LOG (DBUG) << "GPUContext constructor called";
}

GPUContext::~GPUContext()
{
    LOG (DBUG) << "GPUContext destructor called";
    cleanup();
}

bool GPUContext::checkGPUCapability()
{
    int cudaDevice = 0;
    int computeCapability[2] = {0, 0};
    char deviceName[256];
    size_t totalMem = 0;

    try
    {
        CUDADRV_CHECK (cuDeviceGet (&cudaDevice, 0));

        // Get GPU name
        CUDADRV_CHECK (cuDeviceGetName (deviceName, sizeof (deviceName), cudaDevice));

        // Get compute capability
        CUDADRV_CHECK (cuDeviceGetAttribute (&computeCapability[0],
                                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                             cudaDevice));
        CUDADRV_CHECK (cuDeviceGetAttribute (&computeCapability[1],
                                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                             cudaDevice));

        // Get total memory
        CUDADRV_CHECK (cuDeviceTotalMem (&totalMem, cudaDevice));

        // Get driver version
        int driverVersion = 0;
        CUDADRV_CHECK (cuDriverGetVersion (&driverVersion));

        // Store compute capability
        computeCapabilityMajor_ = computeCapability[0];
        computeCapabilityMinor_ = computeCapability[1];

        // Log system information
        LOG (INFO) << "GPU Device: " << deviceName;
        LOG (INFO) << "Compute Capability: " << computeCapability[0] << "." << computeCapability[1];
        LOG (INFO) << "Total Memory: " << (totalMem / (1024 * 1024)) << " MB";
        LOG (INFO) << "CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 100);

        // Check minimum requirements
        if (computeCapability[0] < 6)
        {
            LOG (WARNING) << "GPU compute capability " << computeCapability[0] << "."
                          << computeCapability[1] << " is below minimum required 6.0";
            return false;
        }

        if (driverVersion < 10010)
        {
            LOG (WARNING) << "CUDA driver version " << driverVersion
                          << " is below minimum required 10.1";
            return false;
        }

        return true;
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error checking GPU capability: " << e.what();
        return false;
    }
}

bool GPUContext::initialize()
{
    if (initialized_)
    {
        LOG (WARNING) << "GPUContext already initialized";
        return true;
    }

    try
    {
        // Initialize CUDA driver API
        LOG (INFO) << "Initializing CUDA driver API...";
        CUDADRV_CHECK (cuInit (0));

        // Check GPU capability
        if (!checkGPUCapability())
        {
            LOG (WARNING) << "GPU does not meet minimum requirements";
            return false;
        }

        // Get device and create context
        CUdevice device;
        CUDADRV_CHECK (cuDeviceGet (&device, 0));
        CUDADRV_CHECK (cuCtxCreate (&cuContext_, 0, device));
        CUDADRV_CHECK (cuCtxSetCurrent (cuContext_));

        // Create CUDA stream
        CUDADRV_CHECK (cuStreamCreate (&cuStream_, 0));

        // Initialize OptiX context
        LOG (INFO) << "Initializing OptiX context...";
        optixContext_ = optixu::Context::create (
            cuContext_, 4,
            optixu::EnableValidation::DEBUG_SELECT (Yes, No));

        initialized_ = true;
        LOG (INFO) << "OptiX context created successfully";
        LOG (INFO) << "GPU context initialized successfully";
        return true;
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Failed to initialize GPU context: " << e.what();
        cleanup();
        return false;
    }
}

void GPUContext::cleanup()
{
    if (!initialized_)
        return;

    try
    {
        LOG (INFO) << "Cleaning up GPU context...";

        // Synchronize before cleanup
        if (cuStream_)
        {
            CUDADRV_CHECK (cuStreamSynchronize (cuStream_));
        }

        // Cleanup OptiX context
        if (optixContext_)
        {
            optixContext_.destroy();
        }

        // Cleanup CUDA resources
        if (cuStream_)
        {
            CUDADRV_CHECK (cuStreamDestroy (cuStream_));
            cuStream_ = nullptr;
        }

        if (cuContext_)
        {
            CUDADRV_CHECK (cuCtxSynchronize());
            CUDADRV_CHECK (cuCtxDestroy (cuContext_));
            cuContext_ = nullptr;
        }

        initialized_ = false;
        LOG (INFO) << "GPU context cleanup completed";
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error during GPU context cleanup: " << e.what();
    }
}
