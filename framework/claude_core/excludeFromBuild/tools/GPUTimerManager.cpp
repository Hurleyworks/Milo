#include "GPUTimerManager.h"

// GPUTimer implementation
void GPUTimerManager::GPUTimer::initialize(CUcontext context)
{
    frame.initialize(context);
    update.initialize(context);
    computePDFTexture.initialize(context);
    setupGBuffers.initialize(context);
    pathTrace.initialize(context);
    denoise.initialize(context);
}

void GPUTimerManager::GPUTimer::finalize()
{
    denoise.finalize();
    pathTrace.finalize();
    setupGBuffers.finalize();
    computePDFTexture.finalize();
    update.finalize();
    frame.finalize();
}

// GPUTimerManager implementation
GPUTimerManager::~GPUTimerManager()
{
    finalize();
}

bool GPUTimerManager::initialize(RenderContextPtr renderContext)
{
    if (initialized_)
    {
        LOG(INFO) << "GPUTimerManager already initialized";
        return true;
    }

    if (!renderContext) {
        LOG(WARNING) << "Invalid RenderContext provided to GPUTimerManager";
        return false;
    }

    cuda_context_ = renderContext->getCudaContext();

    try
    {
        // Initialize both timer buffers
        gpu_timers_[0].initialize(cuda_context_);
        gpu_timers_[1].initialize(cuda_context_);

        initialized_ = true;
        LOG(INFO) << "GPUTimerManager initialized successfully";
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to initialize GPU timers: " << ex.what();
        finalize();
        return false;
    }
}

void GPUTimerManager::finalize()
{
    if (!initialized_)
    {
        return;
    }

    // Finalize timer buffers in reverse order
    gpu_timers_[1].finalize();
    gpu_timers_[0].finalize();

    initialized_ = false;
    cuda_context_ = nullptr;

    LOG(INFO) << "GPUTimerManager finalized";
}

const GPUTimerManager::GPUTimer& GPUTimerManager::getGPUTimer(uint32_t index) const
{
    if (!isValidTimerIndex(index))
    {
        LOG(WARNING) << "Invalid timer index: " << index << ", using index 0";
        // Return first timer as fallback
        return gpu_timers_[0];
    }
    return gpu_timers_[index];
}

GPUTimerManager::GPUTimer& GPUTimerManager::getGPUTimer(uint32_t index)
{
    if (!isValidTimerIndex(index))
    {
        LOG(WARNING) << "Invalid timer index: " << index << ", using index 0";
        // Return first timer as fallback
        return gpu_timers_[0];
    }
    return gpu_timers_[index];
}