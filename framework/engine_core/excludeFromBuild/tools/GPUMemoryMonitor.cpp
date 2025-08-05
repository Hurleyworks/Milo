#include "GPUMemoryMonitor.h"


// Initialize static members
CUcontext GPUMemoryMonitor::currentContext = nullptr;
size_t GPUMemoryMonitor::totalMemory = 0;
size_t GPUMemoryMonitor::freeMemory = 0;
size_t GPUMemoryMonitor::usedMemory = 0;
float GPUMemoryMonitor::memoryThresholdPercent = DEFAULT_THRESHOLD_PERCENT;

void GPUMemoryMonitor::UpdateMemoryStats()
{
    size_t free = 0;
    size_t total = 0;

    CUresult result;
    CUcontext oldContext = nullptr;

    try
    {
        // If we have a specific context to use
        if (currentContext)
        {
            CUDADRV_CHECK (cuCtxPushCurrent (currentContext));
        }

        // Get memory info
        CUDADRV_CHECK (cuMemGetInfo (&free, &total));
        totalMemory = total;
        freeMemory = free;
        usedMemory = total - free;

        // Restore previous context if we pushed one
        if (currentContext)
        {
            CUDADRV_CHECK (cuCtxPopCurrent (&oldContext));
        }
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "UpdateMemoryStats failed: " << e.what();

        // Make sure we pop the context even if there was an error
        if (currentContext)
        {
            cuCtxPopCurrent (nullptr);
        }
        throw;
    }
}


size_t GPUMemoryMonitor::GetTotalMemory()
{
    UpdateMemoryStats();
    return totalMemory;
}

size_t GPUMemoryMonitor::GetUsedMemory()
{
    UpdateMemoryStats();
    return usedMemory;
}

size_t GPUMemoryMonitor::GetAvailableMemory()
{
    UpdateMemoryStats();
    return freeMemory;
}

float GPUMemoryMonitor::GetMemoryUsagePercent()
{
    UpdateMemoryStats();
    if (totalMemory > 0)
    {
        return (static_cast<float> (usedMemory) / totalMemory) * 100.0f;
    }
    return 0.0f;
}

void GPUMemoryMonitor::LogMemoryStats()
{
    UpdateMemoryStats();
    LOG (DBUG) << "GPU Memory Usage: " << GetMemoryUsagePercent() << "%"
               << " (Used: " << formatByteSize (usedMemory)
               << " Free: " << formatByteSize (freeMemory)
               << " Total: " << formatByteSize (totalMemory);
}

void GPUMemoryMonitor::SetMemoryThreshold (float thresholdPercent)
{
    if (thresholdPercent <= 0.0f || thresholdPercent > 100.0f)
    {
        LOG (WARNING) << "Invalid memory threshold percentage. Using default: "
                      << DEFAULT_THRESHOLD_PERCENT << "%";
        memoryThresholdPercent = DEFAULT_THRESHOLD_PERCENT;
    }
    else
    {
        memoryThresholdPercent = thresholdPercent;
    }
}

void GPUMemoryMonitor::CheckMemoryThreshold()
{
    float currentUsage = GetMemoryUsagePercent();

    if (currentUsage >= memoryThresholdPercent)
    {
        std::stringstream ss;
        ss << "Critical GPU memory usage: " << currentUsage << "% "
           << " (Used: " << formatByteSize (usedMemory)
           << " Free: " << formatByteSize (freeMemory)
           << " Total: " << formatByteSize (totalMemory);

        throw GPUMemoryException (ss.str());
    }
}

bool GPUMemoryMonitor::HasEnoughMemoryForAllocation (size_t requestedBytes)
{
    UpdateMemoryStats();

    // Add 10% buffer for safety
    size_t requiredWithBuffer = static_cast<size_t> (requestedBytes * 1.1);

    LOG (DBUG) << "---Free memory  " << formatByteSize (freeMemory);
    LOG (DBUG) << "----Required memory  " << formatByteSize (requiredWithBuffer);

    // Check if we have enough free memory
    return (freeMemory > requiredWithBuffer);
}

size_t GPUMemoryMonitor::CalculateRenderBufferMemory (uint32_t width, uint32_t height)
{
    // Calculate memory for all render buffers (beauty, albedo, normal, etc)
    // 4 channels * 4 bytes per channel * width * height * number of buffers
    constexpr size_t BYTES_PER_PIXEL = 16; // 4 channels * 4 bytes
    constexpr size_t NUM_BUFFERS = 5;      // beauty, albedo, normal, denoised, temporary

    return width * height * BYTES_PER_PIXEL * NUM_BUFFERS;
}

bool GPUMemoryMonitor::IsResolutionSafe (uint32_t width, uint32_t height)
{
    // Calculate memory needed for render buffers
    size_t renderMemory = CalculateRenderBufferMemory (width, height);

    // Calculate denoiser scratch memory based on OptiX's formula
    // Each pixel needs ~302 bytes for scratch space at 4x4 tiles
    constexpr size_t BYTES_PER_PIXEL_DENOISER = 302;
    size_t denoiserScratch = width * height * BYTES_PER_PIXEL_DENOISER;

    // Add 20% safety margin
    size_t totalRequired = static_cast<size_t> ((renderMemory + denoiserScratch) * 1.2);

    if (!HasEnoughMemoryForAllocation (totalRequired))
        return false;

    // Also enforce maximum safe dimensions
    constexpr uint32_t MAX_DIMENSION = 16384;
    if (width > MAX_DIMENSION || height > MAX_DIMENSION)
        return false;

    return true;
}

GPUMemoryStats GPUMemoryMonitor::getStats()
{
    GPUMemoryStats stats;
    stats.freeMemory = freeMemory;
    stats.usedMemory = usedMemory;
    stats.totalMemory = totalMemory;

    return stats;
}

void GPUMemoryMonitor::UpdateMemoryStatsForDevice (int deviceIndex)
{
    CUdevice device;
    CUcontext tempContext = nullptr;
    CUcontext oldContext = nullptr;

    try
    {
        // Get the device
        CUDADRV_CHECK (cuDeviceGet (&device, deviceIndex));

        // Save current context
        CUDADRV_CHECK (cuCtxGetCurrent (&oldContext));

        // Create a temporary context for this device
        CUDADRV_CHECK (cuCtxCreate (&tempContext, 0, device));

        // Get memory info
        size_t free = 0;
        size_t total = 0;
        CUDADRV_CHECK (cuMemGetInfo (&free, &total));

        totalMemory = total;
        freeMemory = free;
        usedMemory = total - free;

        // Clean up temporary context
        CUDADRV_CHECK (cuCtxDestroy (tempContext));

        // Restore previous context
        if (oldContext)
        {
            CUDADRV_CHECK (cuCtxSetCurrent (oldContext));
        }
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Failed to get memory stats for device " << deviceIndex
                      << ": " << e.what();

        // Clean up even on error
        if (tempContext)
        {
            cuCtxDestroy (tempContext);
        }
        if (oldContext)
        {
            cuCtxSetCurrent (oldContext);
        }
        throw;
    }
}
void GPUMemoryMonitor::SetCurrentContext (CUcontext context)
{
    currentContext = context;
}
