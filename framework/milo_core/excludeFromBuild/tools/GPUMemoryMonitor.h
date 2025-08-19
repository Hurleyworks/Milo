#pragma once

#include "../../milo_core.h"
#include "../common/common_shared.h"

// Custom exception for low GPU memory conditions
class GPUMemoryException : public std::runtime_error
{
 public:
    explicit GPUMemoryException (const std::string& message) :
        std::runtime_error (message) {}
};

// Utility class for monitoring GPU memory usage in real-time
class GPUMemoryMonitor
{
 public:
    // Get total GPU memory in bytes
    static size_t GetTotalMemory();

    // Get currently used GPU memory in bytes
    static size_t GetUsedMemory();

    // Get available GPU memory in bytes
    static size_t GetAvailableMemory();

    // Get memory usage as percentage (0-100)
    static float GetMemoryUsagePercent();

    // Log current memory statistics
    static void LogMemoryStats();

    // Check if memory usage is above dangerous threshold
    // Throws GPUMemoryException if memory is critically low
    static void CheckMemoryThreshold();

    // Set custom threshold percentage (0-100)
    static void SetMemoryThreshold (float thresholdPercent);

     // Check if requested allocation is safe
    static bool HasEnoughMemoryForAllocation (size_t requestedBytes);

    // Calculate memory required for render buffers
    static size_t CalculateRenderBufferMemory (uint32_t width, uint32_t height);

    static bool IsResolutionSafe (uint32_t width, uint32_t height);

    static GPUMemoryStats getStats();

    static void UpdateMemoryStatsForDevice (int deviceIndex);
    static void SetCurrentContext (CUcontext context);
    static void UpdateMemoryStats();

 private:
    static size_t totalMemory;
    static size_t freeMemory;
    static size_t usedMemory;
    static float memoryThresholdPercent;

    static CUcontext currentContext;

    // Default threshold is 95% usage
    static constexpr float DEFAULT_THRESHOLD_PERCENT = 95.0f;
};