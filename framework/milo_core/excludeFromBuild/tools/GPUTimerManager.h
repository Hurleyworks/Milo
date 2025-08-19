#pragma once

// GPUTimerManager manages GPU performance timing for rendering pipeline stages.
// Provides double-buffered CUDA event timers for accurate GPU performance measurement.
// Tracks frame time, update, PDF computation, G-buffer setup, path tracing, and denoising.

#include "common/common_host.h"
#include "RenderContext.h"

class GPUTimerManager
{
 public:
    // GPU timer structure for measuring different rendering stages
    struct GPUTimer
    {
        cudau::Timer frame;
        cudau::Timer update;
        cudau::Timer computePDFTexture;
        cudau::Timer setupGBuffers;
        cudau::Timer pathTrace;
        cudau::Timer denoise;

        void initialize (CUcontext context);
        void finalize();
    };

    GPUTimerManager() = default;
    ~GPUTimerManager();

    GPUTimerManager (const GPUTimerManager&) = delete;
    GPUTimerManager& operator= (const GPUTimerManager&) = delete;
    GPUTimerManager (GPUTimerManager&&) = default;
    GPUTimerManager& operator= (GPUTimerManager&&) = default;

    // Initialize GPU timers with the provided context
    bool initialize (RenderContextPtr renderContext);

    // Clean up GPU timer resources
    void finalize();

    // Check if timers are initialized
    bool isInitialized() const { return initialized_; }

    // Access to GPU timers (double-buffered for async operations)
    const GPUTimer& getGPUTimer (uint32_t index) const;
    GPUTimer& getGPUTimer (uint32_t index);

    // Get the number of available timer buffers
    static constexpr uint32_t getTimerBufferCount() { return 2; }

 private:
    bool initialized_ = false;
    CUcontext cuda_context_ = nullptr;

    // Double-buffered GPU timers for async GPU operations
    GPUTimer gpu_timers_[2];

    // Validate timer buffer index
    bool isValidTimerIndex (uint32_t index) const { return index < 2; }
};