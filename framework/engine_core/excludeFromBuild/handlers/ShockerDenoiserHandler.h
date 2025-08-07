#pragma once

// ShockerDenoiserHandler manages OptiX AI-accelerated denoising functionality for Shocker rendering.
// Based on the sample DenoiserManager implementation, provides temporal and HDR denoising modes.
// Handles denoiser state setup, buffer management, and task configuration for the Shocker pipeline.

#include "../common/common_host.h"
#include "../GPUContext.h"
#include <memory>

class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
using ShockerDenoiserHandlerPtr = std::shared_ptr<class ShockerDenoiserHandler>;

class ShockerDenoiserHandler
{
public:
    // Factory method following handler pattern
    static ShockerDenoiserHandlerPtr create(RenderContextPtr renderContext);

    // Constructor with RenderContext for consistency with other handlers
    explicit ShockerDenoiserHandler(RenderContextPtr renderContext);
    ~ShockerDenoiserHandler();

    // Delete copy operations
    ShockerDenoiserHandler(const ShockerDenoiserHandler&) = delete;
    ShockerDenoiserHandler& operator=(const ShockerDenoiserHandler&) = delete;

    // Allow move operations
    ShockerDenoiserHandler(ShockerDenoiserHandler&&) = default;
    ShockerDenoiserHandler& operator=(ShockerDenoiserHandler&&) = default;

    // Initialize denoiser with specified dimensions and model type
    bool initialize(uint32_t width, uint32_t height, bool useTemporalDenoiser = true);

    // Clean up denoiser resources
    void finalize();

    // Resize denoiser for new dimensions
    void resize(uint32_t width, uint32_t height);

    // Switch between temporal and HDR denoising models
    void updateDenoiserType(bool useTemporalDenoiser);

    // Setup denoiser state with provided stream (call after initialization/resize/type change)
    void setupState(CUstream stream);

    // Check if denoiser is initialized
    bool isInitialized() const { return initialized_; }

    // Get current denoiser dimensions
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }

    // Denoiser resource access
    const optixu::Denoiser& getDenoiser() const { return denoiser_; }
    const cudau::Buffer& getStateBuffer() const { return stateBuffer_; }
    const cudau::Buffer& getScratchBuffer() const { return scratchBuffer_; }
    const std::vector<optixu::DenoisingTask>& getTasks() const { return tasks_; }

    // Non-const versions for denoising operations
    optixu::Denoiser& getDenoiser() { return denoiser_; }
    cudau::Buffer& getStateBuffer() { return stateBuffer_; }
    cudau::Buffer& getScratchBuffer() { return scratchBuffer_; }
    std::vector<optixu::DenoisingTask>& getTasks() { return tasks_; }

    // Check if temporal denoiser is active
    bool isTemporalDenoiser() const { return isTemporalDenoiser_; }

private:
    bool initialized_ = false;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool isTemporalDenoiser_ = true;
    bool needsStateSetup_ = false;

    // Render context for GPU resources
    RenderContextPtr renderContext_;

    // GPU contexts cached from render context
    optixu::Context optixContext_;
    CUcontext cudaContext_ = nullptr;

    // Denoiser resources
    optixu::Denoiser denoiser_;
    cudau::Buffer stateBuffer_;
    cudau::Buffer scratchBuffer_;
    std::vector<optixu::DenoisingTask> tasks_;

    // Internal methods
    bool createDenoiser(bool useTemporalDenoiser);
    bool setupBuffersAndTasks(uint32_t width, uint32_t height);
};