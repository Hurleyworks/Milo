#pragma once

// DenoiserHandler manages OptiX AI-accelerated denoising functionality.
// Supports both temporal and non-temporal denoising modes for real-time ray tracing.
// Handles denoiser state setup, buffer management, and task configuration.

#include "../common/common_host.h"
#include <memory>

class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
using DenoiserHandlerPtr = std::shared_ptr<class DenoiserHandler>;

class DenoiserHandler
{
public:
    // Factory method following render_core pattern
    static DenoiserHandlerPtr create(RenderContextPtr renderContext);

    // Constructor takes RenderContext instead of GPUContext
    explicit DenoiserHandler(RenderContextPtr renderContext);
    ~DenoiserHandler();

    // Lifecycle management
    bool initialize(uint32_t width, uint32_t height, bool useTemporalDenoiser);
    void finalize();
    void resize(uint32_t width, uint32_t height);
    void updateDenoiserType(bool useTemporalDenoiser);
    void setupState(CUstream stream);

    // Accessors
    const optixu::Denoiser& getDenoiser() const { return denoiser_; }
    const cudau::Buffer& getScratchBuffer() const { return scratchBuffer_; }
    const std::vector<optixu::DenoisingTask>& getTasks() const { return denoisingTasks_; }
    
    bool isInitialized() const { return initialized_; }

private:
    bool initialized_ = false;
    RenderContextPtr renderContext_;
    
    // Denoiser configuration
    bool useTemporalDenoiser_ = true;
    optixu::GuideAlbedo useAlbedo_ = optixu::GuideAlbedo::Yes;
    optixu::GuideNormal useNormal_ = optixu::GuideNormal::Yes;
    
    // Current dimensions
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    
    // OptiX denoiser resources
    optixu::Denoiser denoiser_;
    cudau::Buffer stateBuffer_;
    cudau::Buffer scratchBuffer_;
    std::vector<optixu::DenoisingTask> denoisingTasks_;
    
    // Tiling parameters (use non-tiled like Shocker)
    const uint32_t tileWidth_ = 0;
    const uint32_t tileHeight_ = 0;
    
    // Private helper methods
    void createDenoiser();
    void destroyDenoiser();
    void prepareDenoiser();
};