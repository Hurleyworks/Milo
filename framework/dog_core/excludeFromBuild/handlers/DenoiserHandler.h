// DenoiserHandler manages OptiX AI denoiser for the Dog rendering system.
// It provides centralized management of denoising resources and operations.
//
// Denoiser Features:
// - Supports both temporal and HDR denoising models
// - Dynamic switching between denoising modes
// - Automatic buffer management for denoiser state
// - Integration with rendering buffers for denoising passes
//
// Resource Management:
// - Manages OptiX denoiser instance lifecycle
// - Handles state and scratch buffer allocation
// - Supports dynamic resizing for resolution changes
// - Automatic cleanup of GPU resources
//
// Denoising Pipeline:
// - Setup denoiser state after initialization/resize
// - Compute intensity for HDR tone mapping
// - Execute denoising with guide layers
// - Support for temporal accumulation
//
// Integration:
// - Works with ScreenBufferHandler for input/output buffers
// - Integrates with OptiX context from RenderContext
// - Supports CUDA stream operations
// - Compatible with progressive rendering
//
// Usage:
// - Create via factory method DenoiserHandler::create()
// - Initialize with render context and dimensions
// - Setup state before first denoising operation
// - Execute denoising passes after ray tracing
//
// Thread Safety:
// - Not thread-safe by default
// - Requires external synchronization for multi-threaded access
// - Denoising operations should be synchronized with rendering

#pragma once

#include "../common/common_host.h"

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

namespace dog
{

using DenoiserHandlerPtr = std::shared_ptr<class DenoiserHandler>;

// DenoiserHandler manages OptiX denoising operations for the rendering pipeline
// Provides centralized denoiser lifecycle management with automatic cleanup
class DenoiserHandler
{
public:
    // Factory method to create a shared DenoiserHandler instance
    static DenoiserHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<DenoiserHandler>(ctx);
    }

    DenoiserHandler(RenderContextPtr ctx);
    ~DenoiserHandler();

    DenoiserHandler(const DenoiserHandler&) = delete;
    DenoiserHandler& operator=(const DenoiserHandler&) = delete;
    DenoiserHandler(DenoiserHandler&&) = default;
    DenoiserHandler& operator=(DenoiserHandler&&) = default;

    // Initialize denoiser with specified dimensions and model type
    // Returns true if successful, false otherwise
    bool initialize(uint32_t width, uint32_t height, bool useTemporalDenoiser = true);

    // Clean up all denoiser resources
    void finalize();

    // Resize denoiser for new dimensions
    // Preserves initialization state if already initialized
    void resize(uint32_t width, uint32_t height);

    // Switch between temporal and HDR denoising models
    // Requires state setup after switching
    void updateDenoiserType(bool useTemporalDenoiser);

    // Setup denoiser state with provided stream
    // Must be called after initialization/resize/type change
    void setupState(CUstream stream);

    // Compute intensity for HDR tone mapping
    // Returns average intensity value for exposure adjustment
    float computeIntensity(CUstream stream, 
                          const cudau::TypedBuffer<float4>& beautyBuffer,
                          uint32_t width, uint32_t height);

    // Execute denoising with guide layers
    // Input buffers should be in linear space
    void denoise(CUstream stream,
                const cudau::TypedBuffer<float4>& beautyBuffer,
                const cudau::TypedBuffer<float4>& albedoBuffer,
                const cudau::TypedBuffer<float4>& normalBuffer,
                const cudau::TypedBuffer<float2>& flowBuffer,
                cudau::TypedBuffer<float4>& denoisedBuffer,
                float blendFactor = 0.0f,
                bool useTemporalMode = true);

    // Check if denoiser is initialized
    bool isInitialized() const { return initialized_; }

    // Get current denoiser configuration
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }
    bool isTemporalMode() const { return is_temporal_; }
    bool needsStateSetup() const { return needs_state_setup_; }

    // Get memory usage statistics
    size_t getStateBufferSize() const;
    size_t getScratchBufferSize() const;
    size_t getTotalMemoryUsage() const;

private:
    // Internal methods
    bool createDenoiser(bool useTemporalDenoiser);
    bool setupBuffersAndTasks(uint32_t width, uint32_t height);
    // Note: Input preparation is done directly in denoise() method

    // Member variables
    RenderContextPtr render_context_;
    bool initialized_ = false;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool is_temporal_ = true;
    bool needs_state_setup_ = false;

    // Denoiser resources
    optixu::Denoiser denoiser_;
    cudau::Buffer state_buffer_;
    cudau::Buffer scratch_buffer_;
    cudau::Buffer intensity_buffer_;
    std::vector<optixu::DenoisingTask> tasks_;
};

} // namespace dog