// ScreenBufferHandler manages all screen-related rendering buffers for the Dog rendering system.
// It provides centralized management of GPU buffers required for path tracing and denoising.
//
// Resource Management:
// - Manages G-buffers for geometry and shading information
// - Handles accumulation buffers for progressive rendering
// - Controls linear buffers for denoising pipeline
// - Manages RNG buffer for Monte Carlo sampling
//
// Buffer Types:
// - G-buffers: Store per-pixel geometry and material properties (2 sets for double buffering)
// - Accumulation buffers: Accumulate beauty, albedo, and normal passes over multiple samples
// - Linear buffers: Store linearized data for denoising and post-processing
// - RNG buffer: Stores per-pixel random number generator states
//
// Memory Management:
// - Automatic CUDA memory allocation and deallocation
// - Proper resource cleanup in destructor
// - Support for dynamic buffer resizing
// - RAII principles for resource safety
//
// Integration:
// - Works with OptiX ray tracing pipeline
// - Provides surface objects for kernel access
// - Supports denoising workflow
// - Integrates with CUDA runtime
//
// Usage:
// - Create via factory method ScreenBufferHandler::create()
// - Initialize with GPU context and dimensions
// - Access buffers through getter methods
// - Resize buffers when window dimensions change
//
// Thread Safety:
// - Not thread-safe by default
// - Requires external synchronization for multi-threaded access
// - Buffer operations should be synchronized with rendering

#pragma once

#include "../common/common_host.h"

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

namespace dog
{

using ScreenBufferHandlerPtr = std::shared_ptr<class ScreenBufferHandler>;

// ScreenBufferHandler manages all screen-related GPU buffers for rendering
// Provides centralized buffer lifecycle management with automatic cleanup
class ScreenBufferHandler
{
public:
    // Factory method to create a shared ScreenBufferHandler instance
    static ScreenBufferHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<ScreenBufferHandler>(ctx);
    }

    ScreenBufferHandler(RenderContextPtr ctx);
    ~ScreenBufferHandler();

    ScreenBufferHandler(const ScreenBufferHandler&) = delete;
    ScreenBufferHandler& operator=(const ScreenBufferHandler&) = delete;
    ScreenBufferHandler(ScreenBufferHandler&&) = default;
    ScreenBufferHandler& operator=(ScreenBufferHandler&&) = default;

    // Initialize all screen buffers with specified dimensions
    // Returns true if successful, false otherwise
    bool initialize(uint32_t width, uint32_t height);

    // Clean up all screen buffers
    void finalize();

    // Resize all screen buffers to new dimensions
    // Preserves initialization state if already initialized
    void resize(uint32_t width, uint32_t height);

    // Check if buffers are initialized
    bool isInitialized() const { return initialized_; }

    // Get current buffer dimensions
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }

    // G-buffer access (index 0 or 1 for double buffering)
    const cudau::Array& getGBuffer0(int index) const { return gbuffers_.gBuffer0[index]; }
    const cudau::Array& getGBuffer1(int index) const { return gbuffers_.gBuffer1[index]; }

    // Accumulation buffer access
    const cudau::Array& getBeautyAccumBuffer() const { return accumulation_buffers_.beautyAccumBuffer; }
    const cudau::Array& getAlbedoAccumBuffer() const { return accumulation_buffers_.albedoAccumBuffer; }
    const cudau::Array& getNormalAccumBuffer() const { return accumulation_buffers_.normalAccumBuffer; }

    // Linear buffer access for denoising pipeline
    const cudau::TypedBuffer<float4>& getLinearBeautyBuffer() const { return linear_buffers_.linearBeautyBuffer; }
    const cudau::TypedBuffer<float4>& getLinearAlbedoBuffer() const { return linear_buffers_.linearAlbedoBuffer; }
    const cudau::TypedBuffer<float4>& getLinearNormalBuffer() const { return linear_buffers_.linearNormalBuffer; }
    const cudau::TypedBuffer<float2>& getLinearFlowBuffer() const { return linear_buffers_.linearFlowBuffer; }
    const cudau::TypedBuffer<float4>& getLinearDenoisedBeautyBuffer() const { return linear_buffers_.linearDenoisedBeautyBuffer; }

    // RNG buffer access for random number generation
    const cudau::Array& getRngBuffer() const { return rng_buffer_; }

    // Surface object access for OptiX kernel parameters
    auto getRngBufferSurfaceObject() const { return rng_buffer_.getSurfaceObject(0); }
    auto getGBuffer0SurfaceObject(int index) const { return gbuffers_.gBuffer0[index].getSurfaceObject(0); }
    auto getGBuffer1SurfaceObject(int index) const { return gbuffers_.gBuffer1[index].getSurfaceObject(0); }
    auto getBeautyAccumSurfaceObject() const { return accumulation_buffers_.beautyAccumBuffer.getSurfaceObject(0); }
    auto getAlbedoAccumSurfaceObject() const { return accumulation_buffers_.albedoAccumBuffer.getSurfaceObject(0); }
    auto getNormalAccumSurfaceObject() const { return accumulation_buffers_.normalAccumBuffer.getSurfaceObject(0); }

private:
    RenderContextPtr ctx_ = nullptr;
    bool initialized_ = false;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    // G-buffer storage for geometry and material properties
    struct GBuffers
    {
        cudau::Array gBuffer0[2];  // Double buffered G-buffer set 0
        cudau::Array gBuffer1[2];  // Double buffered G-buffer set 1

        void initialize(CUcontext cuContext, uint32_t width, uint32_t height);
        void resize(uint32_t width, uint32_t height);
        void finalize();
    };

    // Accumulation buffer storage for progressive rendering
    struct AccumulationBuffers
    {
        cudau::Array beautyAccumBuffer;  // Accumulated beauty pass
        cudau::Array albedoAccumBuffer;  // Accumulated albedo pass
        cudau::Array normalAccumBuffer;  // Accumulated normal pass

        void initialize(CUcontext cuContext, uint32_t width, uint32_t height);
        void resize(uint32_t width, uint32_t height);
        void finalize();
    };

    // Linear buffer storage for denoising and post-processing
    struct LinearBuffers
    {
        cudau::TypedBuffer<float4> linearBeautyBuffer;         // Linear beauty data
        cudau::TypedBuffer<float4> linearAlbedoBuffer;         // Linear albedo data
        cudau::TypedBuffer<float4> linearNormalBuffer;         // Linear normal data
        cudau::TypedBuffer<float2> linearFlowBuffer;           // Motion vectors
        cudau::TypedBuffer<float4> linearDenoisedBeautyBuffer; // Denoised output

        void initialize(CUcontext cuContext, uint32_t width, uint32_t height);
        void resize(uint32_t width, uint32_t height);
        void finalize();
    };

    GBuffers gbuffers_;
    AccumulationBuffers accumulation_buffers_;
    LinearBuffers linear_buffers_;
    cudau::Array rng_buffer_;  // Random number generator states

    // Internal RNG buffer management
    bool initializeRngBuffer(uint32_t width, uint32_t height);
    void finalizeRngBuffer();
    void resizeRngBuffer(uint32_t width, uint32_t height);
};

} // namespace dog