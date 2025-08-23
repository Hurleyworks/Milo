#pragma once

#include "../common/common_host.h"
#include "../common/common_shared.h"  // For PCG32RNG and PickInfo
#include "../RenderContext.h"
#include <array>

class ScreenBufferHandler;
using ScreenBufferHandlerPtr = std::shared_ptr<ScreenBufferHandler>;

// Centralized screen buffer management for rendering pipelines
// Manages G-buffers, accumulation buffers, linear buffers, and RNG states
// Provides efficient buffer lifecycle management with proper CUDA memory handling
class ScreenBufferHandler
{
public:
    // Factory method following standard handler pattern
    static ScreenBufferHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<ScreenBufferHandler>(ctx);
    }

    // Constructor/Destructor
    explicit ScreenBufferHandler(RenderContextPtr ctx);
    ~ScreenBufferHandler();

    ScreenBufferHandler(const ScreenBufferHandler&) = delete;
    ScreenBufferHandler& operator=(const ScreenBufferHandler&) = delete;
    ScreenBufferHandler(ScreenBufferHandler&&) = default;
    ScreenBufferHandler& operator=(ScreenBufferHandler&&) = default;

    // Initialize all screen buffers
    bool initialize(uint32_t width, uint32_t height);

    // Clean up all screen buffers
    void finalize();

    // Resize all screen buffers efficiently
    bool resize(uint32_t width, uint32_t height);

    // Reset buffer contents without reallocation
    void clear();

    // Check if buffers are initialized
    bool isInitialized() const { return initialized_; }

    // Get current buffer dimensions
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }
    uint32_t getPixelCount() const { return width_ * height_; }

    // Current frame index for double buffering
    uint32_t getCurrentFrameIndex() const { return currentFrameIndex_; }
    void swapFrameBuffers() { currentFrameIndex_ = 1 - currentFrameIndex_; }

    // G-buffer access (double buffered for temporal operations)
    const cudau::Array& getGBuffer0(int index) const { return gbuffers_.gBuffer0[index]; }
    const cudau::Array& getGBuffer1(int index) const { return gbuffers_.gBuffer1[index]; }

    // Accumulation buffer access
    const cudau::Array& getBeautyAccumBuffer() const { return accumBuffers_.beautyAccumBuffer; }
    const cudau::Array& getAlbedoAccumBuffer() const { return accumBuffers_.albedoAccumBuffer; }
    const cudau::Array& getNormalAccumBuffer() const { return accumBuffers_.normalAccumBuffer; }
    const cudau::Array& getFlowAccumBuffer() const { return accumBuffers_.flowAccumBuffer; }

    // Linear buffer access for display/denoising
    const cudau::TypedBuffer<float4>& getLinearBeautyBuffer() const { return linearBuffers_.linearBeautyBuffer; }
    const cudau::TypedBuffer<float4>& getLinearAlbedoBuffer() const { return linearBuffers_.linearAlbedoBuffer; }
    const cudau::TypedBuffer<float4>& getLinearNormalBuffer() const { return linearBuffers_.linearNormalBuffer; }
    const cudau::TypedBuffer<float2>& getLinearFlowBuffer() const { return linearBuffers_.linearFlowBuffer; }
    const cudau::TypedBuffer<float4>& getLinearDenoisedBeautyBuffer() const { return linearBuffers_.linearDenoisedBeautyBuffer; }

    // RNG buffer access
    const cudau::Array& getRngBuffer() const { return rngBuffer_; }

    // Pick info buffer access (double buffered)
    const cudau::TypedBuffer<shared::PickInfo>& getPickInfoBuffer(int index = -1) const 
    { 
        return pickInfoBuffers_[index >= 0 ? index : currentFrameIndex_]; 
    }
    
    // Get pick info pointer for direct kernel access
    shared::PickInfo* getPickInfoPointer(int bufferIndex) const
    {
        return pickInfoBuffers_[bufferIndex].isInitialized() ? 
               pickInfoBuffers_[bufferIndex].getDevicePointer() : nullptr;
    }

    // Surface object access for OptiX pipeline parameters
    auto getRngBufferSurfaceObject() const { return rngBuffer_.getSurfaceObject(0); }
    auto getGBuffer0SurfaceObject(int index) const { return gbuffers_.gBuffer0[index].getSurfaceObject(0); }
    auto getGBuffer1SurfaceObject(int index) const { return gbuffers_.gBuffer1[index].getSurfaceObject(0); }
    auto getBeautyAccumSurfaceObject() const { return accumBuffers_.beautyAccumBuffer.getSurfaceObject(0); }
    auto getAlbedoAccumSurfaceObject() const { return accumBuffers_.albedoAccumBuffer.getSurfaceObject(0); }
    auto getNormalAccumSurfaceObject() const { return accumBuffers_.normalAccumBuffer.getSurfaceObject(0); }
    auto getFlowAccumSurfaceObject() const { return accumBuffers_.flowAccumBuffer.getSurfaceObject(0); }

    // Buffer pointers for direct kernel access
    float4* getLinearBeautyPointer() { return linearBuffers_.linearBeautyBuffer.getDevicePointer(); }
    float4* getLinearAlbedoPointer() { return linearBuffers_.linearAlbedoBuffer.getDevicePointer(); }
    float4* getLinearNormalPointer() { return linearBuffers_.linearNormalBuffer.getDevicePointer(); }
    float2* getLinearFlowPointer() { return linearBuffers_.linearFlowBuffer.getDevicePointer(); }
    float4* getLinearDenoisedBeautyPointer() { return linearBuffers_.linearDenoisedBeautyBuffer.getDevicePointer(); }

    // Copy accumulation buffers to linear buffers for display
    void copyAccumToLinearBuffers(CUstream stream = 0);

    // Initialize RNG states with seed
    bool initializeRngStates(uint64_t seed = 0);

    // Get buffer size information
    size_t getTotalGPUMemoryUsage() const;

private:
    bool initialized_ = false;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t currentFrameIndex_ = 0;
    
    RenderContextPtr renderContext_;
    CUcontext cudaContext_ = nullptr;
    
    // CUDA module and kernels for buffer operations
    CUmodule moduleCopyBuffers_ = nullptr;
    CUfunction kernelCopyAccumToLinear_ = nullptr;

    // G-buffer storage
    struct GBuffers
    {
        cudau::Array gBuffer0[2];
        cudau::Array gBuffer1[2];

        void initialize(CUcontext cuContext, uint32_t width, uint32_t height);
        void resize(uint32_t width, uint32_t height);
        void finalize();
    };

    // Accumulation buffer storage
    struct AccumulationBuffers
    {
        cudau::Array beautyAccumBuffer;
        cudau::Array albedoAccumBuffer;
        cudau::Array normalAccumBuffer;
        cudau::Array flowAccumBuffer;  // Motion vectors accumulation

        void initialize(CUcontext cuContext, uint32_t width, uint32_t height);
        void resize(uint32_t width, uint32_t height);
        void finalize();
    };

    // Linear buffer storage for display and denoising
    struct LinearBuffers
    {
        cudau::TypedBuffer<float4> linearBeautyBuffer;
        cudau::TypedBuffer<float4> linearAlbedoBuffer;
        cudau::TypedBuffer<float4> linearNormalBuffer;
        cudau::TypedBuffer<float2> linearFlowBuffer;
        cudau::TypedBuffer<float4> linearDenoisedBeautyBuffer;

        void initialize(CUcontext cuContext, uint32_t width, uint32_t height);
        void resize(uint32_t width, uint32_t height);
        void finalize();
    };

    GBuffers gbuffers_;
    AccumulationBuffers accumBuffers_;
    LinearBuffers linearBuffers_;
    cudau::Array rngBuffer_;
    std::array<cudau::TypedBuffer<shared::PickInfo>, 2> pickInfoBuffers_;

    // Internal methods
    bool initializeRngBuffer(CUcontext cuContext, uint32_t width, uint32_t height);
    void finalizeRngBuffer();
    void resizeRngBuffer(CUcontext cuContext, uint32_t width, uint32_t height);
    
    bool loadKernels();
    void cleanupKernels();
    
    // Validate buffer dimensions
    bool validateDimensions(uint32_t width, uint32_t height) const;
};