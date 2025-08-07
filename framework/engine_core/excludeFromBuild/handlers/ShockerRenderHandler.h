#pragma once

// ShockerRenderHandler - Render handler for Shocker path tracing engine
// Manages accumulation buffers and rendering outputs for Shocker's advanced rendering pipeline
// Provides support for beauty, albedo, normal, and motion vector buffers with denoising integration

#include "../common/common_host.h"

class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
using ShockerRenderHandlerPtr = std::shared_ptr<class ShockerRenderHandler>;

class ShockerRenderHandler
{
public:
    // Factory method
    static ShockerRenderHandlerPtr create(RenderContextPtr ctx) 
    { 
        return std::make_shared<ShockerRenderHandler>(ctx); 
    }
    
    ShockerRenderHandler(RenderContextPtr ctx);
    ~ShockerRenderHandler();
    
    // Initialize handler with dimensions
    bool initialize(uint32_t width, uint32_t height);
    void finalize();
    
    // Get surface objects for launch parameters
    CUsurfObject getBeautyAccumSurfaceObject() const 
    { 
        return beautyAccumBuffer_.isInitialized() ? beautyAccumBuffer_.getSurfaceObject(0) : 0; 
    }
    
    CUsurfObject getAlbedoAccumSurfaceObject() const 
    { 
        return albedoAccumBuffer_.isInitialized() ? albedoAccumBuffer_.getSurfaceObject(0) : 0; 
    }
    
    CUsurfObject getNormalAccumSurfaceObject() const 
    { 
        return normalAccumBuffer_.isInitialized() ? normalAccumBuffer_.getSurfaceObject(0) : 0; 
    }
    
    CUsurfObject getMotionAccumSurfaceObject() const 
    { 
        return motionAccumBuffer_.isInitialized() ? motionAccumBuffer_.getSurfaceObject(0) : 0; 
    }
    
    // State queries
    bool isInitialized() const { return initialized_; }
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }
    
    // Linear buffer accessors (for display and denoising)
    cudau::TypedBuffer<float4>& getLinearBeautyBuffer() { return linearBeautyBuffer_; }
    cudau::TypedBuffer<float4>& getLinearAlbedoBuffer() { return linearAlbedoBuffer_; }
    cudau::TypedBuffer<float4>& getLinearNormalBuffer() { return linearNormalBuffer_; }
    cudau::TypedBuffer<float2>& getLinearMotionBuffer() { return linearMotionBuffer_; }
    
    const cudau::TypedBuffer<float4>& getLinearBeautyBuffer() const { return linearBeautyBuffer_; }
    const cudau::TypedBuffer<float4>& getLinearAlbedoBuffer() const { return linearAlbedoBuffer_; }
    const cudau::TypedBuffer<float4>& getLinearNormalBuffer() const { return linearNormalBuffer_; }
    const cudau::TypedBuffer<float2>& getLinearMotionBuffer() const { return linearMotionBuffer_; }
    
    // Pick info buffer accessor
    float4* getPickInfoPointer(int bufferIndex) const
    {
        return pickInfoBuffers_[bufferIndex].isInitialized() ? 
               pickInfoBuffers_[bufferIndex].getDevicePointer() : nullptr;
    }
    
    // Operations
    void copyAccumToLinearBuffers(CUstream stream);
    void clearAccumBuffers(CUstream stream);
    
    // Denoising operation
    // Returns true if denoising was performed, false if denoiser not available
    bool denoise(CUstream stream, bool isNewSequence, class ShockerDenoiserHandler* denoiserHandler, struct GPUTimerManager::GPUTimer* timer = nullptr);
    
    // Resize support
    bool resize(uint32_t newWidth, uint32_t newHeight);
    
    // Get accumulation count (for progressive rendering)
    uint32_t getAccumulationCount() const { return accumulationCount_; }
    void resetAccumulationCount() { accumulationCount_ = 0; }
    void incrementAccumulationCount() { accumulationCount_++; }
    
private:
    RenderContextPtr renderContext_;
    bool initialized_ = false;
    
    // Current dimensions
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    
    // Accumulation counter for progressive rendering
    uint32_t accumulationCount_ = 0;
    
    // Accumulation buffers (2D arrays for Shocker path tracing output)
    cudau::Array beautyAccumBuffer_;
    cudau::Array albedoAccumBuffer_;
    cudau::Array normalAccumBuffer_;
    cudau::Array motionAccumBuffer_;  // Motion vectors for temporal denoising
    
    // Linear buffers for display and denoising
    cudau::TypedBuffer<float4> linearBeautyBuffer_;
    cudau::TypedBuffer<float4> linearAlbedoBuffer_;
    cudau::TypedBuffer<float4> linearNormalBuffer_;
    cudau::TypedBuffer<float2> linearMotionBuffer_;  // Motion vectors are float2
    
    // Pick info buffers (double buffered)
    cudau::TypedBuffer<float4> pickInfoBuffers_[2];
    
    // CUDA module and kernels for buffer operations
    CUmodule moduleShockerCopyBuffers_ = nullptr;
    cudau::Kernel kernelCopySurfacesToLinear_;
    cudau::Kernel kernelClearAccumBuffers_;
    
    // Private methods
    void loadKernels();
    void cleanupKernels();
};