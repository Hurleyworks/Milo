#pragma once

// ShockerRenderHandler - Render handler for ShockerEngine
// Manages accumulation buffers that Shocker path tracer writes to
// Based on ShockerRenderHandler but adapted for Shocker's specific needs

#include "../../../common/common_host.h"

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
    
    CUsurfObject getFlowAccumSurfaceObject() const 
    { 
        return flowAccumBuffer_.isInitialized() ? flowAccumBuffer_.getSurfaceObject(0) : 0; 
    }
    
    // State queries
    bool isInitialized() const { return initialized_; }
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }
    
    // Linear buffer accessors (for display and denoising)
    cudau::TypedBuffer<float4>& getLinearBeautyBuffer() { return linearBeautyBuffer_; }
    cudau::TypedBuffer<float4>& getLinearAlbedoBuffer() { return linearAlbedoBuffer_; }
    cudau::TypedBuffer<float4>& getLinearNormalBuffer() { return linearNormalBuffer_; }
    cudau::TypedBuffer<float2>& getLinearFlowBuffer() { return linearFlowBuffer_; }
    
    
    // Pick info buffer accessor
    shocker_shared::PickInfo* getPickInfoPointer(int bufferIndex) const
    {
        return pickInfoBuffers_[bufferIndex].isInitialized() ? 
               pickInfoBuffers_[bufferIndex].getDevicePointer() : nullptr;
    }
    
    // Get pick info buffer for reading (similar to Shocker renderer)
    const cudau::TypedBuffer<shocker_shared::PickInfo>& getPickInfo(int bufferIndex) const
    {
        return pickInfoBuffers_[bufferIndex];
    }
    
    // Operations
    void copyAccumToLinearBuffers(CUstream stream);
    void clearAccumBuffers(CUstream stream);
    
    // Denoising operation
    // Returns true if denoising was performed, false if denoiser not available
    bool denoise(CUstream stream, bool isNewSequence, class DenoiserHandler* denoiserHandler, struct GPUTimerManager::GPUTimer* timer = nullptr);
    
    // Resize support
    bool resize(uint32_t newWidth, uint32_t newHeight);
    
private:
    RenderContextPtr renderContext_;
    bool initialized_ = false;
    
    // Current dimensions
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    
    // Accumulation buffers (2D arrays for Shocker path tracing output)
    cudau::Array beautyAccumBuffer_;
    cudau::Array albedoAccumBuffer_;
    cudau::Array normalAccumBuffer_;
    cudau::Array flowAccumBuffer_;  // Motion vectors
    
    // Linear buffers for display and denoising
    cudau::TypedBuffer<float4> linearBeautyBuffer_;
    cudau::TypedBuffer<float4> linearAlbedoBuffer_;
    cudau::TypedBuffer<float4> linearNormalBuffer_;
    cudau::TypedBuffer<float2> linearFlowBuffer_;  // Motion vectors
    
    
    // Pick info buffers (double buffered)
    cudau::TypedBuffer<shocker_shared::PickInfo> pickInfoBuffers_[2];
    
    // CUDA module and kernels for buffer operations
    CUmodule moduleCopyBuffers_ = nullptr;
    cudau::Kernel kernelCopySurfacesToLinear_;
    
    // Private methods
    void loadKernels();
    void cleanupKernels();
};