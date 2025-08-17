#pragma once

#include "../dog_core.h"
#include "GPUContext.h"
#include "common/common_host.h"
#include "handlers/Handlers.h"

// Forward declarations
using RenderContextPtr = std::shared_ptr<class RenderContext>;

// Central coordination class for the dog_core rendering system
// Manages GPU contexts, handlers, and shared resources
// Acts as the main hub that all rendering components interact through
class RenderContext : public std::enable_shared_from_this<RenderContext>
{
 public:
    // Factory method to create a shared_ptr instance
    static RenderContextPtr create() { return std::make_shared<RenderContext>(); }
    
    // Returns shared_ptr to this object
    RenderContextPtr getPtr() { return shared_from_this(); }

    RenderContext() = default;
    ~RenderContext();

    // Initialize the render context with optional device selection
    bool initialize (int deviceIndex = 0);
    
    // Clean up all resources
    void cleanup();
    
    // Check if context is initialized
    bool isInitialized() const { return initialized_; }

    // GPU context access
    GPUContext& getGPUContext() { return gpu_context_; }
    const GPUContext& getGPUContext() const { return gpu_context_; }
    
    // Convenience accessors for commonly used GPU resources
    CUcontext getCudaContext() const { return gpu_context_.getCudaContext(); }
    CUstream getCudaStream() const { return gpu_context_.getCudaStream(); }
    optixu::Context getOptixContext() const { return gpu_context_.getOptixContext(); }
    
    // Device information
    int getComputeCapability() const { return gpu_context_.getComputeCapability(); }
    int getComputeCapabilityMajor() const { return gpu_context_.getComputeCapabilityMajor(); }
    int getComputeCapabilityMinor() const { return gpu_context_.getComputeCapabilityMinor(); }
    std::string getDeviceName() const { return gpu_context_.getDeviceName(); }
    size_t getTotalMemory() const { return gpu_context_.getTotalMemory(); }

    // Scene management
    optixu::Scene getScene() const { return scene_; }
    void setScene (optixu::Scene scene) { scene_ = scene; }
    
    // Render size management - dimensions are automatically managed by camera
    int getRenderWidth() const { return render_width_; }
    int getRenderHeight() const { return render_height_; }
    
    // Handlers access
    dog::Handlers* getHandlers() const { return handlers_.get(); }
    
    // Resource paths
    void setResourcePath (const std::filesystem::path& path) { resource_path_ = path; }
    std::filesystem::path getResourcePath() const { return resource_path_; }
    
    // Camera management
    void setCamera(sabi::CameraHandle camera) { camera_ = camera; }
    sabi::CameraHandle getCamera() const { return camera_; }
    
    // Check and update render dimensions from camera (call per frame or when needed)
    bool updateRenderDimensionsFromCamera();

 private:
    bool initialized_ = false;
    
    // Core GPU context
    GPUContext gpu_context_;
    
    // OptiX scene
    optixu::Scene scene_;
    
    // Render dimensions
    int render_width_ = 1920;
    int render_height_ = 1080;
    
    // Resource management
    std::filesystem::path resource_path_;
    
    // Handler system
    std::unique_ptr<dog::Handlers> handlers_;
    
    // Note: Acceleration structure scratch memory will be managed by SceneHandler/ModelHandler
    // when those components are implemented, not here in RenderContext
    
    // Camera (stub for now)
    sabi::CameraHandle camera_ = nullptr;
    
    // Helper methods
    bool initializeCore (int deviceIndex);
    bool initializeHandlers();
    void cleanupHandlers();
};