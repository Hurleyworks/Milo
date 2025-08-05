#pragma once

// RenderContext aggregates all rendering subsystems and resources.
// Manages GPU context, handlers, camera, and property services.
// Provides a unified interface for pipelines to access rendering infrastructure.

#include "sabi_core/sabi_core.h"
#include "RenderUtilities.h"
#include "GPUContext.h"

using RenderContextPtr = std::shared_ptr<class RenderContext>;
using sabi::CameraHandle;
using ImageCacheHandlerPtr = std::shared_ptr<class ImageCacheHandler>;
class PropertyService;

// Forward declaration for Handlers struct
struct Handlers;

#include "handlers/Handlers.h"

class RenderContext : public std::enable_shared_from_this<RenderContext>
{
 public:
    // Returns shared_ptr to this object
    RenderContextPtr getPtr() { return shared_from_this(); }

    RenderContext() = default;
    ~RenderContext() = default;

    // Initialize GPU context and handlers
    bool initialize(bool skipPipelineInit = false)
    {
        if (!ctx.initialize())
        {
            return false;
        }
        
        // Initialize acceleration structure build scratch memory
        // Start with 1MB, it will be resized as needed
        asBuildScratchMem_.initialize(ctx.getCudaContext(), cudau::BufferType::Device, 1024 * 1024, 1);
        
        handlers.initialize (shared_from_this(), skipPipelineInit);
        return true;
    }

    // Initialize with system resources
    bool initialize (CameraHandle camera, ImageCacheHandlerPtr imageCache, const PropertyService& properties, bool skipPipelineInit = false)
    {
        if (!ctx.initialize())
        {
            return false;
        }

        // Store system resources
        camera_ = camera;
        imageCache_ = imageCache;
        properties_ = properties;
        
        // Initialize acceleration structure build scratch memory
        // Start with 1MB, it will be resized as needed
        asBuildScratchMem_.initialize(ctx.getCudaContext(), cudau::BufferType::Device, 1024 * 1024, 1);

        handlers.initialize (shared_from_this(), skipPipelineInit);
        return true;
    }

    void cleanup()
    {
        handlers.cleanup();
        
        // Clean up acceleration structure scratch memory
        if (asBuildScratchMem_.isInitialized()) {
            asBuildScratchMem_.finalize();
        }
        
        ctx.cleanup();
    }

    bool isInitialized() const { return ctx.isInitialized(); }

    // Getters for context resources
    CUcontext getCudaContext() const { return ctx.getCudaContext(); }
    CUstream getCudaStream() const { return ctx.getCudaStream(); }
    optixu::Context getOptiXContext() const { return ctx.getOptiXContext(); }
    
    // Get pointer to OptiX context (for pipeline initialization)
    optixu::Context* getOptiXContextPtr() { return ctx.getOptiXContextPtr(); }

    // PTX Manager access
    void setPTXManager (PTXManager* ptxManager) { ctx.setPTXManager (ptxManager); }
    PTXManager* getPTXManager() const { return ctx.getPTXManager(); }

    // Handlers access
    Handlers& getHandlers() { return handlers; }
    const Handlers& getHandlers() const { return handlers; }

    // System resources access
    CameraHandle getCamera() const { return camera_; }
    ImageCacheHandlerPtr getImageCache() const { return imageCache_; }
    const PropertyService& getPropertyService() const { return properties_; }

    // Acceleration structure scratch memory access
    cudau::Buffer& getASBuildScratchMem() { return asBuildScratchMem_; }

 private:
    GPUContext ctx;
    Handlers handlers;

    // System resources
    CameraHandle camera_;
    ImageCacheHandlerPtr imageCache_;
    PropertyService properties_;
    
    // Acceleration structure scratch memory
    cudau::Buffer asBuildScratchMem_;
};