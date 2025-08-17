#include "RenderContext.h"
#include <g3log/g3log.hpp>

RenderContext::~RenderContext()
{
    cleanup();
}

bool RenderContext::initialize (int deviceIndex)
{
    if (initialized_)
    {
        LOG(WARNING) << "RenderContext already initialized";
        return true;
    }
    
    LOG(INFO) << "Initializing RenderContext...";
    
    // Initialize core GPU context
    if (!initializeCore (deviceIndex))
    {
        LOG(WARNING) << "Failed to initialize core GPU context";
        return false;
    }
    
    // Create OptiX scene
    scene_ = gpu_context_.getOptixContext().createScene();
    LOG(DBUG) << "OptiX scene created";
    
    // Initialize scratch memory for acceleration structure building
    // Start with 32MB, can be resized as needed
    const size_t initial_scratch_size = 32 * 1024 * 1024;
    as_build_scratch_mem_.initialize (
        gpu_context_.getCudaContext(),
        cudau::BufferType::Device,
        initial_scratch_size,
        1);
    LOG(DBUG) << "Allocated " << (initial_scratch_size / (1024 * 1024)) << " MB scratch memory for AS building";
    
    // Initialize handlers
    if (!initializeHandlers())
    {
        LOG(WARNING) << "Failed to initialize handlers";
        cleanup();
        return false;
    }
    
    initialized_ = true;
    LOG(INFO) << "RenderContext initialization complete";
    LOG(INFO) << "  Device: " << gpu_context_.getDeviceName();
    LOG(INFO) << "  Compute Capability: " << gpu_context_.getComputeCapabilityMajor() 
              << "." << gpu_context_.getComputeCapabilityMinor();
    LOG(INFO) << "  Total Memory: " << (gpu_context_.getTotalMemory() / (1024 * 1024)) << " MB";
    
    return true;
}

void RenderContext::cleanup()
{
    if (!initialized_)
    {
        return;
    }
    
    LOG(INFO) << "Cleaning up RenderContext...";
    
    // Clean up handlers first (when implemented)
    cleanupHandlers();
    
    // Clean up scratch memory
    if (as_build_scratch_mem_.isInitialized())
    {
        as_build_scratch_mem_.finalize();
        LOG(DBUG) << "Released AS build scratch memory";
    }
    
    // Destroy scene
    if (scene_)
    {
        scene_.destroy();
        // scene_ = nullptr; // optixu::Scene doesn't support nullptr assignment
        LOG(DBUG) << "OptiX scene destroyed";
    }
    
    // Clean up GPU context
    gpu_context_.finalize();
    
    initialized_ = false;
    LOG(INFO) << "RenderContext cleanup complete";
}

bool RenderContext::initializeCore (int deviceIndex)
{
    // Initialize GPU context with specified device
    if (!gpu_context_.initialize (deviceIndex))
    {
        LOG(WARNING) << "Failed to initialize GPU context for device " << deviceIndex;
        return false;
    }
    
    // Log device capabilities
    LOG(INFO) << "GPU initialized:";
    LOG(INFO) << "  Device Index: " << gpu_context_.getDeviceIndex();
    LOG(INFO) << "  Device Name: " << gpu_context_.getDeviceName();
    LOG(INFO) << "  Compute Capability: " << gpu_context_.getComputeCapabilityMajor()
              << "." << gpu_context_.getComputeCapabilityMinor();
    LOG(INFO) << "  Total Memory: " << (gpu_context_.getTotalMemory() / (1024 * 1024)) << " MB";
    
    // Check minimum requirements for our rendering system
    if (gpu_context_.getComputeCapabilityMajor() < 6)
    {
        LOG(WARNING) << "Compute capability " << gpu_context_.getComputeCapabilityMajor() 
                     << "." << gpu_context_.getComputeCapabilityMinor() 
                     << " is below recommended 6.0";
        LOG(WARNING) << "Some features may not be available or may run slowly";
    }
    
    return true;
}

bool RenderContext::initializeHandlers()
{
    // Initialize handler system
    handlers_ = std::make_unique<dog::Handlers>(getPtr());
    
    // Initialize screen buffers with default render size
    if (handlers_->screenBuffer)
    {
        if (!handlers_->screenBuffer->initialize(render_width_, render_height_))
        {
            LOG(WARNING) << "Failed to initialize screen buffers";
            return false;
        }
        LOG(INFO) << "Screen buffers initialized at " << render_width_ << "x" << render_height_;
    }
    
    LOG(INFO) << "Handlers initialized";
    return true;
}

void RenderContext::cleanupHandlers()
{
    // Clean up handlers
    if (handlers_)
    {
        handlers_.reset();
        LOG(DBUG) << "Handlers released";
    }
}