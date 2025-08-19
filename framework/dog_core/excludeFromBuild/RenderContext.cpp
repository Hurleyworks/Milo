#include "RenderContext.h"
#include <g3log/g3log.hpp>
#include <sabi_core/sabi_core.h>

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
    
    // Update render dimensions from camera if available
    updateRenderDimensionsFromCamera();
    
    // Initialize core GPU context
    if (!initializeCore (deviceIndex))
    {
        LOG(WARNING) << "Failed to initialize core GPU context";
        return false;
    }
    
    // Create OptiX scene
    scene_ = gpu_context_.getOptixContext().createScene();
    LOG(DBUG) << "OptiX scene created";
    
    // Note: Acceleration structure scratch memory will be allocated by SceneHandler/ModelHandler
    // when those components are implemented
    
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
    
    // Note: AS scratch memory cleanup will be handled by SceneHandler/ModelHandler
    
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
    
    // Set default resource path if not already set
    if (resource_path_.empty())
    {
        resource_path_ = std::filesystem::current_path() / "resources" / "RenderDog";
        LOG(INFO) << "Set default resource path: " << resource_path_.string();
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
    
    // Update render dimensions from camera if available
    updateRenderDimensionsFromCamera();
    
    // Initialize screen buffers with render size
    if (handlers_->screenBuffer)
    {
        if (!handlers_->screenBuffer->initialize(render_width_, render_height_))
        {
            LOG(WARNING) << "Failed to initialize screen buffers";
            return false;
        }
        LOG(INFO) << "Screen buffers initialized at " << render_width_ << "x" << render_height_;
    }
    
    // Initialize pipeline handler
    if (handlers_->pipeline)
    {
        if (!handlers_->pipeline->initialize())
        {
            LOG(WARNING) << "Failed to initialize pipeline handler";
            return false;
        }
        LOG(INFO) << "Pipeline handler initialized";
    }
    
    // Initialize pipeline parameter handler
    if (handlers_->pipelineParameter)
    {
        if (!handlers_->pipelineParameter->initialize())
        {
            LOG(WARNING) << "Failed to initialize pipeline parameter handler";
            return false;
        }
        LOG(INFO) << "Pipeline parameter handler initialized";
    }
    
    // Initialize denoiser handler with temporal denoising by default
    if (handlers_->denoiser)
    {
        if (!handlers_->denoiser->initialize(render_width_, render_height_, true))
        {
            LOG(WARNING) << "Failed to initialize denoiser handler";
            return false;
        }
        LOG(INFO) << "Denoiser handler initialized at " << render_width_ << "x" << render_height_;
    }
    
    // Initialize scene handler
    if (handlers_->scene)
    {
        if (!handlers_->scene->initialize())
        {
            LOG(WARNING) << "Failed to initialize scene handler";
            return false;
        }
        LOG(INFO) << "Scene handler initialized";
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

bool RenderContext::updateRenderDimensionsFromCamera()
{
    if (!camera_)
    {
        LOG(DBUG) << "No camera set, cannot update render dimensions";
        return false;
    }
    
    if (!camera_->getSensor())
    {
        LOG(DBUG) << "Camera has no sensor, cannot update render dimensions";
        return false;
    }
    
    // Extract render dimensions from camera sensor
    Eigen::Vector2i resolution = camera_->getSensor()->getPixelResolution();
    int newWidth = resolution.x();
    int newHeight = resolution.y();
    
    // Check if dimensions have changed
    if (newWidth != render_width_ || newHeight != render_height_)
    {
        render_width_ = newWidth;
        render_height_ = newHeight;
        
        LOG(INFO) << "Render dimensions updated from camera: " 
                 << render_width_ << "x" << render_height_;
        
        // Resize screen buffers if handlers are initialized
        if (handlers_ && handlers_->screenBuffer && handlers_->screenBuffer->isInitialized())
        {
            handlers_->screenBuffer->resize(render_width_, render_height_);
            LOG(DBUG) << "Screen buffers resized to match camera resolution";
        }
        
        // Resize denoiser if initialized
        if (handlers_ && handlers_->denoiser && handlers_->denoiser->isInitialized())
        {
            handlers_->denoiser->resize(render_width_, render_height_);
            LOG(DBUG) << "Denoiser resized to match camera resolution";
        }
        
        return true; // Dimensions changed
    }
    
    return false; // No change
}