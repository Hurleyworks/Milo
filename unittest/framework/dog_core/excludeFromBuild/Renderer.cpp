#include "Renderer.h"
#include <g3log/g3log.hpp>

Renderer::Renderer()
{
    LOG(DBUG) << "Renderer constructor";
}

Renderer::~Renderer()
{
    LOG(DBUG) << "Renderer destructor";
    shutdown();
}

bool Renderer::initialize(int deviceIndex)
{
    if (initialized_)
    {
        LOG(WARNING) << "Renderer already initialized";
        return true;
    }
    
    LOG(INFO) << "Initializing Renderer with device " << deviceIndex;
    
    // Create and initialize render context
    render_context_ = RenderContext::create();
    if (!render_context_)
    {
        LOG(WARNING) << "Failed to create render context";
        return false;
    }
    
    if (!render_context_->initialize(deviceIndex))
    {
        LOG(WARNING) << "Failed to initialize render context";
        render_context_.reset();
        return false;
    }
    
    initialized_ = true;
    LOG(INFO) << "Renderer initialized successfully";
    return true;
}

void Renderer::shutdown()
{
    if (!initialized_)
    {
        return;
    }
    
    LOG(INFO) << "Shutting down Renderer";
    
    if (render_context_)
    {
        render_context_->cleanup();
        render_context_.reset();
    }
    
    initialized_ = false;
    LOG(INFO) << "Renderer shutdown complete";
}