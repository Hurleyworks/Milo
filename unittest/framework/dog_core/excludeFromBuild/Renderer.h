#pragma once

#include "RenderContext.h"

// Stub Renderer class for dog_core
// Provides basic rendering infrastructure and lifetime management
// This is a simplified version for testing RenderContext integration
class Renderer 
{
public:
    Renderer();
    ~Renderer();
    
    // Delete copy operations
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    
    // Initialize the renderer with optional device selection
    bool initialize(int deviceIndex = 0);
    
    // Clean up all resources
    void shutdown();
    
    // Check if renderer is initialized
    bool isInitialized() const { return initialized_ && render_context_ != nullptr; }
    
    // Get the render context for testing
    RenderContextPtr getRenderContext() const { return render_context_; }
    
private:
    bool initialized_ = false;
    RenderContextPtr render_context_ = nullptr;
};