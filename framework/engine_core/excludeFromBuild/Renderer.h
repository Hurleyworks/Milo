#pragma once

// Renderer is the main rendering engine for milo_core.
// Manages pipeline selection, scene rendering, and resource coordination.
// Supports runtime pipeline switching and dynamic scene updates.

#include "../engine_core.h"
#include "RenderContext.h"
#include "tools/GPUTimerManager.h"
#include "engines/RenderEngineManager.h"

using sabi::CameraHandle;

class PTXManager;     // forward declaration
class SkyDomeHandler; // forward declaration

class Renderer
{
 public:
    // Constructor initializes the renderer with default settings
    Renderer();

    // Destructor ensures proper cleanup of rendering resources
    ~Renderer();

    void init (MessageService messengers, const PropertyService& properties);
    void initializeEngine (CameraHandle camera, ImageCacheHandlerPtr imageCache);
    void finalize();

    void render (const InputEvent& input, bool updateMotion, uint32_t frameNumber = 0);

    void addSkyDomeHDR (const std::filesystem::path& hdrPath);
    void addRenderableNode (RenderableWeakRef& weakNode);

  
    // Engine management (new system)
    bool setEngine(const std::string& engineName);
    std::string getCurrentEngineName() const;
    std::vector<std::string> getAvailableEngines() const;
    
    // Resource access
    CameraHandle getCamera() const { return renderContext_ ? renderContext_->getCamera() : nullptr; }

 private:
    MessageService messengers;
    PropertyService properties;

    // Render Context
    RenderContextPtr renderContext_;

    // PTX Manager
    std::unique_ptr<PTXManager> ptxManager_;

    // GPU Timer Manager
    std::unique_ptr<GPUTimerManager> gpuTimerManager_;
    
    // Rendering Engine Manager (new system)
    std::unique_ptr<RenderEngineManager> engineManager_;

    // Initialization state
    bool initialized_ = false;

    // Store renderable nodes for pipeline switching
    std::vector<RenderableWeakRef> renderableNodes_;

    // Store sky dome HDR path for pipeline switching
    std::filesystem::path currentSkyDomeHDR_;
};