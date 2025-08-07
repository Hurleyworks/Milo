#pragma once

// Handlers struct aggregates all specialized handlers for the rendering pipeline.
// Manages initialization and cleanup of SkyDome, Denoiser, Render, Texture, and Material handlers.
// Provides centralized access to all rendering subsystem handlers.
// Note: Scene handlers are managed locally by each rendering engine to maintain isolation.

#include "../../engine_core.h"
#include "SkyDomeHandler.h"
#include "TextureHandler.h"

// #include "MaterialHandler.h"  // Now engine-specific, not shared
// #include "ModelHandler.h"  // Not used in engine-only mode

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

struct Handlers
{
    // Constructor
    Handlers() = default;
    ~Handlers() = default;

    // Initialize all handlers
    void initialize (RenderContextPtr renderContext, bool skipPipelineInit = false)
    {
        // Initialize SkyDomeHandler
        skyDomeHandler = SkyDomeHandler::create (renderContext);

        // Initialize TextureHandler
        textureHandler = TextureHandler::create (renderContext);
    }

    // Cleanup all handlers
    void cleanup()
    {
        // Cleanup SkyDomeHandler
        if (skyDomeHandler)
        {
            skyDomeHandler->finalize();
            skyDomeHandler.reset();
        }

        // Cleanup TextureHandler
        if (textureHandler)
        {
            // TextureHandler's destructor handles cleanup
            textureHandler.reset();
        }
    }

    // Handler members
    SkyDomeHandlerPtr skyDomeHandler;

    TextureHandlerPtr textureHandler;
};