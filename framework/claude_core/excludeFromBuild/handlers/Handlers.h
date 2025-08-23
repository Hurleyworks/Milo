#pragma once

// Handlers struct aggregates all specialized handlers for the rendering pipeline.
// Manages initialization and cleanup of SkyDome, Denoiser, Render, Texture, Pipeline, and Material handlers.
// Provides centralized access to all rendering subsystem handlers.
// Note: Scene handlers are managed locally by each rendering engine to maintain isolation.

#include "../../claude_core.h"
#include "SkyDomeHandler.h"
#include "TextureHandler.h"
#include "PipelineHandler.h"
#include "DenoiserHandler.h"

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
    void initialize (RenderContextPtr renderContext)
    {
        // Initialize SkyDomeHandler
        skyDomeHandler = SkyDomeHandler::create (renderContext);

        // Initialize TextureHandler
        textureHandler = TextureHandler::create (renderContext);

        // Initialize PipelineHandler
        pipelineHandler = PipelineHandler::create (renderContext);

        // Initialize DenoiserHandler
        denoiserHandler = DenoiserHandler::create (renderContext);
    }

    // Cleanup all handlers
    void cleanup()
    {
        // Cleanup DenoiserHandler
        if (denoiserHandler)
        {
            denoiserHandler->finalize();
            denoiserHandler.reset();
        }

        // Cleanup PipelineHandler
        if (pipelineHandler)
        {
            pipelineHandler->destroyAll();
            pipelineHandler.reset();
        }

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
    PipelineHandlerPtr pipelineHandler;
    DenoiserHandlerPtr denoiserHandler;
};