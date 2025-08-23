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
#include "ScreenBufferHandler.h"
#include "MaterialHandler.h"
#include "SceneHandler.h"

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
        // Initialize ScreenBufferHandler (fundamental buffer management)
        screenBufferHandler = ScreenBufferHandler::create (renderContext);

        // Create SceneHandler (manages IAS) - uses lazy initialization
        // Will only initialize when first instance is added
        sceneHandler = SceneHandler::create (renderContext);

        // Initialize SkyDomeHandler
        skyDomeHandler = SkyDomeHandler::create (renderContext);

        // Initialize TextureHandler
        textureHandler = TextureHandler::create (renderContext);

        // Initialize MaterialHandler
        materialHandler = MaterialHandler::create (renderContext);
        if (materialHandler)
            materialHandler->initialize();

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

        // Cleanup MaterialHandler
        if (materialHandler)
        {
            materialHandler->finalize();
            materialHandler.reset();
        }

        // Cleanup SceneHandler
        if (sceneHandler)
        {
            sceneHandler->finalize();
            sceneHandler.reset();
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

        // Cleanup ScreenBufferHandler
        if (screenBufferHandler)
        {
            screenBufferHandler->finalize();
            screenBufferHandler.reset();
        }
    }

    // Handler members
    ScreenBufferHandlerPtr screenBufferHandler;
    SceneHandlerPtr sceneHandler;
    SkyDomeHandlerPtr skyDomeHandler;
    TextureHandlerPtr textureHandler;
    MaterialHandlerPtr materialHandler;
    PipelineHandlerPtr pipelineHandler;
    DenoiserHandlerPtr denoiserHandler;
};