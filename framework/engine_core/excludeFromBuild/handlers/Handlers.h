#pragma once

// Handlers struct aggregates all specialized handlers for the rendering pipeline.
// Manages initialization and cleanup of SkyDome, Denoiser, Render, Texture, and Material handlers.
// Provides centralized access to all rendering subsystem handlers.
// Note: Scene handlers are managed locally by each rendering engine to maintain isolation.

#include "../../engine_core.h"
#include "SkyDomeHandler.h"
#include "DenoiserHandler.h"
//#include "RenderHandler.h"
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

        // Initialize DenoiserHandler (but don't set it up yet - that requires dimensions)
        denoiserHandler = DenoiserHandler::create (renderContext);

        // Initialize RenderHandler (but don't set it up yet - that requires dimensions)
      // renderHandler = RenderHandler::create (renderContext);
       
        
        // Initialize TextureHandler
        textureHandler = TextureHandler::create (renderContext);
        
        // MaterialHandler is now engine-specific, not shared
        // Each engine creates its own material handler
        
        // Initialize ModelHandler (after material handler)
        // modelHandler = ModelHandler::create (renderContext);  // Not used in engine-only mode
    }

    // Cleanup all handlers
    void cleanup()
    {
        // Cleanup RenderHandler
       // if (renderHandler)
       // {
       //     renderHandler->finalize();
       //     renderHandler.reset();
       // }

        // Cleanup DenoiserHandler
        if (denoiserHandler)
        {
            denoiserHandler->finalize();
            denoiserHandler.reset();
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
        
        // MaterialHandler is now engine-specific, cleaned up by each engine
        
        // Cleanup ModelHandler
        // if (modelHandler)
        // {
        //     // ModelHandler's destructor handles cleanup
        //     modelHandler.reset();
        // }  // Not used in engine-only mode
    }

    // Handler members
    SkyDomeHandlerPtr skyDomeHandler;
    DenoiserHandlerPtr denoiserHandler;
   // RenderHandlerPtr renderHandler;
    TextureHandlerPtr textureHandler;
    // MaterialHandlerPtr materialHandler;  // Now engine-specific, not shared
    // ModelHandlerPtr modelHandler;  // Not used in engine-only mode

    // Additional handlers will be added here as we implement them:
    // - MaterialHandler
    // - LightHandler
    // - CameraHandler
    // - etc.
};