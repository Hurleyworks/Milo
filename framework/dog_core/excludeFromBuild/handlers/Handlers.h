#pragma once

// Handlers.h
// Central management structure that creates and coordinates all Dog rendering system handler components

#include "ScreenBufferHandler.h"

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
// Future handlers will be added here:
// #include "MaterialHandler.h"
// #include "ModelHandler.h"
// #include "SceneHandler.h"
// #include "PipelineHandler.h"
// #include "TextureHandler.h"
// #include "SkyDomeHandler.h"
// #include "AreaLightHandler.h"

namespace dog
{

struct Handlers
{
    // Constructor - initializes all handler components with the same render context
    Handlers(RenderContextPtr ctx)
    {
        screenBuffer = ScreenBufferHandler::create(ctx);
        // Future handlers will be initialized here:
        // scene = SceneHandler::create(ctx);
        // material = MaterialHandler::create(ctx);
        // model = ModelHandler::create(ctx);
        // pipeline = PipelineHandler::create(ctx);
        // texture = TextureHandler::create(ctx);
        // skydome = SkyDomeHandler::create(ctx);
        // areaLight = AreaLightHandler::create(ctx);
    }

    // Destructor - explicitly releases all handlers in a specific order
    ~Handlers()
    {
        // Handlers are released in reverse order of potential dependencies
        screenBuffer.reset();
        // Future handlers will be reset here in reverse dependency order:
        // areaLight.reset();
        // scene.reset();
        // material.reset();
        // model.reset();
        // pipeline.reset();
        // texture.reset();
        // skydome.reset();
    }

    ScreenBufferHandlerPtr screenBuffer = nullptr;  // Manages screen rendering buffers
    // Future handlers:
    // SceneHandlerPtr scene = nullptr;            // Manages scene graph and acceleration structures
    // MaterialHandlerPtr material = nullptr;      // Manages material creation and updates
    // ModelHandlerPtr model = nullptr;            // Manages 3D model geometry
    // PipelineHandlerPtr pipeline = nullptr;      // Manages OptiX rendering pipelines
    // TextureHandlerPtr texture = nullptr;        // Manages texture resources
    // SkyDomeHandlerPtr skydome = nullptr;        // Manages environment lighting
    // AreaLightHandlerPtr areaLight = nullptr;    // Manages mesh-based lighting system
};

} // namespace dog