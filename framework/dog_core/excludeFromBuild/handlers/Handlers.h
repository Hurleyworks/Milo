#pragma once

// Handlers.h
// Central management structure that creates and coordinates all Dog rendering system handler components

#include "ScreenBufferHandler.h"
#include "PipelineHandler.h"
#include "PipelineParameterHandler.h"
#include "DenoiserHandler.h"
#include "SceneHandler.h"

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
// Future handlers will be added here:
// #include "MaterialHandler.h"
// #include "ModelHandler.h"
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
        pipeline = PipelineHandler::create(ctx);
        pipelineParameter = PipelineParameterHandler::create(ctx);
        denoiser = DenoiserHandler::create(ctx);
        scene = SceneHandler::create(ctx);
        // Future handlers will be initialized here:
        // material = MaterialHandler::create(ctx);
        // model = ModelHandler::create(ctx);
        // texture = TextureHandler::create(ctx);
        // skydome = SkyDomeHandler::create(ctx);
        // areaLight = AreaLightHandler::create(ctx);
    }

    // Destructor - explicitly releases all handlers in a specific order
    ~Handlers()
    {
        // Handlers are released in reverse order of potential dependencies
        denoiser.reset();
        screenBuffer.reset();
        pipelineParameter.reset();
        pipeline.reset();
        scene.reset();
        // Future handlers will be reset here in reverse dependency order:
        // areaLight.reset();
        // material.reset();
        // model.reset();
        // texture.reset();
        // skydome.reset();
    }

    ScreenBufferHandlerPtr screenBuffer = nullptr;       // Manages screen rendering buffers
    PipelineHandlerPtr pipeline = nullptr;               // Manages OptiX rendering pipelines
    PipelineParameterHandlerPtr pipelineParameter = nullptr; // Manages pipeline launch parameters
    DenoiserHandlerPtr denoiser = nullptr;               // Manages OptiX AI denoising
    SceneHandlerPtr scene = nullptr;                     // Manages scene graph and acceleration structures
    // Future handlers:
    // MaterialHandlerPtr material = nullptr;      // Manages material creation and updates
    // ModelHandlerPtr model = nullptr;            // Manages 3D model geometry
    // TextureHandlerPtr texture = nullptr;        // Manages texture resources
    // SkyDomeHandlerPtr skydome = nullptr;        // Manages environment lighting
    // AreaLightHandlerPtr areaLight = nullptr;    // Manages mesh-based lighting system
};

} // namespace dog