#pragma once

// BaseRenderingEngine.h
// Base class for rendering engines that provides common functionality
// including GPU timer support and pipeline management.

#include "IRenderingEngine.h"
#include "../tools/GPUTimerManager.h"
#include "RenderPipeline.h"
#include "EngineEntryPoints.h"
#include "../common/common_host.h"


// Forward declarations
class RenderContext;
class PTXManager;
class DenoiserHandler;
class SkyDomeHandler;
namespace optixu {
    class Context;
    class Scene;
    class Material;
    class Pipeline;
    class Module;
}

class BaseRenderingEngine : public IRenderingEngine
{
public:
    BaseRenderingEngine();
    virtual ~BaseRenderingEngine();
    
    // IRenderingEngine interface
    void initialize(RenderContext* ctx) override;
    void cleanup() override;
    
    // Default scene management (engines can override)
    void addGeometry(sabi::RenderableNode node) override;
    void clearScene() override;
    void onEnvironmentChanged() override {} // Default: do nothing
    // render() remains pure virtual - must be implemented by derived classes
    
    // Set the GPU timer manager (called by RenderEngineManager)
    void setGPUTimerManager(GPUTimerManager* timerManager);
    
protected:
    // GPU timing helpers
    bool isTimingEnabled() const { return gpuTimerManager_ != nullptr && timingEnabled_; }
    GPUTimerManager::GPUTimer* getCurrentTimer();
    void switchTimerBuffer();
    
    // Timer reporting (call every N frames)
    void reportTimings(uint32_t frameCount, uint32_t reportInterval = 100);
    
protected:
    // Common initialization helpers
    void initializeRenderDimensions();
    
    // Helper to initialize handlers with current dimensions
    template<typename HandlerPtr>
    bool initializeHandlerWithDimensions(HandlerPtr& handler, const char* handlerName)
    {
        if (handler && renderWidth_ > 0 && renderHeight_ > 0)
        {
            if (handler->initialize(renderWidth_, renderHeight_))
            {
                LOG(INFO) << engineName_ << " " << handlerName << " initialized with " 
                          << renderWidth_ << "x" << renderHeight_;
                return true;
            }
            else
            {
                LOG(WARNING) << engineName_ << " failed to initialize " << handlerName;
                return false;
            }
        }
        return false;
    }
    
    // Default render dimensions when no camera is available
    static constexpr uint32_t DEFAULT_RENDER_WIDTH = 1920;
    static constexpr uint32_t DEFAULT_RENDER_HEIGHT = 1080;
    
protected:
    // Helper method to create pipeline with engine-specific entry point type
    template<typename EntryPointType>
    engine_core::RenderPipeline<EntryPointType>* createPipeline() {
        auto pipeline = new engine_core::RenderPipeline<EntryPointType>();
        pipelinePtr_ = static_cast<void*>(pipeline);
        return pipeline;
    }
    
    // Helper method to get typed pipeline
    template<typename EntryPointType>
    engine_core::RenderPipeline<EntryPointType>* getPipeline() {
        return static_cast<engine_core::RenderPipeline<EntryPointType>*>(pipelinePtr_);
    }
    
    // Cleanup pipeline (derived classes must call with correct type)
    template<typename EntryPointType>
    void destroyPipeline() {
        if (pipelinePtr_) {
            auto pipeline = static_cast<engine_core::RenderPipeline<EntryPointType>*>(pipelinePtr_);
            pipeline->destroy();
            delete pipeline;
            pipelinePtr_ = nullptr;
        }
    }
    
    // Common helper methods for pipeline setup
    std::string loadPTXData(const char* ptxFileName, bool useEmbedded = true);
    void configurePipelineDefaults(optixu::Pipeline& pipeline, uint32_t numPayloadDwords, size_t launchParamsSize);
    
    // Helper to get common pipeline flags
    static constexpr uint32_t getDefaultPipelineFlags() {
        return OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    }
    
    static constexpr uint32_t getDefaultExceptionFlags() {
        return OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    }
    
    static constexpr uint32_t getDefaultPrimitiveFlags() {
        return OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    }

protected:
    // Render context (available to all derived engines)
    RenderContext* renderContext_;
    
    // OptiX context and managers
    optixu::Context* context_;
    PTXManager* ptxManager_;
    
    // Engine name for identification
    std::string engineName_;
    
    // GPU timer management
    GPUTimerManager* gpuTimerManager_;
    uint32_t timerIndex_;
    bool timingEnabled_;
    
    // Frame counter for timing reports
    uint32_t frameCounter_;
    
    // Accumulation frame counter for progressive rendering
    uint32_t numAccumFrames_;
    
    // Render dimensions
    uint32_t renderWidth_;
    uint32_t renderHeight_;
    
    // Device memory pointer for launch parameters (actual struct is engine-specific)
    CUdeviceptr plpOnDevice_;
    
    // State flags
    bool isInitialized_;
    bool buffersInitialized_;
    bool needsRebuild_;
    bool needsDimensionCheck_;  // Set when window might have resized
    
    // Common render state flags
    bool restartRender_;        // True when accumulation needs to restart
    bool cameraChanged_;        // True when camera has moved
    bool environmentDirty_;     // True when environment light changed
    
    // Type-erased pipeline pointer (managed by derived class)
    // We use void* because OptiX resources aren't copyable/moveable
    void* pipelinePtr_;
    
    // Stream management for better GPU/CPU overlap
    static constexpr uint32_t NUM_STREAM_BUFFERS = 2;  // Double buffering
    std::unique_ptr<StreamChain<NUM_STREAM_BUFFERS>> streamChain_;
    
    // Common scene components
    optixu::Scene scene_;
    optixu::Material defaultMaterial_;
    
    // Common handlers
    std::shared_ptr<DenoiserHandler> denoiserHandler_;
    std::shared_ptr<SkyDomeHandler> skyDomeHandler_;
};