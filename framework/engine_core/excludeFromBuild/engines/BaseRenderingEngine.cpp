#include "BaseRenderingEngine.h"
#include "../RenderContext.h"
#include "../tools/PTXManager.h"
#include "RenderPipeline.h"

BaseRenderingEngine::BaseRenderingEngine() :
    renderContext_(nullptr),
    engineName_("BaseEngine"),
    gpuTimerManager_(nullptr),
    timerIndex_(0),
    timingEnabled_(true),
    frameCounter_(0),
    numAccumFrames_(0),
    renderWidth_(0),
    renderHeight_(0),
    plpOnDevice_(0),
    isInitialized_(false),
    buffersInitialized_(false),
    needsRebuild_(false),
    needsDimensionCheck_(false),
    restartRender_(true),
    cameraChanged_(false),
    environmentDirty_(false),
    pipelinePtr_(nullptr),
    scene_(),
    defaultMaterial_(),
    denoiserHandler_(nullptr),
    skyDomeHandler_(nullptr)
{
}

BaseRenderingEngine::~BaseRenderingEngine()
{
    // Cleanup is called by derived class destructor
}

void BaseRenderingEngine::initialize(RenderContext* ctx)
{
    if (isInitialized_)
    {
        LOG(WARNING) << "BaseRenderingEngine already initialized";
        return;
    }
    
    if (!ctx)
    {
        LOG(WARNING) << "BaseRenderingEngine::initialize() called with null context";
        return;
    }
    
    renderContext_ = ctx;
    
    // Get OptiX context and PTXManager from render context
    context_ = renderContext_->getOptiXContextPtr();
    ptxManager_ = renderContext_->getPTXManager();
    
    if (!context_ || !ptxManager_)
    {
        LOG(WARNING) << "BaseRenderingEngine missing required OptiX components";
        return;
    }
    
    // Set engine name based on derived class
    engineName_ = getName();
    
    // Initialize render dimensions from camera
    initializeRenderDimensions();
    
    // Create OptiX scene
    scene_ = context_->createScene();
    LOG(INFO) << engineName_ << " OptiX scene created";
    
    // Reset frame counters and state
    frameCounter_ = 0;
    numAccumFrames_ = 0;
    timerIndex_ = 0;
    buffersInitialized_ = false;
    needsRebuild_ = false;
    needsDimensionCheck_ = true;
    
    // Initialize StreamChain for better GPU/CPU overlap
    streamChain_ = std::make_unique<StreamChain<NUM_STREAM_BUFFERS>>();
    streamChain_->initialize(renderContext_->getCudaContext());
    LOG(INFO) << "Initialized StreamChain with " << NUM_STREAM_BUFFERS << " streams";
    
    isInitialized_ = true;
}

void BaseRenderingEngine::initializeRenderDimensions()
{
    // Get initial render dimensions from camera
    if (renderContext_ && renderContext_->getCamera())
    {
        renderWidth_ = renderContext_->getCamera()->getChangedSensorPixelRes().x();
        renderHeight_ = renderContext_->getCamera()->getChangedSensorPixelRes().y();
        LOG(INFO) << engineName_ << " render dimensions from camera: " << renderWidth_ << "x" << renderHeight_;
    }
    else
    {
        // Use default dimensions if no camera available
        renderWidth_ = DEFAULT_RENDER_WIDTH;
        renderHeight_ = DEFAULT_RENDER_HEIGHT;
        LOG(WARNING) << engineName_ << " no camera available, using default dimensions: " << renderWidth_ << "x" << renderHeight_;
    }
}

void BaseRenderingEngine::cleanup()
{
    if (!isInitialized_)
    {
        return;
    }
    
    // Clean up StreamChain
    if (streamChain_)
    {
        streamChain_->waitAllWorkDone();
        streamChain_->finalize();
        streamChain_.reset();
        LOG(INFO) << "StreamChain finalized";
    }
    
    // Reset state
    renderContext_ = nullptr;
    gpuTimerManager_ = nullptr;
    frameCounter_ = 0;
    numAccumFrames_ = 0;
    renderWidth_ = 0;
    renderHeight_ = 0;
    isInitialized_ = false;
}

void BaseRenderingEngine::setGPUTimerManager(GPUTimerManager* timerManager)
{
    gpuTimerManager_ = timerManager;
    if (!gpuTimerManager_)
    {
        LOG(WARNING) << "GPU timer manager not available for " << engineName_;
        timingEnabled_ = false;
    }
}

GPUTimerManager::GPUTimer* BaseRenderingEngine::getCurrentTimer()
{
    if (!isTimingEnabled())
    {
        return nullptr;
    }
    
    return &gpuTimerManager_->getGPUTimer(timerIndex_);
}

void BaseRenderingEngine::switchTimerBuffer()
{
    if (isTimingEnabled())
    {
        timerIndex_ = (timerIndex_ + 1) % GPUTimerManager::getTimerBufferCount();
    }
}

void BaseRenderingEngine::reportTimings(uint32_t frameCount, uint32_t reportInterval)
{
    if (!isTimingEnabled() || frameCount % reportInterval != 0)
    {
        return;
    }
    
    try
    {
        auto& timer = gpuTimerManager_->getGPUTimer(timerIndex_);
        
        // Get timing results
        float frameTime = timer.frame.report();
        float pathTraceTime = timer.pathTrace.report();
        float denoiseTime = timer.denoise.report();
        
        // Calculate FPS
        float fps = frameTime > 0.0f ? 1000.0f / frameTime : 0.0f;
        
        // Log timing report
        LOG(INFO) << engineName_ << " GPU Timing Report - "
                  << "Total: " << frameTime << "ms (" << fps << " FPS), "
                  << "Path trace: " << pathTraceTime << "ms, "
                  << "Denoise: " << denoiseTime << "ms";
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to report GPU timings: " << e.what();
    }
}

void BaseRenderingEngine::addGeometry(sabi::RenderableNode node)
{
    // Default implementation - log warning
    LOG(WARNING) << "Engine " << engineName_ << " does not implement addGeometry()";
}

void BaseRenderingEngine::clearScene()
{
    // Default implementation - reset accumulation counter
    // Derived classes should override to clear their scene handlers
    numAccumFrames_ = 0;
    LOG(WARNING) << "Engine " << engineName_ << " should override clearScene() to clear scene handlers";
}

std::string BaseRenderingEngine::loadPTXData(const char* ptxFileName, bool useEmbedded)
{
    if (!ptxManager_)
    {
        LOG(WARNING) << "PTXManager not available";
        return std::string();
    }
    
    // Get PTX from manager
    std::vector<char> ptxData = ptxManager_->getPTXData(ptxFileName, useEmbedded);
    if (ptxData.empty())
    {
        LOG(WARNING) << "Failed to load PTX: " << ptxFileName;
        return std::string();
    }
    
    LOG(INFO) << "Loaded PTX for " << engineName_ << ": " << ptxFileName 
              << " (size: " << ptxData.size() << " bytes)";
    
    // Convert to string
    return std::string(ptxData.begin(), ptxData.end());
}

void BaseRenderingEngine::configurePipelineDefaults(optixu::Pipeline& pipeline, 
                                                   uint32_t numPayloadDwords, 
                                                   size_t launchParamsSize)
{
    pipeline.setPipelineOptions(
        numPayloadDwords,
        optixu::calcSumDwords<float2>(),  // Standard attribute size for barycentrics
        "plp",                            // Pipeline launch parameters name
        launchParamsSize,
        static_cast<OptixTraversableGraphFlags>(getDefaultPipelineFlags()),
        static_cast<OptixExceptionFlags>(getDefaultExceptionFlags()),
        static_cast<OptixPrimitiveTypeFlags>(getDefaultPrimitiveFlags())
    );
}


