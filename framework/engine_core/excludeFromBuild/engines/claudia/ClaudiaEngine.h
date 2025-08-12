#pragma once

#include "../base/BaseRenderingEngine.h"
#include "../base/RenderPipeline.h"
#include "../base/EngineEntryPoints.h"
#include "claudia_shared.h"

// Forward declarations
using ClaudiaModelPtr = std::shared_ptr<class ClaudiaModel>;

class ClaudiaEngine : public BaseRenderingEngine
{
public:
    // Entry point enums for dual pipeline architecture (matching RiPR)
    enum class GBufferEntryPoint
    {
        setupGBuffers = 0,
        NumEntryPoints
    };
    
    ClaudiaEngine();
    ~ClaudiaEngine() override;

    // IRenderingEngine interface
    void initialize(RenderContext* ctx) override;
    void cleanup() override;
    void addGeometry(sabi::RenderableNode node) override;
    void clearScene() override;
    void render(const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber) override;
    void onEnvironmentChanged() override;
    std::string getName() const override { return "Claudia Engine"; }
    std::string getDescription() const override { return "Claudia Path Tracing with adaptive sampling and improved convergence"; }
    
    // Pipeline accessors for material handler
    std::shared_ptr<engine_core::RenderPipeline<GBufferEntryPoint>> getGBufferPipeline() const { return gbufferPipeline_; }
    std::shared_ptr<engine_core::RenderPipeline<engine_core::PathTracingEntryPoint>> getPathTracePipeline() const { return pathTracePipeline_; }
    
    // Light probability computation kernels structure
    struct ComputeProbTex {
        CUmodule cudaModule = nullptr;
        cudau::Kernel computeFirstMip;
        cudau::Kernel computeTriangleProbTexture;
        cudau::Kernel computeGeomInstProbTexture;
        cudau::Kernel computeInstProbTexture;
        cudau::Kernel computeMip;
        cudau::Kernel computeTriangleProbBuffer;
        cudau::Kernel computeGeomInstProbBuffer;
        cudau::Kernel computeInstProbBuffer;
        cudau::Kernel finalizeDiscreteDistribution1D;
        cudau::Kernel test;
    };
    
    // Accessor for light probability computation kernels
    const ComputeProbTex& getComputeProbTex() const { return computeProbTex_; }

private:
    // Pipeline setup methods
    void setupPipelines();
    void createModules();
    void createPrograms();
    void createSBT();
    void updateSBT();  // Update SBT after scene changes
    void linkPipelines();
    void updateMaterialHitGroups(ClaudiaModelPtr model);  // Set hit groups on a specific model's materials
    void initializeLightProbabilityKernels();  // Initialize CUDA kernels for light probability computation
    
    // Launch parameter management
    void updateLaunchParameters(const mace::InputEvent& input);
    void allocateLaunchParameters();
    
    // Split parameter updates
    void updateStaticParameters();  // Update static params (buffers, textures)
    void updatePerFrameParameters(const mace::InputEvent& input);  // Update per-frame params (camera, etc)
    
    // Camera update methods
    void updateCameraBody(const mace::InputEvent& input);
    void updateCameraSensor();
    
    // Rendering methods for dual pipelines
    void renderGBuffer(CUstream stream);
    void renderPathTracing(CUstream stream);
    
    // Dual pipelines (matching RiPR)
    std::shared_ptr<engine_core::RenderPipeline<GBufferEntryPoint>> gbufferPipeline_;
    std::shared_ptr<engine_core::RenderPipeline<engine_core::PathTracingEntryPoint>> pathTracePipeline_;
    
    // Default material for GBuffer pipeline
    optixu::Material defaultMaterial_;
    
    // Scene management
    std::shared_ptr<class ClaudiaSceneHandler> sceneHandler_;
    std::shared_ptr<class ClaudiaMaterialHandler> materialHandler_;
    std::shared_ptr<class ClaudiaModelHandler> modelHandler_;
    
    // Render handler
    std::shared_ptr<class ClaudiaRenderHandler> renderHandler_;
    
    // Denoiser handler (Claudia-specific to avoid conflicts with other engines)
    std::shared_ptr<class ClaudiaDenoiserHandler> denoiserHandler_;
    
    // Launch parameters (OLD - for backward compatibility)
    // COMMENTED OUT to ensure we're not using the flat structure anymore
    // claudia_shared::PipelineLaunchParameters plp_;
    // CUdeviceptr plpOnDevice_ = 0;
    
    // NEW: Split launch parameters (like RiPR)
    claudia_shared::StaticPipelineLaunchParameters staticParams_;
    claudia_shared::PerFramePipelineLaunchParameters perFrameParams_;
    claudia_shared::PipelineLaunchParametersSplit plpSplit_;
    CUdeviceptr staticParamsOnDevice_ = 0;
    CUdeviceptr perFrameParamsOnDevice_ = 0;
    CUdeviceptr plpSplitOnDevice_ = 0;
    
    // Camera state
    claudia_shared::PerspectiveCamera lastCamera_;
    claudia_shared::PerspectiveCamera prevCamera_;  // For temporal effects
    mace::InputEvent lastInput_;
    bool cameraChanged_ = false;
    
    // Render state
    bool restartRender_ = true;
    uint32_t frameCounter_ = 0;
    
    // RNG buffer
    optixu::HostBlockBuffer2D<shared::PCG32RNG, 1> rngBuffer_;
    
    // GBuffer storage (using cudau::Array like in RiPR and sample code patterns)
    cudau::Array gbuffer0_[2];
    cudau::Array gbuffer1_[2];
    
    // GBuffer state
    bool gbufferEnabled_ = false;
    
    // Environment light state
    bool environmentDirty_ = true;
    
    // Light probability computation kernels
    ComputeProbTex computeProbTex_;
};