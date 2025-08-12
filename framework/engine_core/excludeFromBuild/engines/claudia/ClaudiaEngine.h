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
    
    // GBuffer entry points
    enum GBufferEntryPoint
    {
        GBufferEntryPoint_SetupGBuffers = 0,
        GBufferEntryPoint_Count
    };

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
    
    // GBuffer rendering
    void renderGBuffer(CUstream stream);
    
    // Launch parameter management
    void updateLaunchParameters(const mace::InputEvent& input);
    void allocateLaunchParameters();
    
    // Camera update methods
    void updateCameraBody(const mace::InputEvent& input);
    void updateCameraSensor();
    
    // Pipelines
    std::shared_ptr<engine_core::RenderPipeline<engine_core::PathTracingEntryPoint>> pathTracePipeline_;
    std::shared_ptr<engine_core::RenderPipeline<GBufferEntryPoint>> gbufferPipeline_;
    
    // Scene management
    std::shared_ptr<class ClaudiaSceneHandler> sceneHandler_;
    std::shared_ptr<class ClaudiaMaterialHandler> materialHandler_;
    std::shared_ptr<class ClaudiaModelHandler> modelHandler_;
    
    // Render handler
    std::shared_ptr<class ClaudiaRenderHandler> renderHandler_;
    
    // Denoiser handler (Claudia-specific to avoid conflicts with other engines)
    std::shared_ptr<class ClaudiaDenoiserHandler> denoiserHandler_;
    
    // Launch parameters
   // claudia_shared::PipelineLaunchParameters plp_;
   // CUdeviceptr plpOnDevice_ = 0;

     // Pipeline parameter structures
    claudia_shared::StaticPipelineLaunchParameters static_plp_;
    claudia_shared::PerFramePipelineLaunchParameters per_frame_plp_;
    claudia_shared::PipelineLaunchParameters plp_;

    // Device memory for pipeline parameters
    CUdeviceptr static_plp_on_device_ = 0;
    CUdeviceptr per_frame_plp_on_device_ = 0;
    CUdeviceptr plp_on_device_ = 0;

     // G-buffer storage
    struct GBuffers
    {
        cudau::Array gBuffer0[2];
        cudau::Array gBuffer1[2];

        void initialize (CUcontext cuContext, uint32_t width, uint32_t height);
        void resize (uint32_t width, uint32_t height);
        void finalize();
    };

    GBuffers gbuffers_;
    //
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
    
    // Environment light state
    bool environmentDirty_ = true;
    
    // Light probability computation kernels
    ComputeProbTex computeProbTex_;
};