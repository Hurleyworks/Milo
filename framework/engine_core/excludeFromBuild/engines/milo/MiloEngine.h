#pragma once

#include "../base/BaseRenderingEngine.h"
#include "../base/RenderPipeline.h"
#include "../base/EngineEntryPoints.h"
#include "milo_shared.h"

// Forward declarations
using MiloModelPtr = std::shared_ptr<class MiloModel>;

class MiloEngine : public BaseRenderingEngine
{
public:
    MiloEngine();
    ~MiloEngine() override;

    // IRenderingEngine interface
    void initialize(RenderContext* ctx) override;
    void cleanup() override;
    void addGeometry(sabi::RenderableNode node) override;
    void clearScene() override;
    void render(const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber) override;
    void onEnvironmentChanged() override;
    std::string getName() const override { return "Milo Engine"; }
    std::string getDescription() const override { return "Milo Path Tracing with adaptive sampling and improved convergence"; }
    
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

private:
    // Pipeline setup methods
    void setupPipelines();
    void createModules();
    void createPrograms();
    void createSBT();
    void updateSBT();  // Update SBT after scene changes
    void linkPipelines();
    void updateMaterialHitGroups(MiloModelPtr model);  // Set hit groups on a specific model's materials
    void initializeLightProbabilityKernels();  // Initialize CUDA kernels for light probability computation
    
    // Launch parameter management
    void updateLaunchParameters(const mace::InputEvent& input);
    void allocateLaunchParameters();
    
    // Camera update methods
    void updateCameraBody(const mace::InputEvent& input);
    void updateCameraSensor();
    
    // Pipeline
    std::shared_ptr<engine_core::RenderPipeline<engine_core::PathTracingEntryPoint>> pathTracePipeline_;
    
    // Scene management
    std::shared_ptr<class MiloSceneHandler> sceneHandler_;
    std::shared_ptr<class MiloMaterialHandler> materialHandler_;
    std::shared_ptr<class MiloModelHandler> modelHandler_;
    
    // Render handler
    std::shared_ptr<class MiloRenderHandler> renderHandler_;
    
    // Denoiser handler (Milo-specific to avoid conflicts with other engines)
    std::shared_ptr<class MiloDenoiserHandler> denoiserHandler_;
    
    // Launch parameters
    milo_shared::PipelineLaunchParameters plp_;
    CUdeviceptr plpOnDevice_ = 0;
    
    // Camera state
    milo_shared::PerspectiveCamera lastCamera_;
    milo_shared::PerspectiveCamera prevCamera_;  // For temporal effects
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