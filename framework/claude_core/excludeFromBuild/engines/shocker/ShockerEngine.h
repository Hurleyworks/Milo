#pragma once

#include "../base/BaseRenderingEngine.h"
#include "../base/EngineEntryPoints.h"
#include "shocker_shared.h"

// Forward declarations
using ShockerModelPtr = std::shared_ptr<class ShockerModel>;

class ShockerEngine : public BaseRenderingEngine
{
public:
    ShockerEngine();
    ~ShockerEngine() override;

    // IRenderingEngine interface
    void initialize(RenderContext* ctx) override;
    void cleanup() override;
    void addGeometry(sabi::RenderableNode node) override;
    void clearScene() override;
    void render(const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber) override;
    void onEnvironmentChanged() override;
    std::string getName() const override { return "Shocker Engine"; }
    std::string getDescription() const override { return "Shocker Path Tracing with adaptive sampling and improved convergence"; }
    
    
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

    void updateMaterialHitGroups(ShockerModelPtr model);  // Set hit groups on a specific model's materials
    void initializeLightProbabilityKernels();  // Initialize CUDA kernels for light probability computation
    
    // GBuffer rendering
    void outputGBufferDebugInfo(CUstream stream);
    void setGBufferDebugEnabled(bool enabled) { enableGBufferDebug_ = enabled; }
    bool isGBufferDebugEnabled() const { return enableGBufferDebug_; }
    
    // Launch parameter management
    void updateLaunchParameters(const mace::InputEvent& input);
    void allocateLaunchParameters();
    
    // Camera update methods
    void updateCameraBody(const mace::InputEvent& input);
    void updateCameraSensor();
    
    // Scene management
    std::shared_ptr<class ShockerSceneHandler> sceneHandler_;
    std::shared_ptr<class ShockerModelHandler> modelHandler_;
    
     // Pipeline parameter structures
    shocker_shared::StaticPipelineLaunchParameters static_plp_;
    shocker_shared::PerFramePipelineLaunchParameters per_frame_plp_;
    shocker_shared::PipelineLaunchParameters plp_;

    // Device memory for pipeline parameters
    CUdeviceptr static_plp_on_device_ = 0;
    CUdeviceptr per_frame_plp_on_device_ = 0;
    CUdeviceptr plp_on_device_ = 0;

    //
    // Camera state
    shocker_shared::PerspectiveCamera lastCamera_;
    shocker_shared::PerspectiveCamera prevCamera_;  // For temporal effects
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
    
    // Debug settings
    bool enableGBufferDebug_ = false;
};