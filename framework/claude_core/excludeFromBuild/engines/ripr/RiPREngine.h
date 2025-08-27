#pragma once

#include "../base/BaseRenderingEngine.h"
#include "ripr_shared.h"

// Forward declarations
using RiPRModelPtr = std::shared_ptr<class RiPRModel>;

class RiPREngine : public BaseRenderingEngine
{
public:
    RiPREngine();
    ~RiPREngine() override;

    // IRenderingEngine interface
    void initialize(RenderContext* ctx) override;
    void cleanup() override;
    void addGeometry(sabi::RenderableNode node) override;
    void clearScene() override;
    void render(const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber) override;
    void onEnvironmentChanged() override;
    std::string getName() const override { return "RiPR Engine"; }
    std::string getDescription() const override { return "RiPR Path Tracing with adaptive sampling and improved convergence"; }
    
    // Build light distributions for emissive geometry
    // Call this after adding emissive geometry to the scene
    void buildLightDistributions();
    
    
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
    const ComputeProbTex& getLightProbKernels() const { return computeProbTex_; }
    
    // Check if light probability kernels are initialized
    bool hasLightProbKernels() const { return computeProbTex_.cudaModule != nullptr; }

private:
    // Pipeline setup methods
    void setupPipelines();

    void updateMaterialHitGroups(RiPRModelPtr model);  // Set hit groups on a specific model's materials
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
    std::shared_ptr<class RiPRSceneHandler> sceneHandler_;
    std::shared_ptr<class RiPRModelHandler> modelHandler_;
    
     // Pipeline parameter structures
    ripr_shared::StaticPipelineLaunchParameters static_plp_;
    ripr_shared::PerFramePipelineLaunchParameters per_frame_plp_;
    ripr_shared::PipelineLaunchParameters plp_;

    // Device memory for pipeline parameters
    CUdeviceptr static_plp_on_device_ = 0;
    CUdeviceptr per_frame_plp_on_device_ = 0;
    CUdeviceptr plp_on_device_ = 0;

    //
    // Camera state
    ripr_shared::PerspectiveCamera lastCamera_;
    ripr_shared::PerspectiveCamera prevCamera_;  // For temporal effects
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