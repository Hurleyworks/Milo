#pragma once

#include "../base/BaseRenderingEngine.h"
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

private:
    // Pipeline setup methods
    void setupPipelines();

    void updateMaterialHitGroups(ClaudiaModelPtr model);  // Set hit groups on a specific model's materials
    
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
    std::shared_ptr<class ClaudiaSceneHandler> sceneHandler_;
    std::shared_ptr<class ClaudiaModelHandler> modelHandler_;
    std::shared_ptr<class ClaudiaAreaLightHandler> areaLightHandler_;
    
     // Pipeline parameter structures
    claudia_shared::StaticPipelineLaunchParameters static_plp_;
    claudia_shared::PerFramePipelineLaunchParameters per_frame_plp_;
    claudia_shared::PipelineLaunchParameters plp_;

    // Device memory for pipeline parameters
    CUdeviceptr static_plp_on_device_ = 0;
    CUdeviceptr per_frame_plp_on_device_ = 0;
    CUdeviceptr plp_on_device_ = 0;

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
    
    // Debug settings
    bool enableGBufferDebug_ = false;
};