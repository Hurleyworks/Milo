#pragma once

#include "../base/BaseRenderingEngine.h"
#include "../base/RenderPipeline.h"
#include "../base/EngineEntryPoints.h"
#include "shocker_shared.h"

// Forward declarations
using ShockerModelPtr = std::shared_ptr<class ShockerModel>;

class ShockerEngine : public BaseRenderingEngine
{
public:
    // Entry point enums for dual pipeline architecture
    enum class GBufferEntryPoint
    {
        setupGBuffers = 0,
    };

    enum class PathTracingEntryPoint
    {
        pathTrace = 0,
    };

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
    std::string getDescription() const override { return "Dual-pipeline ray tracing engine with G-buffer and path tracing modes"; }
    
    // Pipeline accessors for material handler
    std::shared_ptr<engine_core::RenderPipeline<GBufferEntryPoint>> getGBufferPipeline() const { return gbufferPipeline_; }
    std::shared_ptr<engine_core::RenderPipeline<PathTracingEntryPoint>> getPathTracePipeline() const { return pathTracePipeline_; }
    
    // Render mode control
    enum class RenderMode
    {
        GBufferPreview = 0,
        PathTraceFinal = 1,
        DebugNormals = 2,
        DebugAlbedo = 3,
        DebugDepth = 4,
        DebugMotion = 5
    };
    
    void setRenderMode(RenderMode mode) { renderMode_ = mode; }
    RenderMode getRenderMode() const { return renderMode_; }
    
    // Handle window resize
    void resize(uint32_t width, uint32_t height);
    
    // Light probability computation kernels structure
    struct ComputeProbTex
    {
        CUmodule cudaModule = 0;
        cudau::Kernel computeFirstMip;
        cudau::Kernel computeTriangleProbTexture;
        cudau::Kernel computeGeomInstProbTexture;
        cudau::Kernel computeInstProbTexture;
        cudau::Kernel computeMip;
        cudau::Kernel computeTriangleProbBuffer;
        cudau::Kernel computeAreaLightProbBuffer;
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
    void initializeLightProbabilityKernels();
    void createSBT();
    void updateSBT();  // Update SBT after scene changes
    void linkPipelines();
    void updateMaterialHitGroups(ShockerModelPtr model = nullptr);  // Set hit groups on model's materials (nullptr = all models)
    
    // Launch parameter management
    void updateLaunchParameters(const mace::InputEvent& input);
    void allocateLaunchParameters();
    
    // Camera update methods
    void updateCameraBody(const mace::InputEvent& input);
    void updateCameraSensor();
    
    // Rendering methods
    void renderGBuffer(CUstream stream);
    void renderPathTracing(CUstream stream);
    
    // Dual pipelines
    std::shared_ptr<engine_core::RenderPipeline<GBufferEntryPoint>> gbufferPipeline_;
    std::shared_ptr<engine_core::RenderPipeline<PathTracingEntryPoint>> pathTracePipeline_;
    
    // Default material with hit groups set (following working sample pattern)
    optixu::Material defaultMaterial_;
    
    // Scene management (Shocker-specific handlers)
    std::shared_ptr<class ShockerSceneHandler> sceneHandler_;
    std::shared_ptr<class ShockerMaterialHandler> materialHandler_;
    std::shared_ptr<class ShockerModelHandler> modelHandler_;
    
    // Render handler
    std::shared_ptr<class ShockerRenderHandler> renderHandler_;
    
    // Denoiser handler
    std::shared_ptr<class ShockerDenoiserHandler> denoiserHandler_;
    
    // Area light handler
    std::shared_ptr<class AreaLightHandler> areaLightHandler_;
    
    // Launch parameters (split into static and per-frame)
    shocker_shared::StaticPipelineLaunchParameters staticPlp_;
    shocker_shared::PerFramePipelineLaunchParameters perFramePlp_;
    shocker_shared::PipelineLaunchParameters plp_;
    
    // Device pointers for launch parameters (matching sample code)
    CUdeviceptr staticPlpOnDevice_ = 0;
    CUdeviceptr perFramePlpOnDevice_ = 0;
    CUdeviceptr plpOnDevice_ = 0;
    
    // Camera state
    shocker_shared::PerspectiveCamera lastCamera_;
    shocker_shared::PerspectiveCamera prevCamera_;  // For temporal effects
    mace::InputEvent lastInput_;
    
    // Render state
    RenderMode renderMode_ = RenderMode::PathTraceFinal;
    
    // RNG buffer
    optixu::HostBlockBuffer2D<shared::PCG32RNG, 1> rngBuffer_;
    
    // G-buffers (matching sample code structure)
    cudau::Array gBuffer0_[2];  // Double buffered for temporal effects
    cudau::Array gBuffer1_[2];  // Double buffered for temporal effects
    
    // Pick info buffers for mouse interaction (double buffered)
    cudau::TypedBuffer<shocker_shared::PickInfo> pickInfoBuffers_[2];
    
    // Light probability computation kernels
    ComputeProbTex computeProbTex_;
};