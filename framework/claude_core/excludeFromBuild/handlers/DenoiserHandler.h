#pragma once

// DenoiserHandler manages OptiX AI-accelerated denoising functionality for the Claude rendering engine.
// Supports temporal denoising with motion vectors, HDR tone mapping, and kernel prediction modes.
// Based on OptiX 9.0 denoiser API with support for multiple denoising models.
//
// Key features:
// - Temporal denoising with previous frame and flow vectors
// - HDR denoising with automatic exposure normalization
// - Kernel prediction mode for AOV-based denoising
// - Upscaling support (2x) for lower resolution rendering
// - Tiled denoising for large framebuffers
// - Multiple guide layer support (albedo, normal, flow)
//
// Denoising pipeline:
// 1. Render noisy beauty, albedo, normal, and flow buffers
// 2. Compute HDR normalizer from noisy input
// 3. Execute denoising with temporal or spatial model
// 4. Output denoised beauty and optional AOV layers

#include "../RenderContext.h"
#include <vector>
#include <memory>

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
class DenoiserHandler;
using DenoiserHandlerPtr = std::shared_ptr<DenoiserHandler>;

// Denoiser configuration
struct DenoiserConfig
{
    // Denoiser model selection
    enum class Model
    {
        HDR,                    // Basic HDR denoiser (spatial only)
        Temporal,              // Temporal denoiser with motion vectors
        TemporalAOV,           // Temporal with kernel prediction
        AOV,                   // Spatial with kernel prediction
        Upscale2X,             // Spatial with 2x upscaling
        TemporalUpscale2X      // Temporal with 2x upscaling
    };

    Model model = Model::Temporal;
    
    // Guide layers
    bool useAlbedo = true;
    bool useNormal = true;
    bool useFlow = true;  // For temporal models
    
    // Alpha handling
    OptixDenoiserAlphaMode alphaMode = OPTIX_DENOISER_ALPHA_MODE_COPY;
    
    // Tiling for large framebuffers
    bool useTiling = false;
    uint32_t tileWidth = 256;
    uint32_t tileHeight = 256;
    
    // Upscaling (only for Upscale2X models)
    bool performUpscale = false;
    
    // Kernel prediction mode
    bool useKernelPrediction = false;
};

// Input buffers for denoising
struct DenoiserInputBuffers
{
    // Required buffers
    cudau::TypedBuffer<float4>* noisyBeauty = nullptr;
    
    // Optional guide layers
    cudau::TypedBuffer<float4>* albedo = nullptr;
    cudau::TypedBuffer<float4>* normal = nullptr;
    cudau::TypedBuffer<float2>* flow = nullptr;  // Motion vectors for temporal
    
    // Temporal buffers (for temporal models)
    cudau::TypedBuffer<float4>* previousDenoisedBeauty = nullptr;
    optixu::BufferView previousInternalGuideLayer;  // For kernel prediction
    
    // Pixel formats
    OptixPixelFormat beautyFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    OptixPixelFormat albedoFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    OptixPixelFormat normalFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    OptixPixelFormat flowFormat = OPTIX_PIXEL_FORMAT_FLOAT2;
    
    // Validate buffers based on configuration
    bool validate(const DenoiserConfig& config) const;
};

// Output buffers for denoising
struct DenoiserOutputBuffers
{
    cudau::TypedBuffer<float4>* denoisedBeauty = nullptr;
    
    // Optional AOV outputs (for kernel prediction mode)
    cudau::TypedBuffer<float4>* denoisedAlbedo = nullptr;
    cudau::TypedBuffer<float4>* denoisedNormal = nullptr;
    
    // Internal guide layer for next frame (temporal + kernel prediction)
    optixu::BufferView internalGuideLayerForNextFrame;
};

// Denoising task for tiled execution
struct DenoiserTask
{
    uint32_t tileX = 0;
    uint32_t tileY = 0;
    uint32_t tileWidth = 0;
    uint32_t tileHeight = 0;
    uint32_t inputOffsetX = 0;
    uint32_t inputOffsetY = 0;
    uint32_t outputOffsetX = 0;
    uint32_t outputOffsetY = 0;
};

// Main DenoiserHandler class
class DenoiserHandler
{
public:
    // Factory method following handler pattern
    static DenoiserHandlerPtr create(RenderContextPtr renderContext);
    
    // Constructor/Destructor
    explicit DenoiserHandler(RenderContextPtr renderContext);
    ~DenoiserHandler();
    
    // Initialization and configuration
    bool initialize(uint32_t width, uint32_t height, const DenoiserConfig& config = DenoiserConfig());
    void finalize();
    bool isInitialized() const { return initialized_; }
    
    // Resize support
    void resize(uint32_t width, uint32_t height);
    
    // Configuration updates
    void updateConfig(const DenoiserConfig& config);
    void setModel(DenoiserConfig::Model model);
    void enableTemporalMode(bool enable);
    void enableKernelPrediction(bool enable);
    void enableUpscaling(bool enable);
    
    // Main denoising operations
    
    // Compute HDR normalizer from noisy input
    void computeNormalizer(
        CUstream stream,
        const DenoiserInputBuffers& inputs);
    
    // Execute denoising
    void denoise(
        CUstream stream,
        const DenoiserInputBuffers& inputs,
        const DenoiserOutputBuffers& outputs,
        bool isFirstFrame = false,
        float blendFactor = 0.0f);  // 0 = full denoise, 1 = full noisy
    
    // Combined normalizer + denoise (convenience method)
    void computeAndDenoise(
        CUstream stream,
        const DenoiserInputBuffers& inputs,
        const DenoiserOutputBuffers& outputs,
        bool isFirstFrame = false,
        float blendFactor = 0.0f);
    
    // High-level denoising that automatically handles buffer setup from ScreenBufferHandler
    // This method internally accesses ScreenBufferHandler through the Handlers system
    void denoiseFrame(
        CUstream stream,
        bool isFirstFrame = false,
        float blendFactor = 0.0f);
    
    // Tiled denoising for large framebuffers
    void denoiseTiled(
        CUstream stream,
        const DenoiserInputBuffers& inputs,
        const DenoiserOutputBuffers& outputs,
        bool isFirstFrame = false);
    
    // Query methods
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }
    uint32_t getDenoisedWidth() const { return denoisedWidth_; }
    uint32_t getDenoisedHeight() const { return denoisedHeight_; }
    
    const DenoiserConfig& getConfig() const { return config_; }
    bool isTemporalMode() const { return isTemporalModel(config_.model); }
    bool isUpscalingMode() const { return isUpscaleModel(config_.model); }
    
    // Buffer size queries
    size_t getStateBufferSize() const { return stateBufferSize_; }
    size_t getScratchBufferSize() const { return scratchBufferSize_; }
    size_t getInternalGuideLayerSize() const { return internalGuideLayerSize_; }
    
    // Get denoising tasks for manual execution
    const std::vector<optixu::DenoisingTask>& getTasks() const { return denoisingTasks_; }
    
    // Internal buffer access (for advanced usage)
    cudau::Buffer& getStateBuffer() { return stateBuffer_; }
    cudau::Buffer& getScratchBuffer() { return scratchBuffer_; }
    cudau::Buffer& getHDRNormalizer() { return hdrNormalizer_; }
    cudau::Buffer& getInternalGuideLayer(uint32_t index);
    
private:
    // Private members
    bool initialized_ = false;
    RenderContextPtr renderContext_;
    
    // Configuration
    DenoiserConfig config_;
    
    // Dimensions
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t denoisedWidth_ = 0;   // May be 2x for upscaling
    uint32_t denoisedHeight_ = 0;
    
    // OptiX denoiser
    optixu::Denoiser denoiser_;
    OptixDenoiserModelKind modelKind_ = OPTIX_DENOISER_MODEL_KIND_HDR;
    
    // Denoiser buffers
    cudau::Buffer stateBuffer_;
    cudau::Buffer scratchBuffer_;
    cudau::Buffer hdrNormalizer_;
    
    // Internal guide layers for temporal + kernel prediction
    cudau::Buffer internalGuideLayers_[2];  // Double buffered
    uint32_t currentGuideLayerIndex_ = 0;
    
    // Buffer sizes
    size_t stateBufferSize_ = 0;
    size_t scratchBufferSize_ = 0;
    size_t normalizerScratchSize_ = 0;
    size_t internalGuideLayerSize_ = 0;
    size_t internalGuideLayerPixelSize_ = 0;
    
    // Denoising tasks for tiling
    std::vector<optixu::DenoisingTask> denoisingTasks_;
    
    // Helper methods
    void createDenoiser();
    void destroyDenoiser();
    void prepareDenoiser();
    void allocateBuffers();
    void deallocateBuffers();
    
    // Model type helpers
    static bool isTemporalModel(DenoiserConfig::Model model);
    static bool isUpscaleModel(DenoiserConfig::Model model);
    static bool isKernelPredictionModel(DenoiserConfig::Model model);
    static OptixDenoiserModelKind getOptixModelKind(DenoiserConfig::Model model);
    
    // Validate configuration consistency
    void validateConfig();
};

// Inline helper implementations
inline bool DenoiserHandler::isTemporalModel(DenoiserConfig::Model model)
{
    return model == DenoiserConfig::Model::Temporal ||
           model == DenoiserConfig::Model::TemporalAOV ||
           model == DenoiserConfig::Model::TemporalUpscale2X;
}

inline bool DenoiserHandler::isUpscaleModel(DenoiserConfig::Model model)
{
    return model == DenoiserConfig::Model::Upscale2X ||
           model == DenoiserConfig::Model::TemporalUpscale2X;
}

inline bool DenoiserHandler::isKernelPredictionModel(DenoiserConfig::Model model)
{
    return model == DenoiserConfig::Model::AOV ||
           model == DenoiserConfig::Model::TemporalAOV;
}

inline OptixDenoiserModelKind DenoiserHandler::getOptixModelKind(DenoiserConfig::Model model)
{
    switch (model)
    {
        case DenoiserConfig::Model::HDR:
            return OPTIX_DENOISER_MODEL_KIND_HDR;
        case DenoiserConfig::Model::Temporal:
            return OPTIX_DENOISER_MODEL_KIND_TEMPORAL;
        case DenoiserConfig::Model::TemporalAOV:
            return OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV;
        case DenoiserConfig::Model::AOV:
            return OPTIX_DENOISER_MODEL_KIND_AOV;
        case DenoiserConfig::Model::Upscale2X:
            return OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
        case DenoiserConfig::Model::TemporalUpscale2X:
            return OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;
        default:
            return OPTIX_DENOISER_MODEL_KIND_HDR;
    }
}