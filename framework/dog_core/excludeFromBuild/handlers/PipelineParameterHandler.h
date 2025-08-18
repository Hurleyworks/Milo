// PipelineParameterHandler manages OptiX/CUDA pipeline launch parameters for the Dog rendering system.
// It provides centralized management of static and per-frame parameters with proper resource lifecycle.
//
// Parameter Management:
// - Static parameters: Image size, buffers, material data, scene data, environment lighting
// - Per-frame parameters: Camera, traversable handle, frame index, rendering options
// - Combined parameters: Device pointers to static and per-frame parameter blocks
// - Pick info buffers: Double-buffered pick information for mouse interaction
//
// Memory Management:
// - Automatic device memory allocation for parameter blocks
// - Double-buffered pick info for thread-safe reading
// - Proper cleanup through RAII principles
// - Support for copying to pure CUDA device memory
//
// Integration:
// - Works with RenderContext for CUDA/OptiX access
// - Coordinates with scene and material systems
// - Provides device pointers for kernel launches
// - Supports screen buffer and scene data updates
//
// Usage:
// - Create via factory method PipelineParameterHandler::create()
// - Initialize with render context
// - Update static/per-frame parameters as needed
// - Copy parameters to device before kernel launches
//
// Thread Safety:
// - Not thread-safe by default
// - Requires external synchronization for multi-threaded access
// - Parameter updates should be synchronized with rendering

#pragma once

#include "../common/common_host.h"
#include "../DogShared.h"

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

namespace dog
{

using PipelineParameterHandlerPtr = std::shared_ptr<class PipelineParameterHandler>;

// PipelineParameterHandler manages all OptiX/CUDA pipeline launch parameters
// Provides centralized parameter management with automatic resource cleanup
class PipelineParameterHandler
{
public:
    // Factory method to create a shared PipelineParameterHandler instance
    static PipelineParameterHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<PipelineParameterHandler>(ctx);
    }

    PipelineParameterHandler(RenderContextPtr ctx);
    ~PipelineParameterHandler();

    PipelineParameterHandler(const PipelineParameterHandler&) = delete;
    PipelineParameterHandler& operator=(const PipelineParameterHandler&) = delete;
    PipelineParameterHandler(PipelineParameterHandler&&) = default;
    PipelineParameterHandler& operator=(PipelineParameterHandler&&) = default;

    // Initialize pipeline parameters and device memory
    // Returns true if successful, false otherwise
    bool initialize();

    // Clean up pipeline parameter resources
    void finalize();

    // Check if parameter manager is initialized
    bool isInitialized() const { return initialized_; }

    // Update static pipeline parameters
    // Should be called when screen dimensions or buffer configuration changes
    void updateStaticParameters(uint32_t width, uint32_t height);

    // Copy static parameters to device (call after updating parameters)
    void copyStaticParametersToDevice();

    // Get direct access to static parameters for external coordination
    DogShared::StaticPipelineLaunchParameters& getStaticParameters() { return static_plp_; }
    const DogShared::StaticPipelineLaunchParameters& getStaticParameters() const { return static_plp_; }

    // Set static scene-related data
    // Updates material, instance, and geometry buffers from scene
    void setStaticSceneData(const shared::ROBuffer<shared::DisneyData>& materialBuffer,
                           const shared::ROBuffer<shared::InstanceData>* instanceBuffers,
                           const shared::ROBuffer<shared::GeometryInstanceData>& geometryBuffer);

    // Set environment lighting data
    void setEnvironmentLighting(CUtexObject envLightTexture,
                               const shared::RegularConstantContinuousDistribution2D& envLightImportanceMap);

    // Set light distribution for area lights
    void setLightDistribution(const shared::LightDistribution& lightDist);

    // Update per-frame pipeline parameters
    void updatePerFrameParameters(const DogShared::PerFramePipelineLaunchParameters& params);

    // Copy all parameters to device memory
    // Should be called before kernel launch
    void copyParametersToDevice(CUstream stream);

    // Copy to pure CUDA device memory for kernel coordination
    void copyToPureCUDADevice(CUstream stream, CUdeviceptr pureCudaDevicePtr);

    // Access to device memory pointers
    CUdeviceptr getStaticParametersDevice() const { return static_plp_on_device_; }
    CUdeviceptr getPerFrameParametersDevice() const { return per_frame_plp_on_device_; }
    CUdeviceptr getCombinedParametersDevice() const { return plp_on_device_; }

    // Access to pick info buffers (index 0 or 1 for double buffering)
    const cudau::TypedBuffer<DogShared::PickInfo>& getPickInfo(uint32_t index) const;
    cudau::TypedBuffer<DogShared::PickInfo>& getPickInfo(uint32_t index);

    // Get pick info device pointer for kernel parameters
    DogShared::PickInfo* getPickInfoPointer(uint32_t index) const;

    // Get number of pick info buffers
    static constexpr uint32_t getPickInfoBufferCount() { return 2; }

    // Set screen buffer surface objects (called by ScreenBufferHandler)
    void setScreenBufferSurfaces(const optixu::NativeBlockBuffer2D<DogShared::GBuffer0Elements>* gbuffer0,
                                 const optixu::NativeBlockBuffer2D<DogShared::GBuffer1Elements>* gbuffer1,
                                 const optixu::NativeBlockBuffer2D<float4>& beautyAccum,
                                 const optixu::NativeBlockBuffer2D<float4>& albedoAccum,
                                 const optixu::NativeBlockBuffer2D<float4>& normalAccum,
                                 const optixu::NativeBlockBuffer2D<shared::PCG32RNG>& rngBuffer);

private:
    RenderContextPtr render_context_;
    bool initialized_ = false;
    CUcontext cuda_context_ = nullptr;

    // Pipeline parameter structures
    DogShared::StaticPipelineLaunchParameters static_plp_;
    DogShared::PerFramePipelineLaunchParameters per_frame_plp_;
    DogShared::PipelineLaunchParameters plp_;

    // Device memory for pipeline parameters
    CUdeviceptr static_plp_on_device_ = 0;
    CUdeviceptr per_frame_plp_on_device_ = 0;
    CUdeviceptr plp_on_device_ = 0;

    // Pick info buffers (double-buffered for thread safety)
    cudau::TypedBuffer<DogShared::PickInfo> pick_infos_[2];

    // Internal initialization helpers
    void initializePickInfoBuffers();
    void setupParameterPointers();
    bool isValidPickInfoIndex(uint32_t index) const { return index < 2; }
};

} // namespace dog