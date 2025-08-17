#pragma once

#include "common/common_host.h"

// Minimal GPU context manager focused solely on CUDA and OptiX context lifecycle
// Provides the foundational GPU environment that all other systems depend on
// Module loading, kernels, and other resources are managed by separate specialized classes
class GPUContext
{
 public:
    GPUContext() = default;
    ~GPUContext();

    GPUContext (const GPUContext&) = delete;
    GPUContext& operator= (const GPUContext&) = delete;
    GPUContext (GPUContext&&) = default;
    GPUContext& operator= (GPUContext&&) = default;

    // Initialize the GPU environment (CUDA and OptiX contexts only)
    bool initialize (int deviceIndex = 0);

    // Clean up GPU resources
    void finalize();

    // Check if GPU context is initialized
    bool isInitialized() const { return initialized_; }

    // Access to GPU contexts
    CUcontext getCudaContext() const { return cuda_context_; }
    CUstream getCudaStream() const { return cuda_stream_; }
    optixu::Context getOptixContext() const { return optix_context_; }

    // Get compute capability
    int getComputeCapabilityMajor() const { return compute_capability_major_; }
    int getComputeCapabilityMinor() const { return compute_capability_minor_; }
    int getComputeCapability() const { return compute_capability_major_ * 10 + compute_capability_minor_; }

    // Device information
    int getDeviceIndex() const { return device_index_; }
    std::string getDeviceName() const { return device_name_; }
    size_t getTotalMemory() const { return total_memory_; }

 private:
    bool initialized_ = false;
    
    // Device information
    int device_index_ = 0;
    std::string device_name_;
    size_t total_memory_ = 0;
    int compute_capability_major_ = 0;
    int compute_capability_minor_ = 0;

    // Core GPU contexts
    CUcontext cuda_context_ = nullptr;
    CUstream cuda_stream_ = nullptr;
    optixu::Context optix_context_;

    // Internal initialization methods
    bool initializeCuda (int deviceIndex);
    bool initializeOptix();
    bool queryDeviceCapabilities();
    void cleanupCuda();
    void cleanupOptix();
};