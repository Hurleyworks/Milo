#pragma once

// GPUContext manages CUDA and OptiX context initialization and lifecycle.
// Provides centralized access to GPU resources including device, stream, and OptiX context.
// Handles GPU device selection and capability checking.

#include "common/common_host.h"

class PTXManager; // forward declaration

class GPUContext
{
 public:
    GPUContext();
    ~GPUContext();

    // Initialize GPU context and OptiX
    bool initialize();

    // Cleanup resources
    void cleanup();

    // Check if initialized
    bool isInitialized() const { return initialized_; }

    // Set PTX manager (called from Renderer after PTX compilation)
    void setPTXManager (PTXManager* ptxManager) { ptxManager_ = ptxManager; }

    // Get PTX manager
    PTXManager* getPTXManager() const { return ptxManager_; }

    // Getters for context resources
    CUcontext getCudaContext() const { return cuContext_; }
    CUstream getCudaStream() const { return cuStream_; }
    optixu::Context getOptiXContext() const { return optixContext_; }

    // Get pointer to OptiX context (for pipeline initialization)
    optixu::Context* getOptiXContextPtr() { return &optixContext_; }

    // GPU capability info
    int getComputeCapabilityMajor() const { return computeCapabilityMajor_; }
    int getComputeCapabilityMinor() const { return computeCapabilityMinor_; }

 private:
    // GPU resources
    CUcontext cuContext_ = nullptr;
    CUstream cuStream_ = nullptr;
    optixu::Context optixContext_;

    int computeCapabilityMajor_ = 0;
    int computeCapabilityMinor_ = 0;

    // PTX manager for loading kernels
    PTXManager* ptxManager_ = nullptr;

    // State
    bool initialized_ = false;

    // Helper methods
    bool checkGPUCapability();
};