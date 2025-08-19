#pragma once

// Stub AreaLightHandler for RiPR compatibility
// This is a placeholder to maintain compatibility after removing Shocker engine

#include <memory>

class AreaLightHandler
{
public:
    AreaLightHandler() = default;
    ~AreaLightHandler() = default;
    
    // Stub methods - add actual implementation if needed
    void initialize() {}
    
    // Initialize with CUDA context and max lights (for RiPR compatibility)
    template<typename CudaContext>
    void initialize(CudaContext context, int maxLights) {
        // Stub implementation
    }
    
    void finalize() {}
    void update() {}
};

using AreaLightHandlerPtr = std::shared_ptr<AreaLightHandler>;