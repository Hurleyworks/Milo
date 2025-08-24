#pragma once

#include "../RenderContext.h"


using InstanceHandlerPtr = std::shared_ptr<class InstanceHandler>;

// InstanceHandler manages the Instance Acceleration Structure (IAS) for OptiX rendering
// It provides scene graph management and traversable handle generation
class InstanceHandler
{
public:
    // Factory method to create a shared InstanceHandler instance
    static InstanceHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<InstanceHandler>(ctx);
    }

    InstanceHandler(RenderContextPtr ctx);
    ~InstanceHandler();

    // Initialize the scene handler
    void initialize();

    // Finalize and cleanup resources
    void finalize();

    // Configure IAS build options
    void setConfiguration(
        optixu::ASTradeoff tradeoff = optixu::ASTradeoff::PreferFastTrace,
        bool allowUpdate = false,
        bool allowCompaction = false);

    // Build/rebuild the IAS from current instances
    void buildIAS();

    // Update the IAS (only works if allowUpdate was true in configuration)
    void updateIAS();
    
    // Smart build/update - chooses update if possible, rebuild if necessary
    void buildOrUpdateIAS();

    // Add an instance to the scene
    void addInstance(const optixu::Instance& instance);

    // Remove instance at index
    void removeInstanceAt(uint32_t index);

    // Clear all instances
    void clearInstances();

    // Update transform for an instance
    void updateInstanceTransform(uint32_t index, const float transform[12]);

    // Mark the IAS as dirty (needs rebuild)
    void markDirty();

    // Check if IAS is ready (built and not dirty)
    bool isReady() const;

    // Get the scene traversable handle for OptiX
    OptixTraversableHandle getTraversableHandle() const;

    // Get number of instances in the scene
    size_t getInstanceCount() const;

    // Get instance at index
    optixu::Instance getInstance(uint32_t index) const;

    // Find instance index
    uint32_t findInstanceIndex(const optixu::Instance& instance) const;

private:
    RenderContextPtr renderContext_ = nullptr;
    
    // OptiX scene management
    optixu::Scene scene_;
    std::vector<optixu::Instance> instances_;
    optixu::InstanceAccelerationStructure ias_;
    
    // Acceleration structure buffers
    cudau::TypedBuffer<OptixInstance> optixInstanceBuffer_;  // Buffer for OptixInstance data
    cudau::Buffer accelBuffer_;
    // Note: scratch buffer is now shared and managed by RenderContext
    
    // Scene state
    bool isDirty_ = true;
    bool needsRebuild_ = true;  // True when structure changes (add/remove), false for transform-only changes
    bool allowUpdate_ = false;
    bool hasBeenBuilt_ = false;  // Track if IAS has been built at least once
    OptixTraversableHandle traversableHandle_ = 0;
};