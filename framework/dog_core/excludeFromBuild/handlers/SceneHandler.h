// SceneHandler manages the scene graph and acceleration structures for the Dog rendering system.
// It provides centralized management of geometry, transforms, and OptiX traversable handles.
//
// Resource Management:
// - Manages scene graph hierarchy
// - Handles OptiX acceleration structures (IAS/GAS)
// - Controls instance transforms and visibility
// - Manages traversable handles for ray tracing
//
// Scene Components:
// - Instance data: Per-instance transforms and properties
// - Geometry data: References to model geometry
// - Material bindings: Links instances to materials
// - Acceleration structures: OptiX BVH structures
//
// Memory Management:
// - Automatic CUDA memory allocation and deallocation
// - Proper resource cleanup in destructor
// - Support for dynamic scene updates
// - RAII principles for resource safety
//
// Integration:
// - Works with OptiX ray tracing pipeline
// - Provides traversable handles for kernel access
// - Supports dynamic scene updates
// - Integrates with CUDA runtime
//
// Usage:
// - Create via factory method SceneHandler::create()
// - Initialize with GPU context
// - Build acceleration structures after scene changes
// - Access traversable handle for ray tracing
//
// Thread Safety:
// - Not thread-safe by default
// - Requires external synchronization for multi-threaded access
// - Scene updates should be synchronized with rendering

#pragma once

#include "../common/common_host.h"
#include "../DogShared.h"
#include <sabi_core/sabi_core.h>

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

// Bring in sabi types we'll use
using sabi::RenderableNode;
using sabi::RenderableWeakRef;

namespace dog
{

using SceneHandlerPtr = std::shared_ptr<class SceneHandler>;

// SceneHandler manages scene graph and acceleration structures for ray tracing
// Provides centralized scene management with automatic resource cleanup
class SceneHandler
{
public:
    // Factory method to create a shared SceneHandler instance
    static SceneHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<SceneHandler>(ctx);
    }

    SceneHandler(RenderContextPtr ctx);
    ~SceneHandler();

    SceneHandler(const SceneHandler&) = delete;
    SceneHandler& operator=(const SceneHandler&) = delete;
    SceneHandler(SceneHandler&&) = default;
    SceneHandler& operator=(SceneHandler&&) = default;

    // Initialize the scene handler
    // Returns true if successful, false otherwise
    bool initialize();

    // Clean up all scene resources
    void finalize();

    // Build or rebuild acceleration structures
    // Call after adding/modifying scene geometry
    bool buildAccelerationStructures();

    // Update scene for the current frame
    // Handles any per-frame scene updates
    void update();

    // Check if scene is initialized
    bool isInitialized() const { return initialized_; }

    // Get the main traversable handle for ray tracing
    OptixTraversableHandle getTraversableHandle() const { return traversable_handle_; }

    // Check if scene has any geometry
    bool hasGeometry() const { return has_geometry_; }

    // Get scene data limits
    static constexpr uint32_t getMaxMaterials() { return maxNumMaterials; }
    static constexpr uint32_t getMaxGeometryInstances() { return maxNumGeometryInstances; }
    static constexpr uint32_t getMaxInstances() { return maxNumInstances; }

    // Buffer access for pipeline parameter setup
    const cudau::TypedBuffer<shared::DisneyData>& getMaterialDataBuffer() const { return material_data_buffer_; }
    const cudau::TypedBuffer<shared::GeometryInstanceData>& getGeomInstDataBuffer() const { return geom_inst_data_buffer_; }
    const cudau::TypedBuffer<shared::InstanceData>& getInstanceDataBuffer(int index) const 
    { 
        return inst_data_buffer_[index % 2]; 
    }
    const LightDistribution& getLightInstDistribution() const { return light_inst_dist_; }

private:
    // Scene limits (conservative for interactive use)
    static constexpr uint32_t maxNumMaterials = 256;
    static constexpr uint32_t maxNumGeometryInstances = 4096;
    static constexpr uint32_t maxNumInstances = 1024;

    RenderContextPtr ctx_ = nullptr;
    bool initialized_ = false;
    bool has_geometry_ = false;
    
    // OptiX acceleration structure handles
    OptixTraversableHandle traversable_handle_ = 0;
    
    // Slot management for resource allocation
    SlotFinder material_slot_finder_;
    SlotFinder geom_inst_slot_finder_;
    SlotFinder inst_slot_finder_;
    
    // Data buffers matching DogShared::StaticPipelineLaunchParameters
    cudau::TypedBuffer<shared::DisneyData> material_data_buffer_;
    cudau::TypedBuffer<shared::GeometryInstanceData> geom_inst_data_buffer_;
    cudau::TypedBuffer<shared::InstanceData> inst_data_buffer_[2]; // Double buffered for motion
    
    // OptiX Instance Acceleration Structure
    optixu::InstanceAccelerationStructure ias_;
    cudau::Buffer ias_mem_;
    cudau::TypedBuffer<OptixInstance> ias_instance_buffer_;
    cudau::Buffer as_scratch_mem_;  // Scratch memory for AS builds
    bool ias_needs_rebuild_ = true;
    
    // Light distribution for importance sampling
    LightDistribution light_inst_dist_;
    
    // Future members will include:
    // - Geometry cache (GAS storage)
    // - Instance tracking maps
};

} // namespace dog