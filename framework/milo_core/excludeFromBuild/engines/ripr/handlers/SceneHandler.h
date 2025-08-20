// SceneHandler manages the scene graph and acceleration structures for the Dog rendering system.
// It provides centralized management of instances, transforms, and OptiX traversable handles.
//
// Primary Responsibilities:
// - Instance management and transforms
// - OptiX Instance Acceleration Structure (IAS) building
// - Scene traversable handle management
// - Light distribution for emissive instances
//
// Delegation to ModelHandler:
// - Geometry creation and caching
// - Material slot allocation
// - Geometry Acceleration Structure (GAS) management
// - Emissive material detection
//
// Scene Components:
// - Instance data: Per-instance transforms and properties
// - Geometry references: Cached in ModelHandler
// - Light distribution: For importance sampling emissive geometry
// - Acceleration structures: OptiX IAS for ray traversal
//
// Memory Management:
// - Automatic CUDA memory allocation and deallocation
// - Proper resource cleanup in destructor
// - Support for dynamic scene updates
// - Reference counting for shared geometry (via ModelHandler)
//
// Integration:
// - Works with OptiX ray tracing pipeline
// - Provides traversable handles for kernel access
// - Delegates geometry management to ModelHandler
// - Supports dynamic scene updates
//
// Usage:
// - Create via factory method SceneHandler::create()
// - Initialize through RenderContext
// - Add/remove RenderableNodes dynamically
// - Build acceleration structures after scene changes
// - Access traversable handle for ray tracing

#pragma once

#include "../common/common_host.h"
#include "../DogShared.h"
#include "ModelHandler.h"  // For GeometryGroupResources and GeometryInstanceResources
#include <sabi_core/sabi_core.h>

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

// Bring in sabi types we'll use
using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::CgModelPtr;

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
    
    // Update scene-dependent shader binding tables
    // Must be called after scene changes (geometry added/removed)
    void updateSceneDependentSBTs();

    // Check if scene is initialized
    bool isInitialized() const { return initialized_; }

    // Get the main traversable handle for ray tracing
    OptixTraversableHandle getTraversableHandle() const { return traversable_handle_; }

    // Check if scene has any geometry
    bool hasGeometry() const { return has_geometry_; }

    // Get scene data limits
    static constexpr uint32_t getMaxMaterials() { return ModelHandler::maxNumMaterials; }
    static constexpr uint32_t getMaxGeometryInstances() { return maxNumGeometryInstances; }
    static constexpr uint32_t getMaxInstances() { return maxNumInstances; }

    // Buffer access for pipeline parameter setup
    // Material buffer is now accessed through ModelHandler
    const cudau::TypedBuffer<shared::DisneyData>& getMaterialDataBuffer() const;
    const cudau::TypedBuffer<shared::GeometryInstanceData>& getGeomInstDataBuffer() const { return geom_inst_data_buffer_; }
    const cudau::TypedBuffer<shared::InstanceData>& getInstanceDataBuffer(int index) const 
    { 
        return inst_data_buffer_[index % 2]; 
    }
    const LightDistribution& getLightInstDistribution() const { return light_inst_dist_; }
    
    // RenderableNode management
    bool addRenderableNode(RenderableWeakRef node);
    bool removeRenderableNode(RenderableWeakRef node);
    bool removeRenderableNodeByID(ItemID nodeID);
    size_t getNodeCount() const { return node_resources_.size(); }
    

private:
    // Scene limits (conservative for interactive use)
    // Note: maxNumMaterials is now defined in ModelHandler
    static constexpr uint32_t maxNumGeometryInstances = 4096;
    static constexpr uint32_t maxNumInstances = 1024;
    
    // Structure to track resources for each RenderableNode
    struct NodeResources
    {
        RenderableWeakRef node;
        uint32_t instance_slot = UINT32_MAX;     // Slot in instance buffer
        uint32_t geom_inst_slot = UINT32_MAX;    // Slot in geometry instance buffer
        size_t geometry_hash = 0;                // Hash of the CgModel geometry
        bool is_emissive = false;                // Whether this node has emissive material
        
        // OptiX resources
        optixu::GeometryInstance optix_geom_inst;
        uint32_t optix_instance_index = UINT32_MAX;  // Index in IAS instance buffer
    };
    
    // GeometryInstanceResources and GeometryGroupResources are now defined in ModelHandler.h
    // ModelHandler is accessed through RenderContext

    RenderContextPtr ctx_ = nullptr;
    bool initialized_ = false;
    bool has_geometry_ = false;
    
    // OptiX acceleration structure handles
    OptixTraversableHandle traversable_handle_ = 0;
    
    // Slot management for resource allocation
    SlotFinder geom_inst_slot_finder_;
    SlotFinder inst_slot_finder_;
    
    // Data buffers matching DogShared::StaticPipelineLaunchParameters
    // Note: material_data_buffer_ is now managed by ModelHandler
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
    
    // Node tracking - maps ItemID to resources
    std::unordered_map<ItemID, NodeResources> node_resources_;
    
    // Helper method for creating node instances
    bool createNodeInstance(NodeResources& nodeRes, const GeometryGroupResources& geomGroup);
};

} // namespace dog