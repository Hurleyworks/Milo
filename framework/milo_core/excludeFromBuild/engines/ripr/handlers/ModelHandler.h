// ModelHandler manages geometry and material resources for the Dog rendering system.
// It provides centralized management of geometry caching, material allocation, and GAS building.
//
// Primary Responsibilities:
// - Geometry cache management with reference counting
// - Material slot allocation and tracking
// - Geometry Acceleration Structure (GAS) building
// - Vertex and triangle buffer management
// - Emissive material detection
//
// Geometry Management:
// - Hash-based geometry caching for reuse (flyweight pattern)
// - Reference counting for shared geometry instances
// - Automatic cleanup when geometry is no longer referenced
// - Support for multi-surface models (CgModel)
//
// Material Management:
// - Material slot allocation using SlotFinder
// - Disney material data buffer management
// - Per-surface material assignment
// - Emissive material detection and flagging
//
// Resource Types:
// - GeometryInstanceResources: Per-surface data (triangles, material slot)
// - GeometryGroupResources: Complete geometry (GAS, vertices, surfaces)
// - Material slots: Allocated per surface for material properties
//
// Memory Management:
// - Automatic CUDA memory allocation and deallocation
// - Reference counting for geometry sharing
// - Proper resource cleanup in destructor
// - RAII principles for resource safety
//
// Integration:
// - Works with SceneHandler for instance management
// - Creates OptiX Geometry Acceleration Structures
// - Provides geometry resources for ray tracing
// - Detects emissive surfaces for area lighting
//
// Usage:
// - Create via factory method ModelHandler::create()
// - Initialize through RenderContext
// - Call createGeometryGroup() to build geometry from CgModel
// - Use computeGeometryHash() for geometry identification
// - Access cached geometry via getGeometry()
// - Manage reference counts with increment/decrementRefCount()

#pragma once

#include "../../../common/common_host.h"

// Forward declarations (outside namespace)
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
using shared::DisneyData;

// Geometry instance resources - represents a single surface/material within a geometry group
struct GeometryInstanceResources
{
    optixu::GeometryInstance optix_geom_inst;
    cudau::TypedBuffer<shared::Triangle> triangle_buffer;
    uint32_t material_slot = 0;
    AABB aabb;
};

// Geometry group resources - represents a complete geometry with potentially multiple surfaces
struct GeometryGroupResources
{
    optixu::GeometryAccelerationStructure gas;
    cudau::Buffer gas_mem;
    cudau::TypedBuffer<shared::Vertex> vertex_buffer;  // Shared by all surfaces
    std::vector<GeometryInstanceResources> geom_instances;
    AABB aabb;
    uint32_t ref_count = 0;
    bool is_emissive = false;  // Track if any surface is emissive
};

// Handle returned when adding a model
struct ModelHandle
{
    size_t hash = 0;
    bool is_emissive = false;
    
    bool isValid() const { return hash != 0; }
    static ModelHandle invalid() { return ModelHandle{0, false}; }
};

class ModelHandler
{
public:
    static std::unique_ptr<ModelHandler> create(RenderContextPtr ctx);
    
    ModelHandler() = default;
    ~ModelHandler();
    
    // Disable copy
    ModelHandler(const ModelHandler&) = delete;
    ModelHandler& operator=(const ModelHandler&) = delete;
    
    // Enable move
    ModelHandler(ModelHandler&&) = default;
    ModelHandler& operator=(ModelHandler&&) = default;
    
    // Initialize the handler
    bool initialize(RenderContextPtr ctx);
    
    // Clean up resources
    void finalize();
    
    // Check if initialized
    bool isInitialized() const { return initialized_; }
    
    // Geometry cache management
    bool hasGeometry(size_t hash) const;
    GeometryGroupResources* getGeometry(size_t hash);
    void addGeometry(size_t hash, GeometryGroupResources&& resources);
    void incrementRefCount(size_t hash);
    void decrementRefCount(size_t hash);
    
    // Get the number of cached geometries
    size_t getCachedGeometryCount() const { return geometry_cache_.size(); }
    
    // Material management
    uint32_t allocateMaterialSlot();
    void freeMaterialSlot(uint32_t slot);
    cudau::TypedBuffer<shared::DisneyData>& getMaterialDataBuffer() { return material_data_buffer_; }
    const cudau::TypedBuffer<shared::DisneyData>& getMaterialDataBuffer() const { return material_data_buffer_; }
    
    // Geometry creation
    bool createGeometryGroup(sabi::CgModelPtr cgModel, size_t hash, GeometryGroupResources& resources);
    static size_t computeGeometryHash(sabi::CgModelPtr cgModel);
    
    // Constants
    static constexpr uint32_t maxNumMaterials = 256;
    
private:
    RenderContextPtr ctx_;
    bool initialized_ = false;
    
    // Geometry cache - hash to geometry resources
    std::unordered_map<size_t, GeometryGroupResources> geometry_cache_;
    
    // Material management
    SlotFinder material_slot_finder_;
    cudau::TypedBuffer<shared::DisneyData> material_data_buffer_;
};

using ModelHandlerPtr = std::shared_ptr<ModelHandler>;

