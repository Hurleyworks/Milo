// TriangleMeshHandler manages geometry and material resources for the Dog rendering system.
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
// - TriangleMeshSurface: Per-surface data (triangles, material slot)
// - TriangleMesh: Complete geometry (GAS, vertices, surfaces)
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
// - Create via factory method TriangleMeshHandler::create()
// - Initialize through RenderContext
// - Call createTriangleMesh() to build geometry from CgModel
// - Use computeGeometryHash() for geometry identification
// - Access cached geometry via getGeometry()
// - Manage reference counts with increment/decrementRefCount()

#pragma once

#include "../RenderContext.h"
#include "../common/common_host.h" // For SlotFinder
#include "../material/DeviceDisneyMaterial.h"

// Forward declarations (outside namespace)
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

// TriangleMeshSurface - represents a single surface/material within a TriangleMesh
struct TriangleMeshSurface
{
    optixu::GeometryInstance optix_geom_inst;
    cudau::TypedBuffer<shared::Triangle> triangle_buffer;
    uint32_t material_slot = 0;
    AABB aabb;
};

// TriangleMesh - represents a complete geometry with potentially multiple surfaces
struct TriangleMesh
{
    optixu::GeometryAccelerationStructure gas;
    cudau::Buffer gas_mem;
    cudau::TypedBuffer<shared::Vertex> vertex_buffer; // Shared by all surfaces
    std::vector<TriangleMeshSurface> geom_instances;
    AABB aabb;
    uint32_t ref_count = 0;
    bool is_emissive = false; // Track if any surface is emissive
};

using TriangleMeshHandlerPtr = std::shared_ptr<class TriangleMeshHandler>;

// Handle returned when adding a model
struct ModelHandle
{
    size_t hash = 0;
    bool is_emissive = false;

    bool isValid() const { return hash != 0; }
    static ModelHandle invalid() { return ModelHandle{0, false}; }
};

class TriangleMeshHandler
{
 public:
    static std::unique_ptr<TriangleMeshHandler> create (RenderContextPtr ctx);

    TriangleMeshHandler() = default;
    ~TriangleMeshHandler();

    // Disable copy
    TriangleMeshHandler (const TriangleMeshHandler&) = delete;
    TriangleMeshHandler& operator= (const TriangleMeshHandler&) = delete;

    // Enable move
    TriangleMeshHandler (TriangleMeshHandler&&) = default;
    TriangleMeshHandler& operator= (TriangleMeshHandler&&) = default;

    // Initialize the handler
    bool initialize (RenderContextPtr ctx);

    // Clean up resources
    void finalize();

    // Check if initialized
    bool isInitialized() const { return initialized_; }

    // Geometry cache management
    bool hasGeometry (size_t hash) const;
    TriangleMesh* getGeometry (size_t hash);
    void addGeometry (size_t hash, TriangleMesh&& triangleMesh);
    void incrementRefCount (size_t hash);
    void decrementRefCount (size_t hash);

    // Get the number of cached geometries
    size_t getCachedGeometryCount() const { return geometry_cache_.size(); }

    // Material management
    uint32_t allocateMaterialSlot();
    void freeMaterialSlot (uint32_t slot);
    cudau::TypedBuffer<shared::DisneyData>& getMaterialDataBuffer() { return material_data_buffer_; }
    const cudau::TypedBuffer<shared::DisneyData>& getMaterialDataBuffer() const { return material_data_buffer_; }

    // Geometry creation
    bool createTriangleMesh (sabi::CgModelPtr cgModel, size_t hash, TriangleMesh& triangleMesh);
    static size_t computeGeometryHash (sabi::CgModelPtr cgModel);

    // Constants
    static constexpr uint32_t maxNumMaterials = 256;

 private:
    RenderContextPtr ctx_;
    bool initialized_ = false;

    // Geometry cache - hash to geometry resources
    std::unordered_map<size_t, TriangleMesh> geometry_cache_;

    // Material management
    SlotFinder material_slot_finder_;
    cudau::TypedBuffer<shared::DisneyData> material_data_buffer_;
};

