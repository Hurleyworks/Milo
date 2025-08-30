#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <sabi_core/sabi_core.h>
#include <claude_core/excludeFromBuild/common/common_host.h>

using GASHandlerPtr = std::shared_ptr<class GASHandler>;

// GASHandler manages multiple Geometry Acceleration Structures (GAS) in a scene
// It decouples GAS creation and management from geometry sources
class GASHandler
{
public:
    // Factory method to create a shared GASHandler instance
    static GASHandlerPtr create(optixu::Scene scene, CUcontext cudaContext)
    {
        return std::make_shared<GASHandler>(scene, cudaContext);
    }

    GASHandler(optixu::Scene scene, CUcontext cudaContext);
    ~GASHandler();

    // Initialize the handler
    void initialize();

    // Finalize and cleanup resources
    void finalize();

    // Create a GAS from a geometry instance and return its ID
    // The GAS is not built immediately - call buildGAS or buildAllDirty to build
    uint32_t createGAS(
        const optixu::GeometryInstance& geomInst,
        optixu::ASTradeoff tradeoff = optixu::ASTradeoff::PreferFastTrace,
        bool allowUpdate = false,
        bool allowCompaction = true);

    // Create a GAS from multiple geometry instances (for complex objects)
    uint32_t createGAS(
        const std::vector<optixu::GeometryInstance>& geomInsts,
        optixu::ASTradeoff tradeoff = optixu::ASTradeoff::PreferFastTrace,
        bool allowUpdate = false,
        bool allowCompaction = true);

    // Get a GAS by ID
    optixu::GeometryAccelerationStructure getGAS(uint32_t gasId) const;

    // Get the traversable handle for a GAS
    OptixTraversableHandle getTraversableHandle(uint32_t gasId) const;

    // Check if a GAS exists
    bool hasGAS(uint32_t gasId) const;

    // Remove a GAS by ID
    void removeGAS(uint32_t gasId);

    // Build a specific GAS
    void buildGAS(uint32_t gasId, CUstream stream);

    // Build all dirty (unbuilt or modified) GAS objects
    void buildAllDirty(CUstream stream);

    // Rebuild all GAS (useful after major scene changes)
    void rebuildAll(CUstream stream);

    // Compact a specific GAS if beneficial
    void compactGAS(uint32_t gasId, CUstream stream);

    // Compact all GAS that would benefit from compaction
    void compactAll(CUstream stream);

    // Mark a GAS as dirty (needs rebuild)
    void markDirty(uint32_t gasId);

    // Check if a GAS is dirty
    bool isDirty(uint32_t gasId) const;

    // Get the number of managed GAS objects
    uint32_t getGASCount() const;

    // Get all GAS IDs
    std::vector<uint32_t> getAllGASIds() const;

    // Memory management
    size_t getTotalMemoryUsage() const;
    size_t getGASMemoryUsage(uint32_t gasId) const;

    // Configure material sets and ray types for a GAS
    void setMaterialConfiguration(
        uint32_t gasId,
        uint32_t materialSetCount,
        const std::vector<uint32_t>& rayTypeCounts);

    // Set motion blur options for a GAS
    void setMotionOptions(
        uint32_t gasId,
        uint32_t numKeys,
        float timeBegin,
        float timeEnd,
        OptixMotionFlags flags = OPTIX_MOTION_FLAG_NONE);

    // Find GAS containing a specific geometry instance
    uint32_t findGASContaining(const optixu::GeometryInstance& geomInst) const;

    // Get shared scratch buffer (for external use if needed)
    cudau::Buffer& getScratchBuffer() { return scratchBuffer_; }

private:
    // Internal GAS data structure
    struct GASData
    {
        optixu::GeometryAccelerationStructure gas;
        cudau::Buffer accelBuffer;
        cudau::Buffer compactedBuffer;
        std::vector<optixu::GeometryInstance> children;
        OptixTraversableHandle traversableHandle = 0;
        size_t memoryUsage = 0;
        size_t potentialCompactedSize = 0;
        bool isDirty = true;
        bool hasBeenBuilt = false;
        bool isCompacted = false;
        bool allowUpdate = false;
        bool allowCompaction = true;
        optixu::ASTradeoff tradeoff = optixu::ASTradeoff::PreferFastTrace;
        
        // Motion blur settings
        bool hasMotion = false;
        uint32_t motionKeyCount = 0;
        float motionTimeBegin = 0.0f;
        float motionTimeEnd = 1.0f;
        OptixMotionFlags motionFlags = OPTIX_MOTION_FLAG_NONE;
        
        // Material configuration
        uint32_t materialSetCount = 1;
        std::vector<uint32_t> rayTypeCounts;
    };

    // Build a single GAS
    void buildGASInternal(uint32_t gasId, GASData& gasData, CUstream stream);

    // Compact a single GAS
    void compactGASInternal(uint32_t gasId, GASData& gasData, CUstream stream);

    // Allocate or resize scratch buffer to meet size requirement
    void ensureScratchBufferSize(size_t requiredSize);

    // Get next available GAS ID
    uint32_t getNextGASId();

private:
    optixu::Scene scene_;
    CUcontext cudaContext_ = nullptr;
    
    // GAS storage - using unordered_map for O(1) lookup
    std::unordered_map<uint32_t, GASData> gasMap_;
    
    // Shared scratch buffer for all GAS builds
    cudau::Buffer scratchBuffer_;
    size_t currentScratchSize_ = 0;
    
    // ID generation
    uint32_t nextGASId_ = 1;  // Start from 1, 0 means invalid
    
    // Statistics
    size_t totalMemoryUsage_ = 0;
    uint32_t totalBuilds_ = 0;
    uint32_t totalCompactions_ = 0;
    
    // Configuration
    bool initialized_ = false;
    
    // Default configurations
    static constexpr size_t MIN_SCRATCH_SIZE = 1024 * 1024;      // 1 MB minimum
    static constexpr size_t MAX_SCRATCH_SIZE = 256 * 1024 * 1024; // 256 MB maximum
    static constexpr float COMPACTION_THRESHOLD = 0.8f;          // Compact if saves > 20%
};