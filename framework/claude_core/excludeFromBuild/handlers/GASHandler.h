#pragma once

#include "../RenderContext.h"


using GASHandlerPtr = std::shared_ptr<class GASHandler>;

// GASHandler manages the Geometry Acceleration Structure (GAS) for OptiX rendering
// It provides geometry management and traversable handle generation for triangles, curves, and custom primitives
class GASHandler
{
public:
    // Factory method to create a shared GASHandler instance
    static GASHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<GASHandler>(ctx);
    }

    GASHandler(RenderContextPtr ctx);
    ~GASHandler();

    // Initialize the GAS handler
    void initialize();

    // Finalize and cleanup resources
    void finalize();

    // Configure GAS build options
    void setConfiguration(
        optixu::ASTradeoff tradeoff = optixu::ASTradeoff::PreferFastTrace,
        bool allowUpdate = false,
        bool allowCompaction = false,
        bool allowRandomVertexAccess = false,
        bool allowOpacityMicroMapUpdate = false,
        bool allowDisableOpacityMicroMaps = false);

    // Configure motion blur options
    void setMotionOptions(
        uint32_t numKeys, 
        float timeBegin, 
        float timeEnd, 
        OptixMotionFlags flags = OPTIX_MOTION_FLAG_NONE);

    // Configure material sets and ray types
    void setMaterialSetCount(uint32_t count);
    void setRayTypeCount(uint32_t matSetIdx, uint32_t count);

    // Build/rebuild the GAS from current geometry instances
    void buildGAS();

    // Update the GAS (only works if allowUpdate was true in configuration)
    void updateGAS();
    
    // Smart build/update - chooses update if possible, rebuild if necessary
    void buildOrUpdateGAS();

    // Compact the GAS (only works if allowCompaction was true in configuration)
    void compactGAS();

    // Add a geometry instance to the GAS
    void addGeometryInstance(
        const optixu::GeometryInstance& geomInst, 
        CUdeviceptr preTransform = 0,
        const void* userData = nullptr, 
        uint32_t userDataSize = 0, 
        uint32_t userDataAlignment = 1);

    // Template version for user data
    template <typename T>
    void addGeometryInstance(
        const optixu::GeometryInstance& geomInst, 
        CUdeviceptr preTransform, 
        const T& userData)
    {
        addGeometryInstance(geomInst, preTransform, &userData, sizeof(T), alignof(T));
    }

    // Remove geometry instance at index
    void removeGeometryInstanceAt(uint32_t index);

    // Clear all geometry instances
    void clearGeometryInstances();

    // Update per-child user data
    void setChildUserData(uint32_t index, const void* data, uint32_t size, uint32_t alignment);
    
    template <typename T>
    void setChildUserData(uint32_t index, const T& data)
    {
        setChildUserData(index, &data, sizeof(T), alignof(T));
    }

    // Set GAS-wide user data
    void setUserData(const void* data, uint32_t size, uint32_t alignment);
    
    template <typename T>
    void setUserData(const T& data)
    {
        setUserData(&data, sizeof(T), alignof(T));
    }

    // Mark the GAS as dirty (needs rebuild)
    void markDirty();

    // Check if GAS is ready (built and not dirty)
    bool isReady() const;

    // Get the GAS traversable handle for OptiX
    OptixTraversableHandle getTraversableHandle() const;

    // Get number of geometry instances in the GAS
    uint32_t getChildCount() const;

    // Get geometry instance at index
    optixu::GeometryInstance getChild(uint32_t index, CUdeviceptr* preTransform = nullptr) const;

    // Find geometry instance index
    uint32_t findChildIndex(const optixu::GeometryInstance& geomInst, CUdeviceptr preTransform = 0) const;

    // Get configuration
    uint32_t getMaterialSetCount() const;
    uint32_t getRayTypeCount(uint32_t matSetIdx) const;
    optixu::GeometryType getGeometryType() const;

    // Get compacted size (after compaction)
    size_t getCompactedSize() const { return compactedSize_; }

    // Check if compaction is beneficial
    bool shouldCompact() const;

private:
    RenderContextPtr renderContext_ = nullptr;
    
    // OptiX scene management
    optixu::Scene scene_;
    optixu::GeometryAccelerationStructure gas_;
    
    // Geometry instances and their pre-transforms
    struct GeometryChild {
        optixu::GeometryInstance instance;
        CUdeviceptr preTransform = 0;
    };
    std::vector<GeometryChild> children_;
    
    // Acceleration structure buffers
    cudau::Buffer accelBuffer_;
    cudau::Buffer compactedAccelBuffer_;
    // Note: scratch buffer is now shared and managed by RenderContext
    
    // GAS state
    bool isDirty_ = true;
    bool needsRebuild_ = true;  // True when structure changes (add/remove), false for vertex-only changes
    bool allowUpdate_ = false;
    bool allowCompaction_ = false;
    bool hasBeenBuilt_ = false;  // Track if GAS has been built at least once
    bool isCompacted_ = false;   // Track if GAS has been compacted
    size_t compactedSize_ = 0;   // Size after compaction
    OptixTraversableHandle traversableHandle_ = 0;
    
    // Motion blur settings
    bool hasMotion_ = false;
    uint32_t motionKeyCount_ = 0;
};