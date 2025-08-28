#pragma once
#include "../../../RenderContext.h"
#include "../../../common/common_host.h"
#include "../models/ClaudiaModel.h"
#include "../claudia_shared.h"
#include <memory>
#include <map>
#include <set>

class ClaudiaSceneHandler;
class ClaudiaModelHandler;
class ClaudiaAreaLightHandler;
using ClaudiaAreaLightHandlerPtr = std::shared_ptr<ClaudiaAreaLightHandler>;

// ClaudiaAreaLightHandler - Manages area light distributions at all hierarchy levels
// This handler encapsulates all operations involving light distributions for:
// - Primitive level (triangles within geometry)
// - Geometry instance level (geometry within instances)
// - Instance level (instances within scene)
class ClaudiaAreaLightHandler
{
public:
    // Factory method to create handler
    static ClaudiaAreaLightHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<ClaudiaAreaLightHandler>(ctx);
    }

    // Constructor/Destructor
    ClaudiaAreaLightHandler(RenderContextPtr ctx);
    ~ClaudiaAreaLightHandler();

    // Initialization and cleanup
    void initialize();
    void finalize();

    // Set handler dependencies
    void setSceneHandler(std::shared_ptr<ClaudiaSceneHandler> sceneHandler)
    {
        sceneHandler_ = sceneHandler;
    }
    
    void setModelHandler(std::shared_ptr<ClaudiaModelHandler> modelHandler)
    {
        modelHandler_ = modelHandler;
    }

    // Scene-level light distribution management
    void prepareSceneLightDistribution(uint32_t maxInstances);
    void updateSceneLightDistribution(CUstream stream, uint32_t bufferIndex = 0);
    void setupSceneLightSampling(CUstream stream, CUdeviceptr lightInstDistAddr, uint32_t bufferIndex);

    // Instance-level light distribution management
    void prepareInstanceLightDistribution(ClaudiaModel* model);
    void updateInstanceLightDistribution(CUstream stream, ClaudiaModel* model, uint32_t instSlot);
    void markInstanceDirty(ClaudiaModel* model);

    // Geometry-level light distribution management
    void prepareGeometryLightDistribution(ClaudiaTriangleModel* triModel, uint32_t numTriangles);
    void updateGeometryLightDistribution(CUstream stream, ClaudiaTriangleModel* triModel, uint32_t geomInstSlot, uint32_t materialSlot);
    void markGeometryDirty(ClaudiaTriangleModel* triModel);

    // Full hierarchy update methods
    void rebuildAllDistributions(CUstream stream);
    void updateDirtyDistributions(CUstream stream, uint32_t bufferIndex = 0);
    
    // Query methods
    bool needsRebuild() const { return needsFullRebuild_; }
    bool hasDirtyDistributions() const { return !dirtyInstances_.empty() || !dirtyGeometries_.empty(); }
    float getTotalLightPower() const;
    
    // Memory management
    void resizeScratchMemory(size_t requiredSize);
    
    // Light distribution configuration
    bool getUseProbabilityTextures() const 
    { 
#if USE_PROBABILITY_TEXTURE
        return true;
#else
        return false;
#endif
    }

    // Get the scene light distribution for device use
    LightDistribution* getSceneLightDistribution() { return &sceneLightDist_; }
    const LightDistribution* getSceneLightDistribution() const { return &sceneLightDist_; }

private:
    // Helper methods for distribution updates
    void computeTriangleProbabilities(CUstream stream, uint32_t geomInstSlot, uint32_t numTriangles, uint32_t materialSlot, LightDistribution& dist);
    void computeGeomInstProbabilities(CUstream stream, uint32_t instSlot, uint32_t numGeomInsts, LightDistribution& dist);
    void computeInstProbabilities(CUstream stream, uint32_t numInsts);
    void finalizeLightDistribution(CUstream stream, LightDistribution& dist, uint32_t numElements);
    
public:
    // Finalize emitter distribution after computing triangle probabilities
    void finalizeEmitterDistribution(CUstream stream, ClaudiaModelPtr model, uint32_t geomInstSlot);
    
private:
    
    // Compute probability texture dimensions
    uint2 computeProbabilityTextureDimensions(uint32_t numElements) const;
    
    // Light probability computation kernels structure
    struct ComputeProbTex {
        CUmodule cudaModule = nullptr;
        cudau::Kernel computeFirstMip;
        cudau::Kernel computeTriangleProbTexture;
        cudau::Kernel computeGeomInstProbTexture;
        cudau::Kernel computeInstProbTexture;
        cudau::Kernel computeMip;
        cudau::Kernel computeTriangleProbBuffer;
        cudau::Kernel computeGeomInstProbBuffer;
        cudau::Kernel computeInstProbBuffer;
        cudau::Kernel finalizeDiscreteDistribution1D;
        cudau::Kernel test;
    };
    
    // Initialize light probability computation kernels
    void initializeLightKernels();
    
    // Clean up light kernels
    void cleanupLightKernels();

private:
    RenderContextPtr ctx_ = nullptr;
    
    // Handler references (not owned)
    std::shared_ptr<ClaudiaSceneHandler> sceneHandler_;
    std::shared_ptr<ClaudiaModelHandler> modelHandler_;

    // Light distributions at each hierarchy level
    LightDistribution sceneLightDist_;                              // Scene-level instance distribution
    std::map<ClaudiaModel*, LightDistribution> instanceLightDists_; // Instance-level geometry distributions
    // Note: Geometry-level distributions are now owned by the models themselves

    // Scratch memory for GPU operations
    cudau::Buffer scanScratchMem_;
    size_t scanScratchSize_ = 0;

    // State tracking
    bool needsFullRebuild_ = false;
    std::set<ClaudiaModel*> dirtyInstances_;
    std::set<ClaudiaTriangleModel*> dirtyGeometries_;
    bool sceneDistributionPrepared_ = false;  // Track if scene distribution has been prepared
    uint32_t lastUpdateNumInstances_ = 0;     // Track last number of instances updated

    // Configuration
    uint32_t maxInstances_ = 0;
    uint32_t maxGeomInstances_ = 0;
    
    // Initialization state
    bool isInitialized_ = false;
    
    // Light computation kernels
    ComputeProbTex computeProbTex_;

    // Constants
    static constexpr uint32_t maxNumInstances = 16384;
    static constexpr uint32_t maxNumGeometryInstances = 65536;
    static constexpr uint32_t maxNumMaterials = 1024;
};