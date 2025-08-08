#pragma once

#include "../../../../engine_core.h"
#include "../../../common/LightDistribution.h"

class ShockerSceneHandler;
class ShockerModelHandler;
class ShockerMaterialHandler;

// Use shocker namespace types
namespace shocker
{
    struct ShockerSurface;
    struct ShockerNode;
} // namespace shocker
using ShockerSurface = shocker::ShockerSurface;
using ShockerNode = shocker::ShockerNode;

struct DisneyMaterial;

// Manages importance sampling distributions for mesh-based area lights
class AreaLightHandler
{
 public:
    AreaLightHandler() = default;
    ~AreaLightHandler();

    // Initialization and cleanup
    void initialize (CUcontext context, uint32_t maxInstances);
    void finalize();

    // Set dependencies after construction
    void setSceneHandler (ShockerSceneHandler* sceneHandler);
    void setModelHandler (ShockerModelHandler* modelHandler);
    void setMaterialHandler (ShockerMaterialHandler* materialHandler);

    // Area Light Management

    // Called when materials change
    void onMaterialAssigned (shocker::ShockerSurface* surface,
                             DisneyMaterial* oldMaterial,
                             DisneyMaterial* newMaterial);

    // Called when geometry changes
    void onSurfaceAdded (shocker::ShockerSurface* surface);
    void onSurfaceRemoved (shocker::ShockerSurface* surface);
    void onSurfaceGeometryChanged (shocker::ShockerSurface* surface);

    // Called when instances change
    void onNodeAdded (shocker::ShockerNode* node);
    void onNodeRemoved (shocker::ShockerNode* node);
    void onNodeTransformChanged (shocker::ShockerNode* node);

    // Distribution Updates

    // Main update - called before rendering
    void updateAreaLightDistributions (CUstream stream);

    // Force immediate update (for debugging/profiling)
    void forceUpdateAll (CUstream stream);

    // Queries

    // Check if scene has any area lights
    bool hasAreaLights() const;

    // Get the scene-wide area light distribution
    const engine::LightDistribution& getSceneAreaLightDistribution() const;

    // Statistics
    uint32_t getNumAreaLights() const;        // Number of emissive surfaces
    uint32_t getNumEmissiveTriangles() const; // Total emissive primitives
    float getTotalAreaLightPower() const;     // Total emitted power

    // Debug/Profiling
    float getLastUpdateTimeMs() const;
    bool isDirty() const;

 private:
    // Dependencies (not owned)
    ShockerSceneHandler* sceneHandler_ = nullptr;
    ShockerModelHandler* modelHandler_ = nullptr;
    ShockerMaterialHandler* materialHandler_ = nullptr;

    // CUDA context
    CUcontext cuContext_ = nullptr;
    uint32_t maxInstances_ = 0;

    // Scene-wide distribution for all area lights
    engine::LightDistribution sceneAreaLightDist_;

    // Tracking area lights
    struct AreaLightState
    {
        // Dirty tracking
        bool sceneDistDirty = true;
        std::set<shocker::ShockerNode*> dirtyNodes;
        std::set<shocker::ShockerSurface*> dirtySurfaces;

        // Active area lights
        std::set<shocker::ShockerSurface*> emissiveSurfaces;
        std::set<shocker::ShockerNode*> nodesWithAreaLights;

        // Statistics
        uint32_t totalEmissiveTriangles = 0;
        float totalPower = 0.0f;

        bool isDirty() const
        {
            return sceneDistDirty || !dirtyNodes.empty() || !dirtySurfaces.empty();
        }

        void clear()
        {
            sceneDistDirty = false;
            dirtyNodes.clear();
            dirtySurfaces.clear();
        }
    } areaLightState_;

    // GPU compute kernels for area light importance
    struct AreaLightKernels
    {
        CUfunction computeTriangleImportance = nullptr;
        CUfunction computeGeomInstImportance = nullptr;
        CUfunction computeInstanceImportance = nullptr;
        CUfunction buildProbabilityTexture = nullptr;
    } kernels_;

    // Scratch buffers for GPU computation
    cudau::TypedBuffer<float> scratchImportanceBuffer_;
    cudau::TypedBuffer<uint32_t> scratchIndexBuffer_;

    // Timing
    float lastUpdateTimeMs_ = 0.0f;

    // Internal update methods
    void updateSurfaceAreaLights (CUstream stream);
    void updateNodeAreaLights (CUstream stream);
    void updateSceneAreaLights (CUstream stream);

    // Helper methods
    bool isAreaLight (const shocker::ShockerSurface* surface) const;
    shocker::ShockerNode* findNodeForSurface (shocker::ShockerSurface* surface) const;
    void initializeOrUpdateAreaLight (shocker::ShockerSurface* surface, CUstream stream);
    void cleanupAreaLight (shocker::ShockerSurface* surface);

    // GPU kernel dispatch
    void launchComputeTriangleImportanceKernel (shocker::ShockerSurface* surface, CUstream stream);
    void launchComputeGeomInstImportanceKernel (shocker::ShockerNode* node, CUstream stream);
    void launchComputeInstanceImportanceKernel (CUstream stream);
    void launchBuildProbabilityTextureKernel (engine::LightDistribution& dist, CUstream stream);

    // Load GPU kernels
    bool loadKernels();
    void unloadKernels();
};
