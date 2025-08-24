#pragma once

// RiPRSceneHandler.h
// Manages the OptiX scene graph including instances, acceleration structures,
// selection state, and traversal for ray tracing

#include "../../../RenderContext.h"
#include "../../../handlers/InstanceHandler.h"

using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::WeakRenderableList;

// Forward declaration
using RiPRSceneHandlerPtr = std::shared_ptr<class RiPRSceneHandler>;

// Forward declarations
class RiPRModelHandler;
using RiPRModelHandlerPtr = std::shared_ptr<RiPRModelHandler>;

class RiPRSceneHandler
{
 public:
    // Factory function for creating RiPRSceneHandler objects
    static RiPRSceneHandlerPtr create (RenderContextPtr ctx) { return std::make_shared<RiPRSceneHandler> (ctx); }

    // Map type for tracking scene nodes by instance index
    using NodeMap = std::unordered_map<uint32_t, RenderableWeakRef>;

 public:
    // Constructor initializes the scene handler with a render context
    RiPRSceneHandler (RenderContextPtr ctx);

    // Destructor handles cleanup of scene resources
    ~RiPRSceneHandler();

    // Set the model handler (must be called before using model operations)
    void setModelHandler (RiPRModelHandlerPtr modelHandler) { modelHandler_ = modelHandler; }

    // Set the scene (must be called before using scene operations)
    void setScene (optixu::Scene scene) { scene_ = scene; }

    // Get the scene
    optixu::Scene getScene() { return scene_; }

    // Creates an instance from a renderable node in the OptiX scene
    void createInstance (RenderableWeakRef& weakNode);

    // Creates a geometry instance specialized for geometry processing
    void createGeometryInstance (RenderableWeakRef& weakNode);

    // Creates a physics phantom instance for collision detection
    void createPhysicsPhantom (RenderableWeakRef& weakNode);

    // Creates multiple instances from a list of renderable nodes
    void createInstanceList (const WeakRenderableList& weakNodeList);

    // Updates motion data for animated objects
    // Returns true if any motion updates were performed
    bool updateMotion();

    // Toggle selection material on a specific node
    void toggleSelectionMaterial (RenderableNode& node);

    // Select all nodes in the scene
    void selectAll();

    // Deselect all nodes in the scene
    void deselectAll();

    // Get traversable handle for the scene - used for ray traversal (inline delegation for zero overhead)
    OptixTraversableHandle getHandle() { return instanceHandler_ ? instanceHandler_->getTraversableHandle() : 0; }

    // Rebuilds the entire scene (resizes SBT and rebuilds IAS)
    void rebuild();

    // Finalizes scene setup before rendering
    void finalize();

    // Get instance count (inline delegation for zero overhead)
    size_t getInstanceCount() const { return instanceHandler_ ? instanceHandler_->getInstanceCount() : 0; }
    
    // Acceleration structure scratch memory access (now uses shared buffer from RenderContext)
    cudau::Buffer& getASBuildScratchMem() { return ctx->getASScratchBuffer(); }

    // Initialize Scene Dependent Shader Binding Table (SBT)
    // void initializeSceneDependentSBT (EntryPointType type);  // TODO: Implement with new entry point system

    // Retrieves a renderable node by its instance index
    // Returns an empty node if not found
    RenderableWeakRef getNode (uint32_t instanceIndex)
    {
        auto it = nodeMap.find (instanceIndex);
        if (it != nodeMap.end())
        {
            return it->second;
        }
        else
        {
            RenderableNode node = nullptr;
            return node;
        }
    }

    // Remove multiple nodes by ID in an atomic operation
    void removeNodesByIDs (const std::vector<BodyID>& bodyIDs);

    // Get instance data buffer by index (0 or 1)
    cudau::TypedBuffer<shared::InstanceData>* getInstanceDataBuffer (int index)
    {
        return (index >= 0 && index < 2) ? &instanceDataBuffer_[index] : nullptr;
    }

    uint32_t removeExpiredNodes()
    {
        // Step 1: Find all expired nodes and their indices
        std::vector<uint32_t> expiredIndices;
        for (const auto& [index, weakRef] : nodeMap)
        {
            if (weakRef.expired())
            {
                expiredIndices.push_back (index);
            }
        }

        if (expiredIndices.empty())
            return 0;

        // Step 2: Sort indices in descending order for stable removal
        std::sort (expiredIndices.rbegin(), expiredIndices.rend());

        // Step 3: Remove from SceneHandler in descending order
        for (uint32_t index : expiredIndices)
        {
            if (instanceHandler_ && index < instanceHandler_->getInstanceCount())
            {
                instanceHandler_->removeInstanceAt(index);
            }
        }

        // Step 4: Rebuild nodeMap with correct indices in a single pass
        NodeMap newNodeMap;
        uint32_t newIndex = 0;

        // Go through all existing indices in order
        for (const auto& [oldIndex, weakRef] : nodeMap)
        {
            // Skip if this was an expired index
            if (std::find (expiredIndices.begin(), expiredIndices.end(), oldIndex) == expiredIndices.end())
            {
                if (!weakRef.expired())
                {
                    newNodeMap[newIndex++] = weakRef;
                }
            }
        }

        nodeMap = std::move (newNodeMap);

        // Step 5: Rebuild IAS and update SBT
        LOG (DBUG) << "Removed " << expiredIndices.size() << " expired nodes from the OptiX scene";
        rebuild();

        return expiredIndices.size();
    }

    // Area light support methods
    void updateEmissiveInstances();
    void buildLightInstanceDistribution();
    const LightDistribution& getLightInstDistribution() const { return lightInstDistribution_; }
    uint32_t getNumEmissiveInstances() const { return static_cast<uint32_t> (emissiveInstances_.size()); }

 private:
    // Reference to the render context for OptiX operations
    RenderContextPtr ctx = nullptr;

    // Generic scene handler for IAS management
    InstanceHandlerPtr instanceHandler_;

    // Model handler for managing RiPR models
    RiPRModelHandlerPtr modelHandler_;

    // Scene reference (not owned)
    optixu::Scene scene_;

    // Maps instance indices to renderable nodes
    NodeMap nodeMap;

    // Note: Scratch buffer is now shared and managed by RenderContext

    // Instance data buffers (double buffered for async updates)
    cudau::TypedBuffer<shared::InstanceData> instanceDataBuffer_[2];

    // Maximum number of instances supported
    static constexpr uint32_t maxNumInstances = 16384;

    // Initialize the scene structures
    void init();

    // Helper method to convert RenderableNode to optixu::Instance
    optixu::Instance convertNodeToInstance(const RenderableWeakRef& weakNode);

    // Resize Scene Dependent Shader Binding Table (SBT)
    void resizeSceneDependentSBT();

    // CUDA module for deformation calculations
    CUmodule moduleDeform = nullptr;

    // Kernel for vertex deformation
    cudau::Kernel kernelDeform;
    cudau::Kernel kernelResetDeform;

    // Kernel for accumulating vertex normals during deformation
    cudau::Kernel kernelAccumulateVertexNormals;

    // Kernel for normalizing vertex normals after deformation
    cudau::Kernel kernelNormalizeVertexNormals;

    // Buffer for displacement data
    cudau::Buffer displacementBuffer;

    // Updates a deformed node's geometry
    void updateDeformedNode (RenderableNode node);

    void undeformNode (RenderableNode node);

    // Populate instance data for a given instance index
    void populateInstanceData (uint32_t instanceIndex, const RenderableNode& node);

    // Area light tracking
    std::vector<uint32_t> emissiveInstances_; // Track emissive instance indices
    LightDistribution lightInstDistribution_; // Distribution for sampling light instances (host side)
    bool lightDistributionDirty_ = true;      // Flag for rebuild
};