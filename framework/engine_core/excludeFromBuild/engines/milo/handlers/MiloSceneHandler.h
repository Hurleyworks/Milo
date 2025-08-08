#pragma once

// MiloSceneHandler.h
// Manages the OptiX scene graph including instances, acceleration structures,
// selection state, and traversal for ray tracing

#include "../../../RenderContext.h"

using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::WeakRenderableList;

// Forward declaration
using MiloSceneHandlerPtr = std::shared_ptr<class MiloSceneHandler>;

// Forward declarations
class MiloModelHandler;
using MiloModelHandlerPtr = std::shared_ptr<MiloModelHandler>;

class MiloSceneHandler
{
 public:
    // Factory function for creating MiloSceneHandler objects
    static MiloSceneHandlerPtr create (RenderContextPtr ctx) { return std::make_shared<MiloSceneHandler> (ctx); }

    // Map type for tracking scene nodes by instance index
    using NodeMap = std::unordered_map<uint32_t, RenderableWeakRef>;

 public:
    // Constructor initializes the scene handler with a render context
    MiloSceneHandler (RenderContextPtr ctx);

    // Destructor handles cleanup of scene resources
    ~MiloSceneHandler();

    // Set the model handler (must be called before using model operations)
    void setModelHandler(MiloModelHandlerPtr modelHandler) { modelHandler_ = modelHandler; }

    // Set the scene (must be called before using scene operations)
    void setScene(optixu::Scene* scene) { scene_ = scene; }
    
    // Get the scene
    optixu::Scene* getScene() { return scene_; }

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

    // Get traversable handle for the scene - used for ray traversal
    OptixTraversableHandle getHandle() { return travHandle; }

    // Completely rebuilds the Instance Acceleration Structure
    void rebuildIAS();

    // Updates the Instance Acceleration Structure without full rebuild
    void updateIAS();

    // Rebuilds the entire scene
    void rebuild();

    // Finalizes scene setup before rendering
    void finalize();

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
    cudau::TypedBuffer<shared::InstanceData>* getInstanceDataBuffer(int index) 
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
                expiredIndices.push_back(index);
            }
        }

        if (expiredIndices.empty())
            return 0;

        // Step 2: Sort indices in descending order for stable removal
        std::sort(expiredIndices.rbegin(), expiredIndices.rend());

        // Step 3: Remove from IAS in descending order
        for (uint32_t index : expiredIndices)
        {
            if (index < ias.getNumChildren())
            {
                ias.removeChildAt(index);
            }
        }

        // Step 4: Rebuild nodeMap with correct indices in a single pass
        NodeMap newNodeMap;
        uint32_t newIndex = 0;
        
        // Go through all existing indices in order
        for (const auto& [oldIndex, weakRef] : nodeMap)
        {
            // Skip if this was an expired index
            if (std::find(expiredIndices.begin(), expiredIndices.end(), oldIndex) == expiredIndices.end())
            {
                if (!weakRef.expired())
                {
                    newNodeMap[newIndex++] = weakRef;
                }
            }
        }
        
        nodeMap = std::move(newNodeMap);

        // Step 5: Rebuild IAS and update SBT
        LOG (DBUG) << "Removed " << expiredIndices.size() << " expired nodes from the OptiX scene";
        prepareForBuild();
        rebuild();

        return expiredIndices.size();
    }

    // Area light support methods
    void updateEmissiveInstances();
    void buildLightInstanceDistribution();
    const LightDistribution& getLightInstDistribution() const { return lightInstDistribution_; }
    uint32_t getNumEmissiveInstances() const { return static_cast<uint32_t>(emissiveInstances_.size()); }

 private:
    // Reference to the render context for OptiX operations
    RenderContextPtr ctx = nullptr;

    // Model handler for managing Milo models
    MiloModelHandlerPtr modelHandler_;

    // Scene reference (not owned)
    optixu::Scene* scene_ = nullptr;

    // Maps instance indices to renderable nodes
    NodeMap nodeMap;

    // Instance Acceleration Structure (IAS) for ray traversal optimization
    optixu::InstanceAccelerationStructure ias;

    // Buffer for IAS memory storage
    cudau::Buffer iasMem;

    // Typed buffer for OptixInstance data
    cudau::TypedBuffer<OptixInstance> instanceBuffer;

    // Instance data buffers (double buffered for async updates)
    cudau::TypedBuffer<shared::InstanceData> instanceDataBuffer_[2];
    
    // Maximum number of instances supported
    static constexpr uint32_t maxNumInstances = 16384;

    // Traversable handle for the scene - entry point for ray traversal
    OptixTraversableHandle travHandle = 0;

    // Initialize the scene structures
    void init();

    // Prepare Instance Acceleration Structure (IAS) for build
    void prepareForBuild();

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
    void populateInstanceData(uint32_t instanceIndex, const RenderableNode& node);

    // Area light tracking
    std::vector<uint32_t> emissiveInstances_;      // Track emissive instance indices
    LightDistribution lightInstDistribution_;       // Distribution for sampling light instances (host side)
    bool lightDistributionDirty_ = true;            // Flag for rebuild
};