#include "SceneHandler.h"
#include "../RenderContext.h"
#include "Handlers.h"  // For accessing ModelHandler through Handlers struct

namespace dog
{

SceneHandler::SceneHandler(RenderContextPtr ctx)
    : ctx_(ctx)
{
}

SceneHandler::~SceneHandler()
{
    finalize();
}

bool SceneHandler::initialize()
{
    if (initialized_)
    {
        LOG(WARNING) << "SceneHandler already initialized";
        return true;
    }

    if (!ctx_ || !ctx_->getCudaContext())
    {
        LOG(WARNING) << "SceneHandler: Invalid render context";
        return false;
    }

    try
    {
        // Initialize slot finders for resource allocation tracking
        // Note: material_slot_finder is now managed by ModelHandler
        geom_inst_slot_finder_.initialize(maxNumGeometryInstances);
        inst_slot_finder_.initialize(maxNumInstances);
        
        LOG(DBUG) << "SceneHandler slot finders initialized:";
        LOG(DBUG) << "  Max geometry instances: " << maxNumGeometryInstances;
        LOG(DBUG) << "  Max instances: " << maxNumInstances;
        
        // Initialize data buffers on device
        CUcontext cuContext = ctx_->getCudaContext();
        // Note: material_data_buffer is now managed by ModelHandler
        geom_inst_data_buffer_.initialize(cuContext, cudau::BufferType::Device, maxNumGeometryInstances);
        inst_data_buffer_[0].initialize(cuContext, cudau::BufferType::Device, maxNumInstances);
        inst_data_buffer_[1].initialize(cuContext, cudau::BufferType::Device, maxNumInstances);
        
        LOG(DBUG) << "SceneHandler data buffers initialized";
        LOG(DBUG) << "  GeomInst buffer size: " << maxNumGeometryInstances;
        LOG(DBUG) << "  Instance buffers size: " << maxNumInstances << " (double buffered)";
        
        // Create Instance Acceleration Structure using the RenderContext's scene
        optixu::Scene optixScene = ctx_->getScene();
        ias_ = optixScene.createInstanceAccelerationStructure();
        
        // Configure IAS for fast build (since we're interactive)
        ias_.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::Yes,
            optixu::AllowCompaction::No);
        
        LOG(DBUG) << "SceneHandler IAS created and configured";
        
        // Initialize light distribution for importance sampling
        // Using probability buffer approach (not texture)
        light_inst_dist_.initialize(cuContext, cudau::BufferType::Device, nullptr, maxNumInstances);
        LOG(DBUG) << "Light distribution initialized for " << maxNumInstances << " instances";
        
        // In an empty scene, traversable handle can be 0
        // This is a valid state for OptiX
        traversable_handle_ = 0;
        has_geometry_ = false;

        // Initialize scene-dependent SBTs for empty scene
        updateSceneDependentSBTs();

        initialized_ = true;
        LOG(INFO) << "SceneHandler initialized successfully";
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to initialize scene handler: " << ex.what();
        finalize();
        return false;
    }
}

void SceneHandler::finalize()
{
    if (!initialized_)
    {
        return;
    }

    // Clean up IAS and associated memory
    if (as_scratch_mem_.isInitialized())
    {
        as_scratch_mem_.finalize();
    }
    if (ias_instance_buffer_.isInitialized())
    {
        ias_instance_buffer_.finalize();
    }
    if (ias_mem_.isInitialized())
    {
        ias_mem_.finalize();
    }
    
    // Destroy IAS
    if (ias_)
    {
        ias_.destroy();
    }
    
    // Clean up data buffers
    inst_data_buffer_[1].finalize();
    inst_data_buffer_[0].finalize();
    geom_inst_data_buffer_.finalize();
    // Note: material_data_buffer is now managed by ModelHandler
    
    // Clean up light distribution
    light_inst_dist_.finalize();
    
    // Clean up slot finders
    inst_slot_finder_.finalize();
    geom_inst_slot_finder_.finalize();
    // Note: material_slot_finder is now managed by ModelHandler

    traversable_handle_ = 0;
    has_geometry_ = false;
    initialized_ = false;

    LOG(DBUG) << "SceneHandler finalized";
}

bool SceneHandler::buildAccelerationStructures()
{
    if (!initialized_)
    {
        LOG(WARNING) << "SceneHandler not initialized";
        return false;
    }

    try
    {
        // Check if we have any instances
        uint32_t numInstances = ias_.getNumChildren();
        
        if (numInstances == 0)
        {
            // Empty scene is valid - traversable handle stays 0
            traversable_handle_ = 0;
            has_geometry_ = false;
            LOG(DBUG) << "Empty scene - traversable handle = 0";
            return true;
        }
        
        LOG(INFO) << "Building IAS with " << numInstances << " instances";
        
        // Get build requirements
        OptixAccelBufferSizes bufferSizes;
        ias_.prepareForBuild(&bufferSizes);
        
        // Allocate or resize scratch memory
        if (as_scratch_mem_.isInitialized())
        {
            if (bufferSizes.tempSizeInBytes > as_scratch_mem_.sizeInBytes())
            {
                as_scratch_mem_.resize(bufferSizes.tempSizeInBytes, 1);
            }
        }
        else
        {
            as_scratch_mem_.initialize(ctx_->getCudaContext(), cudau::BufferType::Device,
                                      bufferSizes.tempSizeInBytes, 1);
        }
        
        // Allocate or resize IAS memory
        if (ias_mem_.isInitialized())
        {
            if (bufferSizes.outputSizeInBytes > ias_mem_.sizeInBytes())
            {
                ias_mem_.resize(bufferSizes.outputSizeInBytes, 1);
            }
        }
        else
        {
            ias_mem_.initialize(ctx_->getCudaContext(), cudau::BufferType::Device, 
                               bufferSizes.outputSizeInBytes, 1);
        }
        
        // Allocate or resize instance buffer
        if (ias_instance_buffer_.isInitialized())
        {
            if (numInstances > ias_instance_buffer_.numElements())
            {
                ias_instance_buffer_.resize(numInstances);
            }
        }
        else
        {
            ias_instance_buffer_.initialize(ctx_->getCudaContext(), cudau::BufferType::Device, numInstances);
        }
        
        // Build the IAS
        CUstream stream = ctx_->getCudaStream();
        ias_.rebuild(stream, ias_instance_buffer_, ias_mem_, as_scratch_mem_);
        
        // Get traversable handle
        traversable_handle_ = ias_.getHandle();
        has_geometry_ = true;
        ias_needs_rebuild_ = false;
        
        LOG(INFO) << "IAS built successfully with " << numInstances << " instances";
        LOG(DBUG) << "Traversable handle: " << traversable_handle_;
        
        // Update scene-dependent SBTs after building acceleration structures
        updateSceneDependentSBTs();
        
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to build acceleration structures: " << ex.what();
        return false;
    }
}

const cudau::TypedBuffer<shared::DisneyData>& SceneHandler::getMaterialDataBuffer() const
{
    // Get material buffer from ModelHandler
    auto handlers = ctx_->getHandlers();
    if (!handlers || !handlers->model)
    {
        static cudau::TypedBuffer<shared::DisneyData> emptyBuffer;
        LOG(WARNING) << "ModelHandler not available for material buffer access";
        return emptyBuffer;
    }
    return handlers->model->getMaterialDataBuffer();
}

void SceneHandler::update()
{
    if (!initialized_)
    {
        LOG(WARNING) << "SceneHandler not initialized";
        return;
    }

    // Future implementation will handle:
    // - Transform updates
    // - Animation updates
    // - Dynamic object updates
    // - Visibility updates
    
    LOG(DBUG) << "Scene updated";
}

void SceneHandler::updateSceneDependentSBTs()
{
    if (!initialized_)
    {
        LOG(WARNING) << "SceneHandler not initialized - cannot update SBTs";
        return;
    }

    if (!ctx_)
    {
        LOG(WARNING) << "RenderContext not available - cannot update SBTs";
        return;
    }

    // Get the OptiX scene
    optixu::Scene optixScene = ctx_->getScene();
    
    // Generate shader binding table layout for the scene
    // This marks the scene as "ready" even for empty scenes
    size_t hitGroupSbtSize;
    optixScene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    
    LOG(DBUG) << "Generated SBT layout, size: " << hitGroupSbtSize;
    
    // Get handlers from render context
    auto handlers = ctx_->getHandlers();
    if (!handlers || !handlers->pipeline)
    {
        LOG(WARNING) << "Pipeline handler not available for SBT update";
        return;
    }
    
    // Update hit group SBTs for both pipelines if needed
    if (hitGroupSbtSize > 0)
    {
        // Check if G-buffer pipeline SBT needs initialization or resize
        if (!handlers->pipeline->hasGBufferHitGroupSbt() || 
            handlers->pipeline->getGBufferHitGroupSbtSize() < hitGroupSbtSize)
        {
            handlers->pipeline->initializeGBufferHitGroupSbt(hitGroupSbtSize);
            handlers->pipeline->setGBufferScene(optixScene);
            LOG(DBUG) << "Updated G-buffer pipeline SBT";
        }
        
        // Check if path tracing pipeline SBT needs initialization or resize
        if (!handlers->pipeline->hasPathTracingHitGroupSbt() || 
            handlers->pipeline->getPathTracingHitGroupSbtSize() < hitGroupSbtSize)
        {
            handlers->pipeline->initializePathTracingHitGroupSbt(hitGroupSbtSize);
            handlers->pipeline->setPathTracingScene(optixScene);
            LOG(DBUG) << "Updated path tracing pipeline SBT";
        }
        
        LOG(INFO) << "Scene-dependent SBTs updated for both pipelines";
    }
    else
    {
        LOG(DBUG) << "Empty scene - SBT layout generated but no hit groups needed";
    }
}

bool SceneHandler::addRenderableNode(RenderableWeakRef weakNode)
{
    if (!initialized_)
    {
        bool success = initialize();
        if (!success) return false;
    }

    // Try to lock the weak reference
    RenderableNode node = weakNode.lock();
    if (!node)
    {
        LOG(WARNING) << "Cannot add node - weak reference is expired";
        return false;
    }

    // Get the node's unique ID
    ItemID nodeID = node->getID();
    if (nodeID == INVALID_ID)
    {
        LOG(WARNING) << "Cannot add node - invalid ID";
        return false;
    }

    // Check if node already exists
    if (node_resources_.find(nodeID) != node_resources_.end())
    {
        LOG(DBUG) << "Node " << nodeID << " already exists in scene";
        return true; // Not an error, just already present
    }

    // Get the CgModel from the node
    CgModelPtr cgModel = node->getModel();
    if (!cgModel || !cgModel->isValid())
    {
        LOG(WARNING) << "Cannot add node " << nodeID << " - no valid CgModel";
        return false;
    }

    // Check if any surface has emissive material
    bool hasEmissiveMaterial = false;
    std::vector<uint32_t> emissiveSurfaceIndices;
    
    for (size_t i = 0; i < cgModel->S.size(); ++i)
    {
        const auto& surface = cgModel->S[i];
        const auto& material = surface.cgMaterial;
        
        // Check if material has emission properties
        if (material.emission.luminous > 0.0f)
        {
            hasEmissiveMaterial = true;
            emissiveSurfaceIndices.push_back(static_cast<uint32_t>(i));
            LOG(DBUG) << "Found emissive surface " << i << " with luminous intensity: " 
                      << material.emission.luminous;
        }
    }

    // Create resource tracking for this node
    NodeResources resources;
    resources.node = weakNode;
    resources.is_emissive = hasEmissiveMaterial;
    
    // Allocate instance slot
    resources.instance_slot = inst_slot_finder_.getFirstAvailableSlot();
    if (resources.instance_slot >= maxNumInstances)
    {
        LOG(WARNING) << "Cannot add node " << nodeID << " - instance slots full";
        return false;
    }
    inst_slot_finder_.setInUse(resources.instance_slot);

    // Get ModelHandler from RenderContext
    auto handlers = ctx_->getHandlers();
    if (!handlers || !handlers->model)
    {
        LOG(WARNING) << "ModelHandler not available";
        inst_slot_finder_.setNotInUse(resources.instance_slot);
        return false;
    }
    ModelHandler* modelHandler = handlers->model.get();
    
    // Compute hash of the geometry for caching
    resources.geometry_hash = ModelHandler::computeGeometryHash(cgModel);
    
    // Check if ModelHandler already has this geometry cached
    GeometryGroupResources* geomGroup = modelHandler->getGeometry(resources.geometry_hash);
    
    if (geomGroup)
    {
        // Reuse existing geometry group
        modelHandler->incrementRefCount(resources.geometry_hash);
        LOG(DBUG) << "Reusing cached geometry group (hash: " << resources.geometry_hash << ", refs: " << geomGroup->ref_count << ")";
    }
    else
    {
        // Create new geometry group using ModelHandler
        GeometryGroupResources newGeomGroup;
        if (!modelHandler->createGeometryGroup(cgModel, resources.geometry_hash, newGeomGroup))
        {
            LOG(WARNING) << "Failed to create geometry group for node " << nodeID;
            inst_slot_finder_.setNotInUse(resources.instance_slot);
            return false;
        }
        
        modelHandler->addGeometry(resources.geometry_hash, std::move(newGeomGroup));
        geomGroup = modelHandler->getGeometry(resources.geometry_hash);
        LOG(INFO) << "Created new geometry group (hash: " << resources.geometry_hash << ")";
    }
    
    // Allocate geometry instance slot
    resources.geom_inst_slot = geom_inst_slot_finder_.getFirstAvailableSlot();
    if (resources.geom_inst_slot >= maxNumGeometryInstances)
    {
        LOG(WARNING) << "Cannot add node " << nodeID << " - geometry instance slots full";
        inst_slot_finder_.setNotInUse(resources.instance_slot);
        modelHandler->decrementRefCount(resources.geometry_hash);
        return false;
    }
    geom_inst_slot_finder_.setInUse(resources.geom_inst_slot);
    
    // Create OptiX instance for this node
    if (!createNodeInstance(resources, *geomGroup))
    {
        LOG(WARNING) << "Failed to create OptiX instance for node " << nodeID;
        inst_slot_finder_.setNotInUse(resources.instance_slot);
        geom_inst_slot_finder_.setNotInUse(resources.geom_inst_slot);
        modelHandler->decrementRefCount(resources.geometry_hash);
        return false;
    }
    
    // Mark node as stored in scene handler
    node->getState().state |= sabi::PRenderableState::StoredInSceneHandler;
    
    // Store the resources
    node_resources_[nodeID] = resources;
    
    // Mark that we need to rebuild acceleration structures
    ias_needs_rebuild_ = true;
    has_geometry_ = true;

    LOG(INFO) << "Added RenderableNode " << nodeID << " to scene (slot " << resources.instance_slot 
              << ", emissive: " << (resources.is_emissive ? "yes" : "no") << ")";
    LOG(DBUG) << "Scene now contains " << node_resources_.size() << " nodes";
    
    return true;
}

bool SceneHandler::removeRenderableNode(RenderableWeakRef weakNode)
{
    if (!initialized_)
    {
        LOG(WARNING) << "SceneHandler not initialized";
        return false;
    }

    // Try to lock the weak reference
    RenderableNode node = weakNode.lock();
    if (!node)
    {
        // Node already deleted, but we might still have resources to clean up
        // This would require tracking by weak_ptr, which we'll add later
        LOG(DBUG) << "Cannot remove node - weak reference is expired";
        return false;
    }

    ItemID nodeID = node->getID();
    return removeRenderableNodeByID(nodeID);
}

bool SceneHandler::removeRenderableNodeByID(ItemID nodeID)
{
    if (!initialized_)
    {
        LOG(WARNING) << "SceneHandler not initialized";
        return false;
    }

    auto it = node_resources_.find(nodeID);
    if (it == node_resources_.end())
    {
        LOG(DBUG) << "Node " << nodeID << " not found in scene";
        return false;
    }

    // Get the resources
    const NodeResources& resources = it->second;
    
    // Free the instance slot
    if (resources.instance_slot != UINT32_MAX)
    {
        inst_slot_finder_.setNotInUse(resources.instance_slot);
    }
    
    // Free the geometry instance slot if allocated
    if (resources.geom_inst_slot != UINT32_MAX)
    {
        geom_inst_slot_finder_.setNotInUse(resources.geom_inst_slot);
    }
    
    // Decrement geometry reference count in ModelHandler
    if (resources.geometry_hash != 0)
    {
        auto handlers = ctx_->getHandlers();
        if (handlers && handlers->model)
        {
            handlers->model->decrementRefCount(resources.geometry_hash);
        }
    }
    
    // Clear the stored flag if node still exists
    if (RenderableNode node = resources.node.lock())
    {
        node->getState().state &= ~sabi::PRenderableState::StoredInSceneHandler;
    }
    
    // Remove from tracking
    node_resources_.erase(it);
    
    // Mark that we need to rebuild acceleration structures
    ias_needs_rebuild_ = true;
    
    // Check if scene is now empty
    if (node_resources_.empty())
    {
        has_geometry_ = false;
    }

    LOG(INFO) << "Removed RenderableNode " << nodeID << " from scene";
    LOG(DBUG) << "Scene now contains " << node_resources_.size() << " nodes";
    
    return true;
}

// Note: computeGeometryHash and createGeometryGroup methods have been moved to ModelHandler

bool SceneHandler::createNodeInstance(NodeResources& nodeRes, const GeometryGroupResources& geomGroup)
{
    try
    {
        // Get node's transform
        RenderableNode node = nodeRes.node.lock();
        if (!node)
        {
            LOG(WARNING) << "Node expired while creating instance";
            return false;
        }
        
        // Get transform from SpaceTime
        const sabi::SpaceTime& spacetime = node->getSpaceTime();
        const Eigen::Matrix4f& worldTransform = spacetime.worldTransform.matrix();
        
        // Convert Eigen matrix to OptiX format (row-major float[12])
        float transform[12];
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                transform[row * 4 + col] = worldTransform(row, col);
            }
        }
        
        // Create OptiX instance pointing to the GeometryGroup's GAS
        OptixInstance optixInstance = {};
        std::memcpy(optixInstance.transform, transform, sizeof(transform));
        optixInstance.instanceId = nodeRes.instance_slot;
        optixInstance.sbtOffset = 0;  // Will be set when we have multiple materials
        optixInstance.visibilityMask = 0xFF;  // Visible to all ray types
        optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        optixInstance.traversableHandle = geomGroup.gas.getHandle();
        
        // Add instance to IAS
        if (!ias_instance_buffer_.isInitialized() || 
            ias_instance_buffer_.numElements() <= ias_.getNumChildren())
        {
            // Resize instance buffer
            uint32_t newSize = std::max(16u, (uint32_t)(ias_.getNumChildren() * 2));
            if (ias_instance_buffer_.isInitialized())
            {
                ias_instance_buffer_.resize(newSize);
            }
            else
            {
                ias_instance_buffer_.initialize(ctx_->getCudaContext(), cudau::BufferType::Device, newSize);
            }
        }
        
        // Store instance in buffer
        nodeRes.optix_instance_index = ias_.getNumChildren();
        
        // Create optixu::Instance from the GAS handle and transform
        optixu::Scene optixScene = ctx_->getScene();
        optixu::Instance instance = optixScene.createInstance();
        instance.setChild(geomGroup.gas);
        instance.setTransform(transform);
        
        // Add to IAS
        ias_.addChild(instance);
        
        // Update geometry instance data buffer for each surface in the geometry group
        // For now, we'll use the first surface's buffers as representative
        // In the future, we may need to handle multiple geom instances per node
        if (!geomGroup.geom_instances.empty())
        {
            const auto& firstGeomInst = geomGroup.geom_instances[0];
            
            // Map the buffer to host memory
            geom_inst_data_buffer_.map();
            shared::GeometryInstanceData* geomInstDataHost = geom_inst_data_buffer_.getMappedPointer();
            shared::GeometryInstanceData& geomInstData = geomInstDataHost[nodeRes.geom_inst_slot];
            
            geomInstData.vertexBuffer = geomGroup.vertex_buffer.getROBuffer<shared::enableBufferOobCheck>();
            geomInstData.triangleBuffer = firstGeomInst.triangle_buffer.getROBuffer<shared::enableBufferOobCheck>();
            geomInstData.materialSlot = firstGeomInst.material_slot;
            geomInstData.geomInstSlot = nodeRes.geom_inst_slot;
            
            // Initialize light distribution for emissive geometry instances
            if (nodeRes.is_emissive)
            {
                // For now, we'll set up a uniform distribution for all triangles
                // In the future, this could be importance-sampled based on area and emittance
                uint32_t numTriangles = firstGeomInst.triangle_buffer.numElements();
                if (numTriangles > 0)
                {
                    // Initialize uniform distribution for triangles in this geometry instance
                    // The actual distribution setup would need CUDA context and proper initialization
                    // For now, we just mark it as having emissive primitives
                    LOG(DBUG) << "Geometry instance " << nodeRes.geom_inst_slot 
                              << " has " << numTriangles << " emissive triangles";
                }
            }
            
            // Unmap the buffer
            geom_inst_data_buffer_.unmap();
        }
        
        // Update instance data buffer
        inst_data_buffer_[0].map();
        shared::InstanceData* instDataHost = inst_data_buffer_[0].getMappedPointer();
        shared::InstanceData& instData = instDataHost[nodeRes.instance_slot];
        
        instData.transform = Matrix4x4(
            Vector4D(transform[0], transform[1], transform[2], transform[3]),
            Vector4D(transform[4], transform[5], transform[6], transform[7]),
            Vector4D(transform[8], transform[9], transform[10], transform[11]),
            Vector4D(0, 0, 0, 1));
        instData.curToPrevTransform = instData.transform;  // No motion blur yet
        instData.normalMatrix = instData.transform.getUpperLeftMatrix().invert().transpose();
        instData.uniformScale = 1.0f;  // Assuming uniform scale for now
        
        // For now, single geometry instance per instance
        // In future, could support multiple geom instances per instance
        instData.isEmissive = nodeRes.is_emissive ? 1 : 0;
        instData.emissiveScale = 1.0f;
        
        // Set up light distribution for emissive instances
        if (nodeRes.is_emissive)
        {
            // Store geometry instance slots that have emissive surfaces
            // For now, we assume a single geometry instance per instance node
            // The actual distribution initialization would need proper CUDA setup
            // This marks the instance as a light source for sampling
            LOG(DBUG) << "Instance " << nodeRes.instance_slot << " marked as emissive light source";
            
            // In a complete implementation, we would:
            // 1. Create a buffer of geomInstSlots for this instance
            // 2. Initialize instData.lightGeomInstDist with proper weights
            // 3. Set up importance sampling based on surface area and emittance
        }
        
        inst_data_buffer_[0].unmap();
        
        LOG(DBUG) << "Created OptiX instance " << nodeRes.optix_instance_index << " for node (transform: " 
                  << transform[3] << ", " << transform[7] << ", " << transform[11] << ")";
        
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to create node instance: " << ex.what();
        return false;
    }
}

} // namespace dog