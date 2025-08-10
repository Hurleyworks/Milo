// ShockerSceneHandler.cpp
// Implementation of scene management for the Shocker rendering system

#include "ShockerSceneHandler.h"
#include "ShockerModelHandler.h"
#include "ShockerMaterialHandler.h"
#include "AreaLightHandler.h"
#include "../models/ShockerModel.h"
#include "../models/ShockerCore.h"

ShockerSceneHandler::ShockerSceneHandler(RenderContextPtr ctx)
    : ctx_(ctx)
{
    // Constructor - no logging needed
}

ShockerSceneHandler::~ShockerSceneHandler()
{
    // Clean up OptiX resources
    if (ctx_ && ctx_->getCudaStream()) {
        try {
            CUDADRV_CHECK(cuStreamSynchronize(ctx_->getCudaStream()));
        } catch (...) {
            // Ignore errors during destruction
        }
    }
    
    if (ias_) {
        ias_.destroy();
    }
    
    if (iasMem_.isInitialized()) {
        iasMem_.finalize();
    }
    
    if (instanceBuffer_.isInitialized()) {
        instanceBuffer_.finalize();
    }
    
    // Note: scene_ is not owned by this handler, don't destroy it
    scene_ = nullptr;
    
    clear();
}

void ShockerSceneHandler::initialize()
{
    if (isInitialized_) {
        LOG(WARNING) << "ShockerSceneHandler already initialized";
        return;
    }

    // Initialize instance slot finder
    instanceSlotFinder_.initialize(MaxNumInstances);
    
    // Reserve space for nodes
    nodes_.reserve(1000); // Start with reasonable capacity
    
    isInitialized_ = true;
    LOG(INFO) << "ShockerSceneHandler initialized with capacity for " << MaxNumInstances << " nodes";
}

shocker::ShockerNode* ShockerSceneHandler::createShockerNode(RenderableWeakRef& weakNode)
{
    if (!isInitialized_) {
        LOG(WARNING) << "ShockerSceneHandler not initialized";
        return nullptr;
    }

    if (!modelHandler_) {
        LOG(WARNING) << "Model handler not set";
        return nullptr;
    }

    // Get the node
    RenderableNode node = weakNode.lock();
    if (!node) {
        LOG(WARNING) << "Failed to lock weak node reference";
        return nullptr;
    }

    // Process the node through model handler
    // Note: modelHandler_->processRenderableNode already handles material processing
    ShockerModelPtr model = modelHandler_->processRenderableNode(node);
    if (!model) {
        LOG(WARNING) << "Failed to create model for node: " << node->getName();
        return nullptr;
    }

    // Create ShockerNode from the model
    shocker::ShockerNode* shockerNode = modelHandler_->createShockerNode(model.get(), node->getSpaceTime());
    if (!shockerNode) {
        LOG(WARNING) << "Failed to create ShockerNode for node: " << node->getName();
        return nullptr;
    }

    // Store node
    nodes_.push_back(shockerNode);
    
    // Map node to renderable node
    nodeMap_[shockerNode->instSlot] = weakNode;
    
    // Note: AreaLightHandler notifications will be handled separately
    
    // Successfully created node (no logging needed for routine operations)
    
    return shockerNode;
}

void ShockerSceneHandler::createNodeList(const WeakRenderableList& weakNodeList)
{
    LOG(INFO) << "Creating nodes for " << weakNodeList.size() << " renderable nodes";
    
    size_t successCount = 0;
    size_t failCount = 0;
    
    for (const auto& weakNode : weakNodeList) {
        // Create a copy to pass as non-const reference
        RenderableWeakRef weakNodeCopy = weakNode;
        shocker::ShockerNode* shockerNode = createShockerNode(weakNodeCopy);
        if (shockerNode) {
            successCount++;
        } else {
            failCount++;
        }
    }
    
    LOG(INFO) << "Created " << successCount << " nodes successfully";
    if (failCount > 0) {
        LOG(WARNING) << "Failed to create " << failCount << " nodes";
    }
}

void ShockerSceneHandler::processRenderableNode(RenderableNode& node)
{
    if (!isInitialized_) {
        initialize();
    }

    // Create weak reference
    RenderableWeakRef weakNode = node;
    
    // Create ShockerNode
    createShockerNode(weakNode);
}

void ShockerSceneHandler::clear()
{
    // Note: AreaLightHandler notifications will be handled separately
    
    // Clear nodes (they're owned by model handler)
    nodes_.clear();
    
    // Clear node map
    nodeMap_.clear();
    
    // Reset slot finder
    instanceSlotFinder_.reset();
    
    // Clear handlers if needed
    if (modelHandler_) {
        modelHandler_->clear();
    }
    
    if (materialHandler_) {
        materialHandler_->clear();
    }
}

shocker::ShockerNode* ShockerSceneHandler::getShockerNode(uint32_t index) const
{
    if (index >= nodes_.size()) {
        return nullptr;
    }
    return nodes_[index];
}

RenderableWeakRef ShockerSceneHandler::getRenderableNode(uint32_t nodeIndex) const
{
    auto it = nodeMap_.find(nodeIndex);
    if (it != nodeMap_.end()) {
        return it->second;
    }
    return RenderableWeakRef();
}

void ShockerSceneHandler::buildAccelerationStructures()
{
    if (!modelHandler_) {
        LOG(WARNING) << "Cannot build acceleration structures: model handler not set";
        return;
    }

    LOG(INFO) << "Building acceleration structures for " << nodes_.size() << " nodes";
    
    if (!ctx_) {
        LOG(WARNING) << "No render context available - marking structures as built without GPU resources";
        
        // Even without a render context, we should mark the structures as "built"
        // to maintain consistency in the state management
        for (const auto& [name, model] : modelHandler_->getAllModels()) {
            shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
            if (surfaceGroup && surfaceGroup->needsRebuild) {
                surfaceGroup->needsRebuild = 0;
                LOG(DBUG) << "Marked surface group as built (no GPU) for model: " << name;
            }
        }
        return;
    }
    
    // Build GAS for each surface group that needs rebuilding
    CUcontext cudaContext = ctx_->getCudaContext();
    CUstream cudaStream = ctx_->getCudaStream();
    optixu::Context optixContext = ctx_->getOptiXContext();
    
    // Build GAS for each surface group that needs rebuilding
    for (const auto& [name, model] : modelHandler_->getAllModels()) {
        shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
        if (!surfaceGroup) continue;
        
        if (surfaceGroup->needsRebuild) {
            // Check if we have geometry to build
            if (surfaceGroup->geomInsts.empty()) {
                LOG(WARNING) << "Surface group has no geometry instances for model: " << name;
                surfaceGroup->needsRebuild = 0;
                continue;
            }
            
            // Create GAS if not already created
            if (!surfaceGroup->optixGas) {
                surfaceGroup->optixGas = scene_->createGeometryAccelerationStructure();
                surfaceGroup->optixGas.setConfiguration(
                    optixu::ASTradeoff::PreferFastBuild,
                    optixu::AllowUpdate::No,
                    optixu::AllowCompaction::No);
            }
            
            // Create and add geometry instances to GAS
            for (const shocker::ShockerSurface* surfaceConst : surfaceGroup->geomInsts) {
                // We need non-const access to create the optixGeomInst
                shocker::ShockerSurface* surface = const_cast<shocker::ShockerSurface*>(surfaceConst);
                
                // Create OptiX geometry instance if needed
                if (!surface->optixGeomInst) {
                    if (const TriangleGeometry* triGeom = std::get_if<TriangleGeometry>(&surface->geometry)) {
                        if (triGeom->vertexBuffer.isInitialized() && triGeom->triangleBuffer.isInitialized()) {
                            surface->optixGeomInst = scene_->createGeometryInstance();
                            surface->optixGeomInst.setVertexBuffer(triGeom->vertexBuffer);
                            surface->optixGeomInst.setTriangleBuffer(triGeom->triangleBuffer);
                            surface->optixGeomInst.setNumMaterials(1, optixu::BufferView());
                            surface->optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
                            surface->optixGeomInst.setUserData(surface->geomInstSlot);
                            
                            // Create and set a default material for now
                            // TODO: Use actual materials from the material handler
                            optixu::Material defaultMat = ctx_->getOptiXContext().createMaterial();
                            surface->optixGeomInst.setMaterial(0, 0, defaultMat);
                        }
                    }
                }
                
                if (surface->optixGeomInst) {
                    surfaceGroup->optixGas.addChild(surface->optixGeomInst);
                }
            }
            
            // Prepare for build
            OptixAccelBufferSizes asSizes;
            surfaceGroup->optixGas.prepareForBuild(&asSizes);
            
            // Allocate memory for GAS
            if (!surfaceGroup->optixGasMem.isInitialized()) {
                surfaceGroup->optixGasMem.initialize(
                    cudaContext, cudau::BufferType::Device,
                    asSizes.outputSizeInBytes, 1);
            } else if (surfaceGroup->optixGasMem.sizeInBytes() < asSizes.outputSizeInBytes) {
                surfaceGroup->optixGasMem.resize(asSizes.outputSizeInBytes, 1);
            }
            
            // Allocate scratch memory if needed
            cudau::Buffer scratchMem;
            scratchMem.initialize(cudaContext, cudau::BufferType::Device, asSizes.tempSizeInBytes, 1);
            
            // Build the GAS
            surfaceGroup->optixGas.rebuild(cudaStream, surfaceGroup->optixGasMem, scratchMem);
            
            // Clean up scratch memory
            scratchMem.finalize();
            
            // Log the GAS handle for debugging
            OptixTraversableHandle gasHandle = surfaceGroup->optixGas.getHandle();
            LOG(DBUG) << "Built GAS for model: " << name << ", handle: " << gasHandle;
            
            surfaceGroup->needsRebuild = 0;
        }
    }
    
    // Build IAS if we have nodes and scene is set
    if (!nodes_.empty() && scene_) {
        // Generate shader binding table layout before building IAS
        // This is required to avoid "Shader binding table layout generation has not been done" error
        size_t hitGroupSbtSize;
        scene_->generateShaderBindingTableLayout(&hitGroupSbtSize);
        LOG(DBUG) << "Generated scene SBT layout, size: " << hitGroupSbtSize << " bytes";
        
        // Create IAS if not already created
        if (!ias_) {
            try {
                ias_ = scene_->createInstanceAccelerationStructure();
                ias_.setConfiguration(
                    optixu::ASTradeoff::PreferFastBuild,
                    optixu::AllowUpdate::Yes);
                LOG(DBUG) << "Created IAS for scene";
            } catch (const std::exception& e) {
                LOG(WARNING) << "Failed to create IAS: " << e.what();
                return;
            }
        }
        
        try {
            // Prepare instance buffer
            if (!instanceBuffer_.isInitialized() || instanceBuffer_.numElements() < nodes_.size()) {
                if (instanceBuffer_.isInitialized()) {
                    instanceBuffer_.finalize();
                }
                instanceBuffer_.initialize(cudaContext, cudau::BufferType::Device, nodes_.size());
            }
            
            // Create OptixInstance data for each node
            std::vector<OptixInstance> instances;
            instances.reserve(nodes_.size());
            
            for (size_t i = 0; i < nodes_.size(); ++i) {
                shocker::ShockerNode* node = nodes_[i];
                if (!node) continue;
                
                OptixInstance instance = {};
                
                // Set transform from node
                const Matrix4x4& transform = node->matM2W;
                instance.transform[0] = transform.c0.x;
                instance.transform[1] = transform.c1.x;
                instance.transform[2] = transform.c2.x;
                instance.transform[3] = transform.c3.x;
                instance.transform[4] = transform.c0.y;
                instance.transform[5] = transform.c1.y;
                instance.transform[6] = transform.c2.y;
                instance.transform[7] = transform.c3.y;
                instance.transform[8] = transform.c0.z;
                instance.transform[9] = transform.c1.z;
                instance.transform[10] = transform.c2.z;
                instance.transform[11] = transform.c3.z;
                
                // Set instance ID and other properties
                instance.instanceId = node->instSlot;
                instance.visibilityMask = 255;  // Visible to all ray types
                instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                
                // Get the traversable handle from the surface group's GAS
                OptixTraversableHandle gasHandle = 0;
                if (node->geomGroupInst.geomGroup) {
                    if (node->geomGroupInst.geomGroup->optixGas) {
                        gasHandle = node->geomGroupInst.geomGroup->optixGas.getHandle();
                        LOG(DBUG) << "Got GAS handle for node " << i << ": " << gasHandle;
                    } else {
                        LOG(WARNING) << "Node " << i << " surface group has no optixGas";
                    }
                } else {
                    LOG(WARNING) << "Node " << i << " has no geomGroup";
                }
                instance.traversableHandle = gasHandle;
                
                if (gasHandle == 0) {
                    LOG(WARNING) << "Node " << i << " has no GAS traversable handle";
                }
                
                instances.push_back(instance);
            }
            
            // Upload instances to GPU
            if (!instances.empty()) {
                instanceBuffer_.write(instances.data(), instances.size());
                
                // Mark IAS for rebuild
                ias_.markDirty();
                
                // Clear existing children and add new ones
                while (ias_.getNumChildren() > 0) {
                    ias_.removeChildAt(ias_.getNumChildren() - 1);
                }
                
                // Create instances with their GAS objects
                for (size_t i = 0; i < nodes_.size(); ++i) {
                    optixu::Instance inst = scene_->createInstance();
                    
                    // Get the GAS from the node's surface group
                    auto node = nodes_[i];
                    if (node && node->geomGroupInst.geomGroup && node->geomGroupInst.geomGroup->optixGas) {
                        // Set the child using the GeometryAccelerationStructure object
                        inst.setChild(node->geomGroupInst.geomGroup->optixGas);
                    } else {
                        LOG(WARNING) << "Instance " << i << " has no GAS object";
                    }
                    ias_.addChild(inst);
                }
                
                // Prepare for build
                OptixAccelBufferSizes bufferSizes;
                ias_.prepareForBuild(&bufferSizes);
                
                // Allocate memory if needed
                if (!iasMem_.isInitialized() || iasMem_.sizeInBytes() < bufferSizes.outputSizeInBytes) {
                    if (iasMem_.isInitialized()) {
                        iasMem_.finalize();
                    }
                    iasMem_.initialize(cudaContext, cudau::BufferType::Device, bufferSizes.outputSizeInBytes, 1);
                }
                
                // Build the IAS
                travHandle_ = ias_.rebuild(cudaStream, instanceBuffer_, iasMem_, ctx_->getASBuildScratchMem());
                
                // Synchronize to ensure build completes
                CUDADRV_CHECK(cuStreamSynchronize(cudaStream));
                
                LOG(INFO) << "Built IAS with " << instances.size() << " instances, traversable handle: " << travHandle_;
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to build IAS: " << e.what();
            travHandle_ = 0;  // 0 is valid for empty scene
        }
    } else if (nodes_.empty()) {
        // Empty scene - traversable handle should be 0
        travHandle_ = 0;
        LOG(DBUG) << "No nodes in scene, traversable handle set to 0";
    }
    
    LOG(INFO) << "Acceleration structures built";
}

void ShockerSceneHandler::updateAccelerationStructures()
{
    if (!modelHandler_) {
        LOG(WARNING) << "Cannot update acceleration structures: model handler not set";
        return;
    }

    // Updating acceleration structures
    
    // Update any refittable surface acceleration structures
    for (const auto& [name, model] : modelHandler_->getAllModels()) {
        shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
        if (surfaceGroup && surfaceGroup->refittable) {
            // TODO: Refit GAS when we have OptiX integration
            // GAS refitted for surface group
        }
    }
    
    // TODO: Update instance acceleration structure (IAS) when we have OptiX integration
    
    // Acceleration structures updated
}

size_t ShockerSceneHandler::getSurfaceCount() const
{
    if (!modelHandler_) {
        return 0;
    }
    return modelHandler_->getShockerSurfaceCount();
}

size_t ShockerSceneHandler::getMaterialCount() const
{
    if (!materialHandler_) {
        return 0;
    }
    return materialHandler_->getAllMaterials().size();
}

shocker::ShockerNode* ShockerSceneHandler::findNodeForSurface(shocker::ShockerSurface* surface) const
{
    if (!surface) {
        return nullptr;
    }
    
    // Search through all nodes to find the one containing this surface
    // Note: ShockerSurfaceGroupInstance doesn't directly contain surfaces
    // We need to find the model that contains this surface and then find the node
    // For now, return nullptr as we need a different approach
    // TODO: Implement proper surface-to-node mapping
    
    return nullptr;
}