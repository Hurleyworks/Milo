#include "RiPRSceneHandler.h"
#include "../../../handlers/Handlers.h"
#include "RiPRModelHandler.h"
#include "../models/RiPRModel.h"

using Eigen::Affine3f;
using sabi::PRenderableState;

RiPRSceneHandler::RiPRSceneHandler (RenderContextPtr ctx) :
    ctx (ctx)
{
    LOG (DBUG) << _FN_;
    
    // Initialize the generic scene handler and configure it for RiPR's needs
    instanceHandler_ = ctx->getHandlers().intanceHandler;
    if (instanceHandler_)
    {
        // Configure for fast build with update capability (RiPR needs frequent updates)
        instanceHandler_->setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            true,   // allowUpdate - RiPR needs to update transforms
            false   // allowCompaction - not needed for RiPR
        );
    }
}

// Cleans up displacement and CUDA module resources
RiPRSceneHandler::~RiPRSceneHandler()
{
    finalize();
}

void RiPRSceneHandler::finalize()
{
    try
    {
        // First synchronize stream to ensure pending operations complete
        if (ctx && ctx->getCudaStream())
        {
            CUDADRV_CHECK (cuStreamSynchronize (ctx->getCudaStream()));
        }

        // SceneHandler manages IAS cleanup - just finalize it
        if (instanceHandler_)
        {
            instanceHandler_->finalize();
        }

        // Release instance data buffers
        for (int i = 0; i < 2; ++i)
        {
            if (instanceDataBuffer_[i].isInitialized())
            {
                instanceDataBuffer_[i].finalize();
            }
            
            // Clean up per-instance geometry slot buffers
            for (auto& [instIdx, buffer] : geomInstSlotsBuffers_[i])
            {
                if (buffer.isInitialized())
                {
                    buffer.finalize();
                }
            }
            geomInstSlotsBuffers_[i].clear();
            
            // Clean up per-instance light distributions
            for (auto& [instIdx, dist] : perInstanceLightDists_[i])
            {
                if (dist.isInitialized())
                {
                    dist.finalize();
                }
            }
            perInstanceLightDists_[i].clear();
        }
        
        // Finalize light instance distribution
        if (lightInstDistribution_.isInitialized())
        {
            lightInstDistribution_.finalize();
        }

        // Release displacement buffer
        if (displacementBuffer.isInitialized())
        {
            displacementBuffer.finalize();
        }

        // Clear node map to release references
        nodeMap.clear();


        // Only unload module if we have a valid context
        if (moduleDeform)
        {
            CUcontext current = nullptr;
            CUresult result = cuCtxGetCurrent (&current);

            if (result == CUDA_SUCCESS && current)
            {
                LOG (DBUG) << "Unloading deform module";
                CUDADRV_CHECK (cuModuleUnload (moduleDeform));
            }
            else
            {
                LOG (WARNING) << "No current CUDA context when trying to unload module";
            }
            moduleDeform = 0;
        }

        // Note: Scratch buffer is now shared and managed by RenderContext
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error during RiPRSceneHandler cleanup: " << e.what();
    }
}

void RiPRSceneHandler::init()
{
    LOG (DBUG) << _FN_;
    
    // Initialize SceneHandler if needed
    if (!instanceHandler_)
    {
        instanceHandler_ = ctx->getHandlers().intanceHandler;
        if (instanceHandler_)
        {
            instanceHandler_->initialize();
            // Configure for fast build with update capability (RiPR needs frequent updates)
            instanceHandler_->setConfiguration(
                optixu::ASTradeoff::PreferFastBuild,
                true,   // allowUpdate - RiPR needs to update transforms
                false   // allowCompaction - not needed for RiPR
            );
        }
    }
    
    if (!scene_)
    {
        LOG (WARNING) << "Scene not set - cannot initialize";
        return;
    }

    // Generate initial scene SBT layout before any IAS operations
    // This prevents the "Shader binding table layout generation has not been done" error
    size_t dummySize;
    scene_.generateShaderBindingTableLayout (&dummySize);
    LOG (DBUG) << "Generated initial scene SBT layout";

    // Load deformation kernel PTX/OptiXIR using PTXManager
    bool useEmbedded = ctx->getPropertyService().renderProps->getVal<bool> (RenderKey::UseEmbeddedPTX);
    std::vector<char> ptxData;

    try
    {
        // Use PTXManager to get data for deform kernel
        ptxData = ctx->getPTXManager()->getPTXData ("optix_ripr_deform_kernels", useEmbedded);

        // Load the module from memory
        if (!ptxData.empty())
        {
            LOG (DBUG) << "Loading deform kernel using PTXManager (" << ptxData.size() << " bytes)";
            CUDADRV_CHECK (cuModuleLoadData (&moduleDeform, ptxData.data()));
        }
        else
        {
            throw std::runtime_error ("Empty PTX data for deform kernel");
        }
    }
    catch (const std::exception& e)
    {
        // Fallback to direct file loading on error
        LOG (WARNING) << "Failed to load deform kernel via PTXManager: " << e.what();

        std::filesystem::path ptxPath;
#ifndef NDEBUG
        ptxPath = ctx->getPropertyService().renderProps->getVal<fs::path> (RenderKey::ResourceFolder) / "ptx" / "Debug" / "deform.ptx";
#else
        ptxPath = ctx->getPropertyService().renderProps->getVal<fs::path> (RenderKey::ResourceFolder) / "ptx" / "Release" / "deform.ptx";
#endif
        LOG (DBUG) << "Falling back to loading deform kernel from file: " << ptxPath.string();
        CUDADRV_CHECK (cuModuleLoad (&moduleDeform, ptxPath.generic_string().c_str()));
    }

    // Initialize kernels
    kernelDeform.set (moduleDeform, "deform", cudau::dim3 (32), 0);
    kernelResetDeform.set (moduleDeform, "resetDeform", cudau::dim3 (32), 0);
    kernelAccumulateVertexNormals.set (moduleDeform, "accumulateVertexNormals", cudau::dim3 (32), 0);
    kernelNormalizeVertexNormals.set (moduleDeform, "normalizeVertexNormals", cudau::dim3 (32), 0);

    // Initialize instance data buffers (double buffered)
    // These need to be initialized early so they're ready when geometry is added
    const cudau::BufferType bufferType = cudau::BufferType::Device;
    instanceDataBuffer_[0].initialize (ctx->getCudaContext(), bufferType, maxNumInstances);
    instanceDataBuffer_[1].initialize (ctx->getCudaContext(), bufferType, maxNumInstances);
    LOG (DBUG) << "Initialized instance data buffers with capacity: " << maxNumInstances;

      // Initialize acceleration structure build scratch memory
    // Note: Scratch buffer is now shared and managed by RenderContext
    // We'll use ctx->getASScratchBuffer() when needed for GAS operations
}

// Prepare for building the IAS

// Initialize Scene Dependent Shader Binding Table (SBT)
// TODO: Implement with new entry point system
#if 0
void RiPRSceneHandler::initializeSceneDependentSBT (EntryPointType type)
{
    // Get the pipeline for  entry point
    // Pipeline handling needs to be updated for new architecture
    // auto pipeline = ctx->getHandlers().pl->getPipeline (type);
    // TODO: Implement pipeline handling in RiPREngine
    if (!pipeline)
        throw std::runtime_error ("Pipeline not in database!");

    // Get the size of the Hit Group SBT
    size_t hitGroupSbtSize;
    // Scene SBT generation needs to be handled differently
    // ctx->getOptiXContext().generateShaderBindingTableLayout (&hitGroupSbtSize);
    // TODO: Implement SBT generation in RiPREngine

    // Initialize the scene dependent SBT
    // pipeline->sceneDependentSBT.initialize (ctx->getCudaContext(), cudau::BufferType::Device, hitGroupSbtSize, 1);
    // TODO: Implement SBT initialization in RiPREngine

    // Keep the mapped memory persistent
    // pipeline->sceneDependentSBT.setMappedMemoryPersistent (true);
}
#endif

// Resize scene dependent Shader Binding Table (SBT)
void RiPRSceneHandler::resizeSceneDependentSBT()
{
    if (!scene_)
    {
        LOG (WARNING) << "Scene not set - cannot resize SBT";
        return;
    }

    // Generate the shader binding table layout
    // This MUST be called before building the IAS
    size_t hitGroupSbtSize;
    scene_.generateShaderBindingTableLayout (&hitGroupSbtSize);

    LOG (DBUG) << "Generated scene SBT layout, size: " << hitGroupSbtSize << " bytes";
}

void RiPRSceneHandler::undeformNode (RenderableNode node)
{
    RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getClientID());
    if (!optiXModel)
    {
        LOG (WARNING) << "Could not find OptiXModel for " << node->getName();
        return;
    }

    auto triangleModel = std::dynamic_pointer_cast<RiPRTriangleModel> (optiXModel);
    if (!triangleModel) return;

    try
    {
        const size_t vertexCount = triangleModel->getCurrentVertexBuffer().numElements();

        // Step 1: Reset vertex positions and clear normals
        kernelResetDeform.launchWithThreadDim (
            ctx->getCudaStream(),
            cudau::dim3 (vertexCount),
            static_cast<const shared::Vertex*> (triangleModel->getOriginalVertexBuffer().getDevicePointer()),
            static_cast<shared::Vertex*> (triangleModel->getCurrentVertexBuffer().getDevicePointer()),
            vertexCount);

        // Verify completion of the reset operation
        CUDADRV_CHECK (cuStreamSynchronize (ctx->getCudaStream()));

        // Step 2: Accumulate face normals to vertices
        const auto& triangleBuffer = triangleModel->getTriangleBuffer();
        kernelAccumulateVertexNormals.launchWithThreadDim (
            ctx->getCudaStream(),
            cudau::dim3 (triangleBuffer.numElements()),
            static_cast<shared::Vertex*> (triangleModel->getCurrentVertexBuffer().getDevicePointer()),
            static_cast<shared::Triangle*> (triangleBuffer.getDevicePointer()),
            triangleBuffer.numElements());

        // Step 3: Normalize the accumulated vertex normals
        kernelNormalizeVertexNormals.launchWithThreadDim (
            ctx->getCudaStream(),
            cudau::dim3 (vertexCount),
            static_cast<shared::Vertex*> (triangleModel->getCurrentVertexBuffer().getDevicePointer()),
            vertexCount);

        // Ensure all normal computation is complete before rebuilding GAS
        CUDADRV_CHECK (cuStreamSynchronize (ctx->getCudaStream()));

        // Step 4: Update GAS
        GAS* gas = triangleModel->getGAS();
        if (gas)
        {
            gas->gas.rebuild (ctx->getCudaStream(), gas->gasMem, ctx->getASScratchBuffer());
        }

        // LOG (DBUG) << "Reset deformation and recomputed normals for " << node->getName();
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error undeforming node " << node->getName()
                      << ": " << e.what();
    }
}

void RiPRSceneHandler::updateDeformedNode (RenderableNode node)
{
    RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getClientID());
    if (!optiXModel)
    {
        LOG (DBUG) << "Could not find OptiXModel for " << node->getName();
        return;
    }

    auto triangleModel = std::dynamic_pointer_cast<RiPRTriangleModel> (optiXModel);
    if (!triangleModel || !triangleModel->hasDeformation())
        return;

    try
    {
        CgModelPtr model = node->getModel();
        if (!model || model->VD.cols() == 0)
        {
            LOG (DBUG) << "No displacement data for " << node->getName();
            return;
        }

        const size_t vertexCount = triangleModel->getCurrentVertexBuffer().numElements();

        // Initialize or resize displacement buffer if needed
        if (!displacementBuffer.isInitialized())
        {
            displacementBuffer.initialize (
                ctx->getCudaContext(),
                cudau::BufferType::Device,
                vertexCount,
                sizeof (float3));
        }
        else if (displacementBuffer.numElements() != vertexCount)
        {
            displacementBuffer.resize (
                vertexCount,
                sizeof (float3),
                ctx->getCudaStream());
        }

        // Copy deformed positions to GPU
        float3* mappedDisp = static_cast<float3*> (
            displacementBuffer.map (ctx->getCudaStream(), cudau::BufferMapFlag::WriteOnlyDiscard));

        Eigen::Vector3f scale = node->getSpaceTime().scale;
        for (int i = 0; i < vertexCount; i++)
        {
            mappedDisp[i] = make_float3 (
                model->VD.col (i).x(),
                model->VD.col (i).y(),
                model->VD.col (i).z());
        }

        displacementBuffer.unmap (ctx->getCudaStream());

        // Launch deformation kernels
        kernelDeform.launchWithThreadDim (
            ctx->getCudaStream(),
            cudau::dim3 (vertexCount),
            triangleModel->getOriginalVertexBuffer().getDevicePointer(),
            triangleModel->getCurrentVertexBuffer().getDevicePointer(),
            vertexCount,
            static_cast<const float3*> (displacementBuffer.getDevicePointer()));

        const auto& triangleBuffer = triangleModel->getTriangleBuffer();
        kernelAccumulateVertexNormals.launchWithThreadDim (
            ctx->getCudaStream(),
            cudau::dim3 (triangleBuffer.numElements()),
            static_cast<shared::Vertex*> (triangleModel->getCurrentVertexBuffer().getDevicePointer()),
            static_cast<shared::Triangle*> (triangleBuffer.getDevicePointer()),
            triangleBuffer.numElements());

        kernelNormalizeVertexNormals.launchWithThreadDim (
            ctx->getCudaStream(),
            cudau::dim3 (vertexCount),
            static_cast<shared::Vertex*> (triangleModel->getCurrentVertexBuffer().getDevicePointer()),
            vertexCount);

        // Critical: Update GAS after deformation
        GAS* gas = triangleModel->getGAS();
        if (gas)
        {
            gas->gas.rebuild (ctx->getCudaStream(), gas->gasMem, ctx->getASScratchBuffer());
        }
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error updating deformed node " << node->getName()
                      << ": " << e.what();
    }
}

void RiPRSceneHandler::removeNodesByIDs (const std::vector<BodyID>& bodyIDs)
{
    if (bodyIDs.empty()) return;

    // Step 1: Find all nodes to remove and their indices
    std::vector<uint32_t> indicesToRemove;

    for (BodyID bodyID : bodyIDs)
    {
        for (const auto& [index, weakRef] : nodeMap)
        {
            if (!weakRef.expired())
            {
                RenderableNode node = weakRef.lock();
                if (node && node->getClientID() == bodyID)
                {
                    indicesToRemove.push_back (index);
                    node->getState().state &= ~sabi::PRenderableState::StoredInSceneHandler;
                    break;
                }
            }
        }
    }

    if (indicesToRemove.empty())
    {
        LOG (DBUG) << "No matching instances found in OptiX scene - skipping removal";
        return;
    }

    // Step 2: Sort indices in descending order for stable removal
    std::sort (indicesToRemove.rbegin(), indicesToRemove.rend());

    // Step 3: Remove from IAS in descending order (no index shifts affect remaining removals)
    for (uint32_t index : indicesToRemove)
    {
        if (instanceHandler_ && index < instanceHandler_->getInstanceCount())
        {
            instanceHandler_->removeInstanceAt(index);
        }
        else
        {
            LOG (WARNING) << "Invalid index " << index << " (max: " << (instanceHandler_ ? instanceHandler_->getInstanceCount() - 1 : 0) << ")";
        }
    }

    // Step 4: Rebuild nodeMap with correct indices in a single pass
    NodeMap newNodeMap;
    uint32_t newIndex = 0;

    // Go through all existing indices in order
    for (const auto& [oldIndex, weakRef] : nodeMap)
    {
        // Skip if this index was removed
        if (std::find (indicesToRemove.begin(), indicesToRemove.end(), oldIndex) == indicesToRemove.end())
        {
            if (!weakRef.expired())
            {
                newNodeMap[newIndex++] = weakRef;
            }
        }
    }

    nodeMap = std::move (newNodeMap);

    // Step 5: Rebuild IAS and update SBT
    LOG (DBUG) << "Rebuilding IAS after removing " << indicesToRemove.size() << " instances";
    rebuild();
}

void RiPRSceneHandler::createInstance (RenderableWeakRef& weakNode)
{
    // debug_assert (false);

    if (weakNode.expired()) return;

    RenderableNode node = weakNode.lock();

    if (!instanceHandler_ || instanceHandler_->getTraversableHandle() == 0)
        init();

    // Create a new OptiX instance
    optixu::Instance instance = scene_.createInstance();

    // Instances don't have a LWITEMID converted to a ClientID
    // RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getID());
    RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getClientID());
    optiXModel->setOptiXInstance (instance);

    // Set the Geometry Acceleration Structure (GAS)
    instance.setChild (optiXModel->getGAS()->gas);

    const SpaceTime& st = node->getSpaceTime();
    MatrixRowMajor34f t;
    getWorldTransform (t, st);
    instance.setTransform (t.data());

    // Add the instance to the scene using SceneHandler
    if (instanceHandler_)
    {
        instanceHandler_->addInstance(instance);
    }

    node->getState().state |= sabi::PRenderableState::StoredInSceneHandler;

    // Track the instance in nodeMap using the current count as index
    uint32_t index = instanceHandler_ ? static_cast<uint32_t>(instanceHandler_->getInstanceCount() - 1) : 0;
    nodeMap[index] = weakNode;
    
    // Set instance ID BEFORE rebuilding - this is critical!
    instance.setID (index);

    GAS* gasData = optiXModel->getGAS();
    gasData->gas.rebuild (ctx->getCudaStream(), gasData->gasMem, ctx->getASScratchBuffer());


    rebuild();

    // Populate instance data for GPU access
    populateInstanceData (index, node);
    
    // Don't build light distributions here - they should be built after
    // the geometry distributions are computed in RiPRModelHandler::computeLightProbabilities
}

void RiPRSceneHandler::populateInstanceData (uint32_t instanceIndex, const RenderableNode& node)
{
    if (instanceIndex >= maxNumInstances)
    {
        LOG (WARNING) << "Instance index " << instanceIndex << " exceeds max instances " << maxNumInstances;
        return;
    }

    // Map both buffers and populate with instance data
    for (int bufferIdx = 0; bufferIdx < 2; ++bufferIdx)
    {
        shared::InstanceData* instDataOnHost = instanceDataBuffer_[bufferIdx].map();
        if (!instDataOnHost)
        {
            LOG (WARNING) << "Failed to map instance data buffer " << bufferIdx;
            continue;
        }

        shared::InstanceData& instData = instDataOnHost[instanceIndex];

        // Get world transform from node
        const Eigen::Matrix4f& worldTransform = node->getSpaceTime().worldTransform.matrix();

        // Convert Eigen matrix to shared Matrix4x4 (column-major)
        Matrix4x4 transform (
            Vector4D (worldTransform (0, 0), worldTransform (1, 0), worldTransform (2, 0), worldTransform (3, 0)), // column 0
            Vector4D (worldTransform (0, 1), worldTransform (1, 1), worldTransform (2, 1), worldTransform (3, 1)), // column 1
            Vector4D (worldTransform (0, 2), worldTransform (1, 2), worldTransform (2, 2), worldTransform (3, 2)), // column 2
            Vector4D (worldTransform (0, 3), worldTransform (1, 3), worldTransform (2, 3), worldTransform (3, 3))  // column 3
        );
        instData.transform = transform;

        // For now, no motion blur - identity transform
        instData.curToPrevTransform = Matrix4x4();

        // Compute normal matrix (inverse transpose of upper 3x3)
        Matrix3x3 upperLeft (
            Vector3D (worldTransform (0, 0), worldTransform (1, 0), worldTransform (2, 0)), // column 0
            Vector3D (worldTransform (0, 1), worldTransform (1, 1), worldTransform (2, 1)), // column 1
            Vector3D (worldTransform (0, 2), worldTransform (1, 2), worldTransform (2, 2))  // column 2
        );
        instData.normalMatrix = transpose (invert (upperLeft));

        // Compute uniform scale from the transform
        float uniformScale = node->getSpaceTime().scale.norm() / std::sqrt(3.0f);
        instData.uniformScale = uniformScale;

        // Get the model to check if it has emissive geometry
        if (modelHandler_)
        {
            auto model = modelHandler_->getModel(node);
            if (auto triModel = std::dynamic_pointer_cast<RiPRTriangleModel>(model))
            {
                // Check if this model has emissive materials
                if (triModel->hasEmissiveMaterials())
                {
                    // For now, populate with the single geometry instance slot
                    // In a full implementation, we'd need to handle multiple geometry instances
                    uint32_t geomInstSlot = triModel->getGeometryInstanceSlot();
                    
                    if (geomInstSlot != SlotFinder::InvalidSlotIndex)
                    {
                        // Create a buffer for the geometry instance slots
                        // For single geometry, we just have one slot
                        if (!geomInstSlotsBuffers_[bufferIdx][instanceIndex].isInitialized())
                        {
                            geomInstSlotsBuffers_[bufferIdx][instanceIndex].initialize(
                                ctx->getCudaContext(), cudau::BufferType::Device, 1);
                        }
                        uint32_t* slotPtr = geomInstSlotsBuffers_[bufferIdx][instanceIndex].map();
                        slotPtr[0] = geomInstSlot;
                        geomInstSlotsBuffers_[bufferIdx][instanceIndex].unmap();
                        
                        instData.geomInstSlots = geomInstSlotsBuffers_[bufferIdx][instanceIndex].getROBuffer<shared::enableBufferOobCheck>();
                        
                        // Create a simple distribution with one geometry (weight = 1.0)
                        float geomImportance = 1.0f;
                        if (!perInstanceLightDists_[bufferIdx][instanceIndex].isInitialized())
                        {
                            perInstanceLightDists_[bufferIdx][instanceIndex].initialize(
                                ctx->getCudaContext(), cudau::BufferType::Device, &geomImportance, 1);
                        }
                        perInstanceLightDists_[bufferIdx][instanceIndex].getDeviceType(&instData.lightGeomInstDist);
                        
                        // Mark as emissive
                        instData.isEmissive = 1;
                        instData.emissiveScale = 1.0f;
                    }
                    else
                    {
                        // No valid geometry slot
                        instData.geomInstSlots = shared::ROBuffer<uint32_t>();
                        instData.lightGeomInstDist = shared::LightDistribution();
                        instData.isEmissive = 0;
                    }
                }
                else
                {
                    // Not emissive - clean up any existing buffers
                    // Clean up buffers if they exist from a previous emissive state
                    if (geomInstSlotsBuffers_[bufferIdx][instanceIndex].isInitialized())
                    {
                        geomInstSlotsBuffers_[bufferIdx][instanceIndex].finalize();
                    }
                    if (perInstanceLightDists_[bufferIdx][instanceIndex].isInitialized())
                    {
                        perInstanceLightDists_[bufferIdx][instanceIndex].finalize();
                    }
                    
                    instData.geomInstSlots = shared::ROBuffer<uint32_t>();
                    instData.lightGeomInstDist = shared::LightDistribution();
                    instData.isEmissive = 0;
                }
            }
            else
            {
                // Not a triangle model or no model - clean up any existing buffers
                if (geomInstSlotsBuffers_[bufferIdx][instanceIndex].isInitialized())
                {
                    geomInstSlotsBuffers_[bufferIdx][instanceIndex].finalize();
                }
                if (perInstanceLightDists_[bufferIdx][instanceIndex].isInitialized())
                {
                    perInstanceLightDists_[bufferIdx][instanceIndex].finalize();
                }
                
                instData.geomInstSlots = shared::ROBuffer<uint32_t>();
                instData.lightGeomInstDist = shared::LightDistribution();
                instData.isEmissive = 0;
            }
        }
        else
        {
            // No model handler - clean up any existing buffers
            if (geomInstSlotsBuffers_[bufferIdx][instanceIndex].isInitialized())
            {
                geomInstSlotsBuffers_[bufferIdx][instanceIndex].finalize();
            }
            if (perInstanceLightDists_[bufferIdx][instanceIndex].isInitialized())
            {
                perInstanceLightDists_[bufferIdx][instanceIndex].finalize();
            }
            
            instData.geomInstSlots = shared::ROBuffer<uint32_t>();
            instData.lightGeomInstDist = shared::LightDistribution();
            instData.isEmissive = 0;
        }

        instanceDataBuffer_[bufferIdx].unmap();
    }
}

void RiPRSceneHandler::createGeometryInstance (RenderableWeakRef& weakNode)
{
    if (weakNode.expired()) return;

    RenderableNode node = weakNode.lock();

    RenderableNode instancedFrom = node->getInstancedFrom();
    if (!instancedFrom)
    {
        LOG (WARNING) << "Invalid instance has no Instanced From " << node->getName();
        return;
    }

    //  LOG (DBUG) << "Creating geometry instance for " << node->getName()
    //           << " instancedFrom: " << instancedFrom->getName()
    //         << " instancedFrom clientID: " << instancedFrom->getClientID();

    // Check if the instancedFrom model exists
    RiPRModelPtr instancedFromModel = modelHandler_->getRiPRModel (instancedFrom->getClientID());
    if (!instancedFromModel)
    {
        LOG (WARNING) << "Can't find Instanced from in database. Name: " << instancedFrom->getName()
                      << " ID: " << instancedFrom->getID()
                      << " ClientID: " << instancedFrom->getClientID();
        return;
    }

    RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getClientID());
    if (!optiXModel)
    {
        LOG (WARNING) << "Could not find OptiXModel for " << node->getName();
        return;
    }

    // Add safety check for GAS
    GAS* gasData = instancedFromModel->getGAS();
    if (!gasData)
    {
        LOG (WARNING) << "Source model has no GAS: " << instancedFrom->getName();
        return;
    }

    // Create a new OptiX instance
    optixu::Instance instance = scene_.createInstance();

    optiXModel->setOptiXInstance (instance);

    // Set the Geometry Acceleration Structure (GAS) using the InstancedFrom
    instance.setChild (gasData->gas);

    // Set the instance transform using the given pose
    const Eigen::Matrix4f& m = node->getSpaceTime().worldTransform.matrix();
    MatrixRowMajor34f t = m.block<3, 4> (0, 0);
    instance.setTransform (t.data());

    // Add the instance to the scene using SceneHandler
    if (instanceHandler_)
    {
        instanceHandler_->addInstance(instance);
    }

    // Set the flag to indicate this node is stored in RiPRSceneHandler
    node->getState().state |= sabi::PRenderableState::StoredInSceneHandler;

    // Track the instance in nodeMap using the current count as index
    uint32_t index = instanceHandler_ ? static_cast<uint32_t>(instanceHandler_->getInstanceCount() - 1) : 0;
    nodeMap[index] = weakNode;


    rebuild();

    // Populate instance data for GPU access
    populateInstanceData (index, node);
    
    // Don't build light distributions here - they should be built after
    // the geometry distributions are computed in RiPRModelHandler::computeLightProbabilities
}

void RiPRSceneHandler::createPhysicsPhantom (RenderableWeakRef& weakNode)
{
    if (weakNode.expired()) return;
    try
    {
        RenderableNode node = weakNode.lock();
        if (node->isInstance())
            throw std::runtime_error ("Can't make a phantom from an instance!");

        // Create a new OptiX instance
        optixu::Instance instance = scene_.createInstance();

        RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getID());
        if (!optiXModel) throw std::runtime_error ("could not find optix model bound to " + node->getName());

        optiXModel->setOptiXInstance (instance);

        RenderableNode phantomFrom = node->getPhantomFrom();
        if (!phantomFrom) throw std::runtime_error (node->getName() + " has no phantom model");

        RiPRModelPtr phantomFromModel = modelHandler_->getRiPRModel (phantomFrom->getID());
        if (!phantomFromModel) throw std::runtime_error (node->getName() + " has no phantomFrom model");

        // Set the Geometry Acceleration Structure (GAS) using the PhantomFrom
        // set the phantom material(blue);
        instance.setChild (phantomFromModel->getGAS()->gas, 2);

        // Set the instance transform using the given pose
        const Eigen::Matrix4f& m = node->getSpaceTime().worldTransform.matrix();
        MatrixRowMajor34f t = m.block<3, 4> (0, 0);
        instance.setTransform (t.data());

        // Add the instance to the scene using SceneHandler
        if (instanceHandler_)
        {
            instanceHandler_->addInstance(instance);
        }

        // Set the flag to indicate this node is stored in RiPRSceneHandler
        node->getState().state |= sabi::PRenderableState::StoredInSceneHandler;

        // Track the instance in nodeMap using the current count as index
        uint32_t index = instanceHandler_ ? static_cast<uint32_t>(instanceHandler_->getInstanceCount() - 1) : 0;
        nodeMap[index] = weakNode;

    
        rebuild();

        // Populate instance data for GPU access
        populateInstanceData (index, node);
    }
    catch (const std::exception& e)
    {
        LOG (CRITICAL) << e.what();
    }
}

void RiPRSceneHandler::createInstanceList (const WeakRenderableList& weakNodeList)
{
    if (!instanceHandler_ || instanceHandler_->getTraversableHandle() == 0)
        init();

    for (const auto& weakNode : weakNodeList)
    {
        if (weakNode.expired()) continue;
        RenderableNode node = weakNode.lock();

        if (node->isInstance())
        {
            // Create a new OptiX instance
            optixu::Instance instance = scene_.createInstance();

            RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getID());
            optiXModel->setOptiXInstance (instance);

            RenderableNode instancedFrom = node->getInstancedFrom();
            if (!instancedFrom) continue;

            RiPRModelPtr instancedFromModel = modelHandler_->getRiPRModel (instancedFrom->getID());
            if (!instancedFromModel) continue;

            // Set the Geometry Acceleration Structure (GAS) using the InstancedFrom
            instance.setChild (instancedFromModel->getGAS()->gas);

            // Set the instance transform using the given pose
            const Eigen::Matrix4f& m = node->getSpaceTime().worldTransform.matrix();
            MatrixRowMajor34f t = m.block<3, 4> (0, 0);
            instance.setTransform (t.data());

            // Add the instance to the scene using SceneHandler
            if (instanceHandler_)
            {
                instanceHandler_->addInstance(instance);
            }

            // Set the flag to indicate this node is stored in RiPRSceneHandler
            node->getState().state |= sabi::PRenderableState::StoredInSceneHandler;

            // Track the instance in nodeMap using the current count as index
        uint32_t index = instanceHandler_ ? static_cast<uint32_t>(instanceHandler_->getInstanceCount() - 1) : 0;
            nodeMap[index] = weakNode;

            // Store index for later instance data population
            populateInstanceData (index, node);
        }
        else
        {
            RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getID());
            if (!optiXModel)
            {
                LOG (CRITICAL) << "Failed to find RiPRModelPtr";
                continue;
            }

            LOG (DBUG) << "Processing " << node->getName();

            // Create a new OptiX instance
            optixu::Instance instance = scene_.createInstance();
            optiXModel->setOptiXInstance (instance);

            // Set the Geometry Acceleration Structure (GAS)
            instance.setChild (optiXModel->getGAS()->gas);

            // Set the instance transform using the given pose
            const Eigen::Matrix4f& m = node->getSpaceTime().worldTransform.matrix();
            MatrixRowMajor34f t = m.block<3, 4> (0, 0);
            instance.setTransform (t.data());

            // Add the instance to the scene using SceneHandler
            if (instanceHandler_)
            {
                instanceHandler_->addInstance(instance);
            }

            // Set the flag to indicate this node is stored in RiPRSceneHandler
            node->getState().state |= sabi::PRenderableState::StoredInSceneHandler;

            // Track the instance in nodeMap using the current count as index
        uint32_t index = instanceHandler_ ? static_cast<uint32_t>(instanceHandler_->getInstanceCount() - 1) : 0;
            nodeMap[index] = weakNode;

            GAS* gasData = optiXModel->getGAS();
            gasData->gas.rebuild (ctx->getCudaStream(), gasData->gasMem, ctx->getASScratchBuffer());

            // Populate instance data for GPU access
            populateInstanceData (index, node);
        }
    }


    rebuild();
    
    // Don't build light distributions here - they should be built after
    // the geometry distributions are computed in RiPRModelHandler::computeLightProbabilities
}

bool RiPRSceneHandler::updateMotion()
{
    // Updates node transformations and visibility in the scene
    bool restartRender = false;
    uint32_t bodyCount = nodeMap.size();
    if (!bodyCount) return restartRender;
    uint32_t invisibleBodies = 0;
    uint32_t bodiesSleeping = 0;
    for (auto& it : nodeMap)
    {
        RenderableWeakRef weakNode = it.second;
        if (weakNode.expired()) continue;
        RenderableNode node = weakNode.lock();
        if (!node) continue;

        bool isDeformed = node->getState().isDeformed();

        // Check for deformation state changes
        if (node->getState().state & PRenderableState::DeformedStateChanged)
        {
            if (isDeformed)
            {
                // Handle transition to deformed state
                updateDeformedNode (node);
            }
            else
            {
                // Handle transition to undeformed state
                undeformNode (node);
            }

            // Clear the state change flag after handling
            node->getState().state &= ~PRenderableState::DeformedStateChanged;
        }
        else if (isDeformed)
        {
            updateDeformedNode (node);
        }

        optixu::Instance instance = instanceHandler_ ? instanceHandler_->getInstance(it.first) : optixu::Instance();
        const SpaceTime& st = node->getSpaceTime();
        MatrixRowMajor34f t;

        // deformed meshes are already in world space and have scale applied
        getWorldTransform (t, isDeformed ? Eigen::Affine3f::Identity() : st.worldTransform, isDeformed ? Eigen::Vector3f::Ones() : st.scale);
        instance.setTransform (t.data());
        // check visibility state
        uint32_t visiblityMask = node->getState().isVisible() ? 255 : 0;
        if (visiblityMask == 0) ++invisibleBodies;
        instance.setVisibilityMask (visiblityMask);
        if (node->description().sleepState)
            ++bodiesSleeping;

        // Update instance data for this instance
        populateInstanceData (it.first, node);
    }
    if (bodiesSleeping < bodyCount)
    {
        if (instanceHandler_)
        {
            instanceHandler_->buildOrUpdateIAS();
        }
        restartRender = true;
    }
    return restartRender;
}

void RiPRSceneHandler::toggleSelectionMaterial (RenderableNode& node)
{
    // node is checked upstream
    RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getID());

    optixu::Instance& inst = optiXModel->getOptiXInstance();

    if (node->isInstance())
    {
        RenderableNode instancedFrom = node->getInstancedFrom();
        if (!instancedFrom) return; // FIXME better error handling

        RiPRModelPtr instancedFromModel = modelHandler_->getRiPRModel (instancedFrom->getID());
        if (!instancedFromModel) return; // FIXME better error handling

        inst.setChild (instancedFromModel->getGAS()->gas, node->getState().isSelected() ? 1 : 0);
    }
    else
    {
        inst.setChild (optiXModel->getGAS()->gas, node->getState().isSelected() ? 1 : 0);
    }

    // apparently no need to resize SceneDependentSBT
    if (instanceHandler_)
    {
        instanceHandler_->buildOrUpdateIAS();
    }
}

void RiPRSceneHandler::selectAll()
{
    for (auto& it : nodeMap)
    {
        if (it.second.expired()) continue;

        RenderableNode node = it.second.lock();
        node->getState().state |= sabi::PRenderableState::Selected;

        RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getID());
        if (!optiXModel) continue; // FIXME error handling

        optixu::Instance& inst = optiXModel->getOptiXInstance();

        if (node->isInstance())
        {
            RenderableNode instancedFrom = node->getInstancedFrom();
            if (!instancedFrom) return; // FIXME better error handling

            RiPRModelPtr instancedFromModel = modelHandler_->getRiPRModel (instancedFrom->getID());
            if (!instancedFromModel) return; // FIXME better error handling

            inst.setChild (instancedFromModel->getGAS()->gas, node->getState().isSelected() ? 1 : 0);
        }
        else
        {
            inst.setChild (optiXModel->getGAS()->gas, node->getState().isSelected() ? 1 : 0);
        }
    }

    // apparently no need to resize SceneDependentSBT
    if (instanceHandler_)
    {
        instanceHandler_->buildOrUpdateIAS();
    }
}

// FIXME lot of duplicate code here
void RiPRSceneHandler::deselectAll()
{
    for (auto& it : nodeMap)
    {
        if (it.second.expired()) continue;

        RenderableNode node = it.second.lock();
        if (node->getState().isSelected())
        {
            node->getState().state ^= sabi::PRenderableState::Selected;
        }

        RiPRModelPtr optiXModel = modelHandler_->getRiPRModel (node->getID());
        if (!optiXModel) continue; // FIXME error handling

        optixu::Instance& inst = optiXModel->getOptiXInstance();

        if (node->isInstance())
        {
            RenderableNode instancedFrom = node->getInstancedFrom();
            if (!instancedFrom) return; // FIXME better error handling

            RiPRModelPtr instancedFromModel = modelHandler_->getRiPRModel (instancedFrom->getID());
            if (!instancedFromModel) return; // FIXME better error handling

            inst.setChild (instancedFromModel->getGAS()->gas, node->getState().isSelected() ? 1 : 0);
        }
        else
        {
            inst.setChild (optiXModel->getGAS()->gas, node->getState().isSelected() ? 1 : 0);
        }
    }

    // apparently no need to resize SceneDependentSBT
    if (instanceHandler_)
    {
        instanceHandler_->buildOrUpdateIAS();
    }
}

void RiPRSceneHandler::rebuild()
{
    resizeSceneDependentSBT();
    
    // Rebuild the IAS using SceneHandler
    if (instanceHandler_)
    {
        instanceHandler_->buildOrUpdateIAS();
    }
}

// Area light support implementation
void RiPRSceneHandler::updateEmissiveInstances()
{
    emissiveInstances_.clear();

    // Check if we have a valid model handler
    if (!modelHandler_)
    {
        LOG (WARNING) << "No model handler set for emissive instance update";
        return;
    }

    // Iterate through all instances and check if they have emissive materials
    for (const auto& [instanceIndex, weakRef] : nodeMap)
    {
        if (weakRef.expired()) continue;

        auto node = weakRef.lock();
        if (!node) continue;

        // Get the model from the node
        auto model = modelHandler_->getModel (node);
        if (!model) continue;

        // Check if the model has any emissive materials
        bool hasEmissive = false;

        // For RiPRTriangleModel, check if it was marked as having emissive materials
        if (auto triModel = std::dynamic_pointer_cast<RiPRTriangleModel> (model))
        {
            hasEmissive = triModel->hasEmissiveMaterials();
        }

        if (hasEmissive)
        {
            emissiveInstances_.push_back (instanceIndex);

            // Mark instance as emissive in instance data
            for (int bufferIdx = 0; bufferIdx < 2; ++bufferIdx)
            {
                shared::InstanceData* instDataOnHost = instanceDataBuffer_[bufferIdx].map();
                if (instDataOnHost && instanceIndex < maxNumInstances)
                {
                    instDataOnHost[instanceIndex].isEmissive = 1;
                    instDataOnHost[instanceIndex].emissiveScale = 1.0f; // Default scale
                }
                instanceDataBuffer_[bufferIdx].unmap();
            }
        }
    }

    lightDistributionDirty_ = true;
    // LOG(DBUG) << "Found " << emissiveInstances_.size() << " emissive instances";
}

void RiPRSceneHandler::buildLightInstanceDistribution()
{
    if (!lightDistributionDirty_) return;

    if (emissiveInstances_.empty())
    {
        // Initialize empty distribution (finalize first if already initialized)
        if (lightInstDistribution_.isInitialized())
        {
            lightInstDistribution_.finalize();
        }
        // Don't initialize with nullptr - leave it uninitialized so integral() returns 0
        // The DiscreteDistribution1D class should handle uninitialized state properly
        lightDistributionDirty_ = false;
        LOG(DBUG) << "No emissive instances found - light distribution left uninitialized";
        return;
    }

    // Get engine and kernels through model handler
    if (!modelHandler_) {
        LOG(WARNING) << "Model handler not set";
        return;
    }
    
    // The model handler should have the engine set
    // We need to add a getter to access it - for now use a workaround
    // by calling computeLightProbabilities on the model handler
    // which will handle the kernel calls internally
    
    // Since we can't easily access the engine from here, let's simplify:
    // Just build a simple distribution for now
   // LOG(INFO) << "Building light instance distribution (simplified)";
    
    // Collect importances for all emissive instances
    std::vector<float> importances;
    importances.reserve(emissiveInstances_.size());
    
    for (uint32_t instIdx : emissiveInstances_)
    {
        float importance = 1.0f;  // Default importance
        
        // Get node and model to get actual importance if available
        auto it = nodeMap.find(instIdx);
        if (it != nodeMap.end() && !it->second.expired())
        {
            auto node = it->second.lock();
            auto model = modelHandler_->getModel(node);
            
            if (auto triModel = std::dynamic_pointer_cast<RiPRTriangleModel>(model))
            {
                // Get importance from the model's emitter distribution
                const LightDistribution& emitterDist = triModel->getEmitterPrimDistribution();
                importance = emitterDist.getIntengral();
                
                // If importance is still 0, use a default value
                if (importance == 0.0f)
                {
                    importance = 1.0f;
                }
                
                // Account for instance scaling
                const auto& spaceTime = node->getSpaceTime();
                float uniformScale = spaceTime.scale.norm() / std::sqrt(3.0f);
                importance *= uniformScale * uniformScale;  // Area scales by square
            }
        }
        
        importances.push_back(std::max(importance, 1e-6f));
    }
    
    // Build distribution (finalize first if already initialized)
    if (lightInstDistribution_.isInitialized())
    {
        lightInstDistribution_.finalize();
    }
    lightInstDistribution_.initialize(
        ctx->getCudaContext(),
        cudau::BufferType::Device,
        importances.data(),
        static_cast<uint32_t>(importances.size()));
    
    lightDistributionDirty_ = false;
   // LOG(INFO) << "Built light instance distribution with " << importances.size()
      //        << " lights, total importance: " << lightInstDistribution_.getIntengral();
}