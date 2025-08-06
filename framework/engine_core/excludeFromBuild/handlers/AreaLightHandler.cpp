#include "AreaLightHandler.h"
#include "ShockerSceneHandler.h"
#include "ShockerModelHandler.h"
#include "ShockerMaterialHandler.h"
#include "../model/ShockerCore.h"
#include "../common/common_host.h"
#include "../common/common_shared.h"
#include "../tools/PTXManager.h"


AreaLightHandler::~AreaLightHandler()
{
    finalize();
}

void AreaLightHandler::initialize(CUcontext context, uint32_t maxInstances)
{
    if (cuContext_)
    {
        LOG(WARNING) << "AreaLightHandler already initialized";
        return;
    }

    cuContext_ = context;
    maxInstances_ = maxInstances;

    // Initialize scene-wide distribution
    sceneAreaLightDist_.initialize(cuContext_, maxInstances_);

    // Allocate scratch buffers
    scratchImportanceBuffer_.initialize(cuContext_, cudau::BufferType::Device, 65536);
    scratchIndexBuffer_.initialize(cuContext_, cudau::BufferType::Device, 65536);

    // Load GPU kernels
    if (!loadKernels())
    {
        LOG(WARNING) << "Failed to load area light GPU kernels - CPU fallback will be used";
    }

    LOG(DBUG) << "AreaLightHandler initialized with max instances: " << maxInstances_;
}

void AreaLightHandler::finalize()
{
    if (!cuContext_)
    {
        return;
    }

    // Clean up all area lights
    for (auto* surface : areaLightState_.emissiveSurfaces)
    {
        cleanupAreaLight(surface);
    }

    // Finalize distributions
    sceneAreaLightDist_.finalize();

    // Clean up scratch buffers
    scratchImportanceBuffer_.finalize();
    scratchIndexBuffer_.finalize();

    // Unload kernels
    unloadKernels();

    // Clear state
    areaLightState_ = {};
    cuContext_ = nullptr;
    maxInstances_ = 0;

    LOG(DBUG) << "AreaLightHandler finalized";
}

void AreaLightHandler::setSceneHandler(ShockerSceneHandler* sceneHandler)
{
    sceneHandler_ = sceneHandler;
}

void AreaLightHandler::setModelHandler(ShockerModelHandler* modelHandler)
{
    modelHandler_ = modelHandler;
}

void AreaLightHandler::setMaterialHandler(ShockerMaterialHandler* materialHandler)
{
    materialHandler_ = materialHandler;
}

// Area Light Management

void AreaLightHandler::onMaterialAssigned(shocker::ShockerSurface* surface,
                                         DisneyMaterial* oldMaterial,
                                         DisneyMaterial* newMaterial)
{
    if (!surface)
    {
        return;
    }

    bool wasEmissive = oldMaterial && oldMaterial->emissive != nullptr;
    bool isEmissive = newMaterial && newMaterial->emissive != nullptr;

    if (wasEmissive != isEmissive)
    {
        // Emissive status changed
        if (isEmissive)
        {
            // Surface became emissive - track it immediately
            areaLightState_.emissiveSurfaces.insert(surface);
            areaLightState_.dirtySurfaces.insert(surface);
        }
        else
        {
            // Surface no longer emissive
            areaLightState_.emissiveSurfaces.erase(surface);
            areaLightState_.dirtySurfaces.erase(surface);
            cleanupAreaLight(surface);
        }
        
        // Find parent node and mark it dirty too
        if (auto* node = findNodeForSurface(surface))
        {
            areaLightState_.dirtyNodes.insert(node);
        }
        
        areaLightState_.sceneDistDirty = true;
        
        LOG(DBUG) << "Surface emissive status changed: " << (isEmissive ? "now emissive" : "no longer emissive");
    }
    else if (isEmissive)
    {
        // Still emissive but values might have changed
        areaLightState_.dirtySurfaces.insert(surface);
        
        LOG(DBUG) << "Emissive surface material updated";
    }
}

void AreaLightHandler::onSurfaceAdded(shocker::ShockerSurface* surface)
{
    if (!surface || !isAreaLight(surface))
    {
        return;
    }

    // Track as emissive surface immediately
    areaLightState_.emissiveSurfaces.insert(surface);
    areaLightState_.dirtySurfaces.insert(surface);
    areaLightState_.sceneDistDirty = true;
    
    LOG(DBUG) << "Area light surface added";
}

void AreaLightHandler::onSurfaceRemoved(shocker::ShockerSurface* surface)
{
    if (!surface)
    {
        return;
    }

    // Remove from tracking
    areaLightState_.emissiveSurfaces.erase(surface);
    areaLightState_.dirtySurfaces.erase(surface);
    
    // Clean up any distributions
    cleanupAreaLight(surface);
    
    areaLightState_.sceneDistDirty = true;
    
    LOG(DBUG) << "Surface removed from area lights";
}

void AreaLightHandler::onSurfaceGeometryChanged(shocker::ShockerSurface* surface)
{
    if (!surface || !isAreaLight(surface))
    {
        return;
    }

    areaLightState_.dirtySurfaces.insert(surface);
    
    if (auto* node = findNodeForSurface(surface))
    {
        areaLightState_.dirtyNodes.insert(node);
    }
    
    areaLightState_.sceneDistDirty = true;
    
    LOG(DBUG) << "Area light geometry changed";
}

void AreaLightHandler::onNodeAdded(shocker::ShockerNode* node)
{
    if (!node)
    {
        return;
    }

    // Check if this node contains any area lights
    bool hasAreaLight = false;
    if (node->geomGroupInst.geomGroup)
    {
        for (const auto* surface : node->geomGroupInst.geomGroup->geomInsts)
        {
            if (isAreaLight(surface))
            {
                hasAreaLight = true;
                areaLightState_.dirtySurfaces.insert(const_cast<ShockerSurface*>(surface));
            }
        }
    }

    if (hasAreaLight)
    {
        areaLightState_.dirtyNodes.insert(node);
        areaLightState_.sceneDistDirty = true;
        
        LOG(DBUG) << "Node with area lights added";
    }
}

void AreaLightHandler::onNodeRemoved(shocker::ShockerNode* node)
{
    if (!node)
    {
        return;
    }

    areaLightState_.nodesWithAreaLights.erase(node);
    areaLightState_.dirtyNodes.erase(node);
    areaLightState_.sceneDistDirty = true;
    
    LOG(DBUG) << "Node removed from area lights";
}

void AreaLightHandler::onNodeTransformChanged(shocker::ShockerNode* node)
{
    if (!node || areaLightState_.nodesWithAreaLights.find(node) == areaLightState_.nodesWithAreaLights.end())
    {
        return;
    }

    // Transform affects instance-level importance (due to scale)
    areaLightState_.sceneDistDirty = true;
    
    LOG(DBUG) << "Area light node transform changed";
}

// Distribution Updates

void AreaLightHandler::updateAreaLightDistributions(CUstream stream)
{
    if (!areaLightState_.isDirty())
    {
        return; // Nothing to update
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Level 1: Update primitive distributions in surfaces
    if (!areaLightState_.dirtySurfaces.empty())
    {
        updateSurfaceAreaLights(stream);
    }

    // Level 2: Update geometry instance distributions in nodes
    if (!areaLightState_.dirtyNodes.empty())
    {
        updateNodeAreaLights(stream);
    }

    // Level 3: Update scene-wide instance distribution
    if (areaLightState_.sceneDistDirty)
    {
        updateSceneAreaLights(stream);
    }

    // Clear dirty flags
    areaLightState_.clear();

    auto endTime = std::chrono::high_resolution_clock::now();
    lastUpdateTimeMs_ = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    if (lastUpdateTimeMs_ > 16.0f)
    {
        LOG(WARNING) << "Area light distribution update took " << lastUpdateTimeMs_ << "ms";
    }
}

void AreaLightHandler::forceUpdateAll(CUstream stream)
{
    // Mark everything dirty
    for (auto* surface : areaLightState_.emissiveSurfaces)
    {
        areaLightState_.dirtySurfaces.insert(surface);
    }
    
    for (auto* node : areaLightState_.nodesWithAreaLights)
    {
        areaLightState_.dirtyNodes.insert(node);
    }
    
    areaLightState_.sceneDistDirty = true;

    // Update
    updateAreaLightDistributions(stream);
}

// Queries

bool AreaLightHandler::hasAreaLights() const
{
    return !areaLightState_.emissiveSurfaces.empty();
}

const engine::LightDistribution& AreaLightHandler::getSceneAreaLightDistribution() const
{
    return sceneAreaLightDist_;
}

uint32_t AreaLightHandler::getNumAreaLights() const
{
    return static_cast<uint32_t>(areaLightState_.emissiveSurfaces.size());
}

uint32_t AreaLightHandler::getNumEmissiveTriangles() const
{
    // This is calculated during updateAreaLightDistributions
    // For testing, we'll need to accept this is 0 until update is called
    return areaLightState_.totalEmissiveTriangles;
}

float AreaLightHandler::getTotalAreaLightPower() const
{
    return areaLightState_.totalPower;
}

float AreaLightHandler::getLastUpdateTimeMs() const
{
    return lastUpdateTimeMs_;
}

bool AreaLightHandler::isDirty() const
{
    return areaLightState_.isDirty();
}

// Private Implementation

void AreaLightHandler::updateSurfaceAreaLights(CUstream stream)
{
    for (auto* surface : areaLightState_.dirtySurfaces)
    {
        if (isAreaLight(surface))
        {
            initializeOrUpdateAreaLight(surface, stream);
            areaLightState_.emissiveSurfaces.insert(surface);
        }
        else
        {
            cleanupAreaLight(surface);
            areaLightState_.emissiveSurfaces.erase(surface);
        }
    }
}

void AreaLightHandler::updateNodeAreaLights(CUstream stream)
{
    for (auto* node : areaLightState_.dirtyNodes)
    {
        bool hasAreaLight = false;
        
        // Check if this node has any area lights
        if (node->geomGroupInst.geomGroup)
        {
            for (const auto* surface : node->geomGroupInst.geomGroup->geomInsts)
            {
                if (isAreaLight(surface))
                {
                    hasAreaLight = true;
                    break;
                }
            }
        }

        if (hasAreaLight)
        {
            // Initialize node's light distribution if needed
            // Initialize node's light distribution if needed
            // Note: LightDistribution initialization is handled separately

            // Update the distribution
            launchComputeGeomInstImportanceKernel(node, stream);
            
            areaLightState_.nodesWithAreaLights.insert(node);
        }
        else
        {
            // Clean up distribution if no longer needed
            if (node->lightGeomInstDist.isInitialized())
            {
                node->lightGeomInstDist.finalize();
            }
            
            areaLightState_.nodesWithAreaLights.erase(node);
        }
    }
}

void AreaLightHandler::updateSceneAreaLights(CUstream stream)
{
    if (areaLightState_.nodesWithAreaLights.empty())
    {
        // No area lights in scene
        if (sceneAreaLightDist_.isInitialized())
        {
            sceneAreaLightDist_.finalize();
            sceneAreaLightDist_.initialize(cuContext_, maxInstances_);
        }
        return;
    }

    // Compute importance for all instances with area lights
    launchComputeInstanceImportanceKernel(stream);

    // Build probability texture
    launchBuildProbabilityTextureKernel(sceneAreaLightDist_, stream);

    // Update statistics
    areaLightState_.totalEmissiveTriangles = 0;
    areaLightState_.totalPower = 0.0f;
    
    for (auto* surface : areaLightState_.emissiveSurfaces)
    {
        if (auto* triGeom = std::get_if<TriangleGeometry>(&surface->geometry))
        {
            areaLightState_.totalEmissiveTriangles += triGeom->triangleBuffer.numElements();
            
            if (surface->mat && surface->mat->emissive)
            {
                // For now, assume uniform emissive strength
                // TODO: Sample the emissive texture/array to get actual values
                float power = 1.0f; // Default emissive power
                if (surface->mat->emissiveStrength) {
                    // TODO: Sample emissiveStrength array for actual value
                    power = 1.0f;
                }
                areaLightState_.totalPower += power * triGeom->triangleBuffer.numElements();
            }
        }
    }
}

bool AreaLightHandler::isAreaLight(const shocker::ShockerSurface* surface) const
{
    return surface && surface->mat && surface->mat->emissive != nullptr;
}

shocker::ShockerNode* AreaLightHandler::findNodeForSurface(shocker::ShockerSurface* surface) const
{
    if (!sceneHandler_)
    {
        return nullptr;
    }

    return sceneHandler_->findNodeForSurface(surface);
}

void AreaLightHandler::initializeOrUpdateAreaLight(shocker::ShockerSurface* surface, CUstream stream)
{
    if (auto* triGeom = std::get_if<TriangleGeometry>(&surface->geometry))
    {
        bool needsDist = isAreaLight(surface);
        bool hasDist = triGeom->emitterPrimDist.isInitialized();

        if (needsDist && !hasDist)
        {
            // Initialize new distribution with uniform weights
            uint32_t numTriangles = triGeom->triangleBuffer.numElements();
            std::vector<float> uniformWeights(numTriangles, 1.0f);
            triGeom->emitterPrimDist.initialize(
                cuContext_, 
                cudau::BufferType::Device,
                uniformWeights.data(),
                numTriangles);
        }
        else if (!needsDist && hasDist)
        {
            // Destroy unneeded distribution
            triGeom->emitterPrimDist.finalize();
        }
        else if (needsDist && hasDist)
        {
            // Update existing distribution
            launchComputeTriangleImportanceKernel(surface, stream);
        }
    }
}

void AreaLightHandler::cleanupAreaLight(shocker::ShockerSurface* surface)
{
    if (auto* triGeom = std::get_if<TriangleGeometry>(&surface->geometry))
    {
        if (triGeom->emitterPrimDist.isInitialized())
        {
            triGeom->emitterPrimDist.finalize();
        }
    }
}

// GPU kernel dispatch methods
void AreaLightHandler::launchComputeTriangleImportanceKernel(shocker::ShockerSurface* surface, CUstream stream)
{
    // TODO: Implement GPU kernel dispatch
    // For now, use CPU fallback
    if (auto* triGeom = std::get_if<TriangleGeometry>(&surface->geometry))
    {
        uint32_t numTriangles = triGeom->triangleBuffer.numElements();
        std::vector<float> importances(numTriangles, 1.0f); // Uniform for now
        
        // In a real implementation, compute actual importance based on:
        // - Triangle area
        // - Emittance magnitude
        // - Normal orientation
        
        // Re-initialize distribution with new importance values
        triGeom->emitterPrimDist.finalize();
        triGeom->emitterPrimDist.initialize(
            cuContext_,
            cudau::BufferType::Device,
            importances.data(),
            numTriangles);
    }
}

void AreaLightHandler::launchComputeGeomInstImportanceKernel(shocker::ShockerNode* node, CUstream stream)
{
    // TODO: Implement GPU kernel dispatch
    // For now, use CPU fallback
    if (node->geomGroupInst.geomGroup)
    {
        std::vector<float> importances;
        
        for (const auto* surface : node->geomGroupInst.geomGroup->geomInsts)
        {
            float importance = 0.0f;
            
            if (isAreaLight(surface))
            {
                if (auto* triGeom = std::get_if<TriangleGeometry>(&surface->geometry))
                {
                    // Sum of all triangle importances in this surface
                    importance = static_cast<float>(triGeom->triangleBuffer.numElements());
                }
            }
            
            importances.push_back(importance);
        }
        
        // Note: Node light distribution handled by LightDistribution class
    }
}

void AreaLightHandler::launchComputeInstanceImportanceKernel(CUstream stream)
{
    // TODO: Implement GPU kernel dispatch
    // For now, use CPU fallback
    uint32_t numActiveInstances = static_cast<uint32_t>(areaLightState_.nodesWithAreaLights.size());
    if (numActiveInstances == 0)
    {
        return;
    }
    
    std::vector<float> importances(numActiveInstances, 0.0f);
    
    uint32_t idx = 0;
    for (auto* node : areaLightState_.nodesWithAreaLights)
    {
        if (idx >= numActiveInstances)
        {
            break;
        }
        
        // Instance importance is sum of all geometry instance importances
        // scaled by the transform's scale factor
        float importance = 1.0f; // TODO: Extract scale from node->matM2W
        importances[idx] = importance;
        idx++;
    }
    
    // Only pass the actual number of active instances, not the maximum capacity
    sceneAreaLightDist_.setWeights(importances.data(), numActiveInstances, stream);
}

void AreaLightHandler::launchBuildProbabilityTextureKernel(engine::LightDistribution& dist, CUstream stream)
{
    dist.buildProbabilityTexture(stream);
}

bool AreaLightHandler::loadKernels()
{
    // TODO: Load PTX kernels for GPU computation
    // For now, return false to use CPU fallback
    return false;
}

void AreaLightHandler::unloadKernels()
{
    // TODO: Unload PTX kernels
    kernels_ = {};
}

