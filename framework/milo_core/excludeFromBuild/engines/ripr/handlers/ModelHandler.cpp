#include "ModelHandler.h"
#include "../../../RenderContext.h"
#include "../../../common/common_host.h"  // For SlotFinder
#include "../../../material/DeviceDisneyMaterial.h"

std::unique_ptr<ModelHandler> ModelHandler::create(RenderContextPtr ctx)
{
    // Just create the handler, initialization will be done by RenderContext
    return std::make_unique<ModelHandler>();
}

ModelHandler::~ModelHandler()
{
    if (initialized_)
    {
        finalize();
    }
}

bool ModelHandler::initialize(RenderContextPtr ctx)
{
    if (initialized_)
    {
        LOG(WARNING) << "ModelHandler already initialized";
        return false;
    }
    
    if (!ctx)
    {
        LOG(WARNING) << "Invalid RenderContext provided to ModelHandler";
        return false;
    }
    
    ctx_ = ctx;
    
    // Initialize material management
    material_slot_finder_.initialize(maxNumMaterials);
    LOG(DBUG) << "ModelHandler material slot finder initialized for " << maxNumMaterials << " materials";
    
    // Initialize material data buffer
    CUcontext cuContext = ctx_->getCudaContext();
    material_data_buffer_.initialize(cuContext, cudau::BufferType::Device, maxNumMaterials);
    LOG(DBUG) << "ModelHandler material data buffer initialized for " << maxNumMaterials << " materials";
    
    initialized_ = true;
    LOG(INFO) << "ModelHandler initialized successfully";
    
    return true;
}

void ModelHandler::finalize()
{
    if (!initialized_)
        return;
    
    LOG(DBUG) << "ModelHandler finalizing...";
    
    // Clean up geometry cache
    if (!geometry_cache_.empty())
    {
        LOG(DBUG) << "Clearing " << geometry_cache_.size() << " cached geometries";
        geometry_cache_.clear();
    }
    
    // Clean up material buffers
    material_data_buffer_.finalize();
    material_slot_finder_.finalize();
    LOG(DBUG) << "Material management cleaned up";
    
    initialized_ = false;
    ctx_.reset();
    
    LOG(INFO) << "ModelHandler finalized";
}

bool ModelHandler::hasGeometry(size_t hash) const
{
    return geometry_cache_.find(hash) != geometry_cache_.end();
}

GeometryGroupResources* ModelHandler::getGeometry(size_t hash)
{
    auto it = geometry_cache_.find(hash);
    return (it != geometry_cache_.end()) ? &it->second : nullptr;
}

void ModelHandler::addGeometry(size_t hash, GeometryGroupResources&& resources)
{
    geometry_cache_[hash] = std::move(resources);
    LOG(DBUG) << "Added geometry to cache (hash: " << hash << ", total cached: " << geometry_cache_.size() << ")";
}

void ModelHandler::incrementRefCount(size_t hash)
{
    auto it = geometry_cache_.find(hash);
    if (it != geometry_cache_.end())
    {
        it->second.ref_count++;
        LOG(DBUG) << "Incremented ref count for geometry " << hash << " to " << it->second.ref_count;
    }
}

void ModelHandler::decrementRefCount(size_t hash)
{
    auto it = geometry_cache_.find(hash);
    if (it != geometry_cache_.end())
    {
        if (it->second.ref_count > 0)
        {
            it->second.ref_count--;
            LOG(DBUG) << "Decremented ref count for geometry " << hash << " to " << it->second.ref_count;
            
            // Clean up if no longer referenced
            if (it->second.ref_count == 0)
            {
                LOG(INFO) << "Removing unreferenced geometry from cache (hash: " << hash << ")";
                geometry_cache_.erase(it);
            }
        }
    }
}

uint32_t ModelHandler::allocateMaterialSlot()
{
    uint32_t slot = material_slot_finder_.getFirstAvailableSlot();
    if (slot >= maxNumMaterials)
    {
        LOG(WARNING) << "Material slots exhausted (max: " << maxNumMaterials << ")";
        return UINT32_MAX;
    }
    
    material_slot_finder_.setInUse(slot);
    LOG(DBUG) << "Allocated material slot " << slot;
    return slot;
}

void ModelHandler::freeMaterialSlot(uint32_t slot)
{
    if (slot < maxNumMaterials)
    {
        material_slot_finder_.setNotInUse(slot);
        LOG(DBUG) << "Freed material slot " << slot;
    }
}

size_t ModelHandler::computeGeometryHash(sabi::CgModelPtr cgModel)
{
    // Simple hash based on vertex count, triangle count, and a sample of vertex positions
    std::hash<size_t> hasher;
    size_t hash = hasher(cgModel->vertexCount());
    hash ^= hasher(cgModel->triangleCount()) << 1;
    
    // Sample a few vertices for better uniqueness
    if (cgModel->V.cols() > 0)
    {
        hash ^= hasher(cgModel->V(0, 0)) << 2;
        if (cgModel->V.cols() > 1)
            hash ^= hasher(cgModel->V(0, cgModel->V.cols() - 1)) << 3;
    }
    
    return hash;
}

bool ModelHandler::createGeometryGroup(sabi::CgModelPtr cgModel, size_t hash, GeometryGroupResources& resources)
{
    if (!cgModel || !cgModel->isValid())
    {
        LOG(WARNING) << "Invalid CgModel for geometry creation";
        return false;
    }
    
    try
    {
        CUcontext cuContext = ctx_->getCudaContext();
        optixu::Scene optixScene = ctx_->getScene();
        
        // Convert CgModel vertices to shared::Vertex format (shared by all surfaces)
        std::vector<shared::Vertex> vertices;
        vertices.reserve(cgModel->V.cols());
        
        for (int i = 0; i < cgModel->V.cols(); ++i)
        {
            shared::Vertex v;
            v.position = Point3D(cgModel->V(0, i), cgModel->V(1, i), cgModel->V(2, i));
            
            // Use normals if available, otherwise default to up vector
            if (cgModel->N.cols() > i)
            {
                v.normal = Normal3D(cgModel->N(0, i), cgModel->N(1, i), cgModel->N(2, i));
                v.normal = normalize(v.normal);
            }
            else
            {
                v.normal = Normal3D(0, 1, 0);
            }
            
            // Calculate tangent from normal
            Vector3D tangent, bitangent;
            float sign = v.normal.z >= 0 ? 1.0f : -1.0f;
            const float a = -1 / (sign + v.normal.z);
            const float b = v.normal.x * v.normal.y * a;
            tangent = Vector3D(1 + sign * v.normal.x * v.normal.x * a, sign * b, -sign * v.normal.x);
            v.texCoord0Dir = normalize(tangent);
            
            // Use texture coordinates if available
            if (cgModel->UV0.cols() > i)
            {
                v.texCoord = Point2D(cgModel->UV0(0, i), cgModel->UV0(1, i));
            }
            else
            {
                v.texCoord = Point2D(0, 0);
            }
            
            vertices.push_back(v);
            
            // Update AABB
            resources.aabb.unify(v.position);
        }
        
        // Create shared vertex buffer for all surfaces
        resources.vertex_buffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        
        // Create Geometry Acceleration Structure
        resources.gas = optixScene.createGeometryAccelerationStructure();
        
        // Check for emissive materials across all surfaces
        resources.is_emissive = false;
        
        // Create a GeometryInstance for each surface
        for (size_t surfIdx = 0; surfIdx < cgModel->S.size(); ++surfIdx)
        {
            const auto& surface = cgModel->S[surfIdx];
            
            // Check if this surface has emissive material
            if (surface.cgMaterial.emission.luminous > 0.0f)
            {
                resources.is_emissive = true;
            }
            
            // Convert surface triangles to shared::Triangle format
            std::vector<shared::Triangle> triangles;
            triangles.reserve(surface.F.cols());
            
            for (int i = 0; i < surface.F.cols(); ++i)
            {
                shared::Triangle tri;
                tri.index0 = surface.F(0, i);
                tri.index1 = surface.F(1, i);
                tri.index2 = surface.F(2, i);
                triangles.push_back(tri);
            }
            
            if (triangles.empty())
            {
                LOG(DBUG) << "Skipping surface " << surfIdx << " - no triangles";
                continue;
            }
            
            // Create per-surface resources
            GeometryInstanceResources geomInstRes;
            
            // Each surface gets its own triangle buffer
            geomInstRes.triangle_buffer.initialize(cuContext, cudau::BufferType::Device, triangles);
            
            // Allocate material slot for this surface
            geomInstRes.material_slot = allocateMaterialSlot();
            if (geomInstRes.material_slot == UINT32_MAX)
            {
                LOG(WARNING) << "Failed to allocate material slot for surface " << surfIdx;
                geomInstRes.material_slot = 0; // Fallback to default
            }
            
            // Create OptiX geometry instance for this surface
            geomInstRes.optix_geom_inst = optixScene.createGeometryInstance();
            geomInstRes.optix_geom_inst.setVertexBuffer(resources.vertex_buffer);  // Use shared vertex buffer
            geomInstRes.optix_geom_inst.setTriangleBuffer(geomInstRes.triangle_buffer);
            geomInstRes.optix_geom_inst.setNumMaterials(1, optixu::BufferView());
            geomInstRes.optix_geom_inst.setMaterial(0, 0, ctx_->getDefaultMaterial());  // Set default material
            geomInstRes.optix_geom_inst.setUserData(surfIdx);  // Store surface index
            
            // Add to GAS
            resources.gas.addChild(geomInstRes.optix_geom_inst);
            
            // Store in geometry group
            resources.geom_instances.push_back(std::move(geomInstRes));
            
            LOG(DBUG) << "Created GeometryInstance for surface " << surfIdx 
                      << " (" << triangles.size() << " triangles, material slot " 
                      << resources.geom_instances.back().material_slot << ")";
        }
        
        if (resources.geom_instances.empty())
        {
            LOG(WARNING) << "No valid surfaces found in CgModel";
            return false;
        }
        
        LOG(INFO) << "Creating GeometryGroup with " << resources.geom_instances.size() 
                  << " surfaces, " << vertices.size() << " vertices total"
                  << (resources.is_emissive ? " (EMISSIVE)" : "");
        
        // Configure and build GAS
        resources.gas.setNumMaterialSets(1);
        resources.gas.setNumRayTypes(0, 1);  // Single ray type for now
        resources.gas.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No);
        
        // Prepare and build GAS
        OptixAccelBufferSizes gasSizes;
        resources.gas.prepareForBuild(&gasSizes);
        
        resources.gas_mem.initialize(cuContext, cudau::BufferType::Device, gasSizes.outputSizeInBytes, 1);
        
        cudau::Buffer gasScratch;
        gasScratch.initialize(cuContext, cudau::BufferType::Device, gasSizes.tempSizeInBytes, 1);
        
        CUstream stream = ctx_->getCudaStream();
        resources.gas.rebuild(stream, resources.gas_mem, gasScratch);
        
        gasScratch.finalize();
        
        LOG(INFO) << "Created GAS with handle: " << resources.gas.getHandle();
        
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to create geometry resources: " << ex.what();
        return false;
    }
}

