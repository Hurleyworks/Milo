#include "RiPRModelHandler.h"
#include "RiPRMaterialHandler.h"
#include "../../../RenderContext.h"
#include "../models/RiPRCore.h"


void RiPRModelHandler::initialize(RenderContextPtr context)
{
    renderContext_ = context;
    clear();
    
    // Initialize slot finders with reasonable capacity
    const uint32_t maxGeometryInstances = 10000;
    const uint32_t maxInstances = 100000;
    
    geomInstSlotFinder_.initialize(maxGeometryInstances);
    instanceSlotFinder_.initialize(maxInstances);
    
    // Initialize GPU buffer for geometry instance data
    if (renderContext_) {
        CUcontext cudaContext = renderContext_->getCudaContext();
        // Use mapped buffer following the sample pattern
        geometryInstanceDataBuffer_.initialize(
            cudaContext, 
            cudau::BufferType::Device,  // Still use Device type like the sample
            maxGeometryInstances);
    }
    
    // Handler initialized
}

RiPRModelPtr RiPRModelHandler::processRenderableNode(const sabi::RenderableNode& node)
{
    // Get the CgModel from the node
    sabi::CgModelPtr cgModel = node->getModel();
    if (!cgModel) {
        // Create phantom model for nodes without geometry
        RiPRModelPtr phantomModel = RiPRPhantomModel::create();
        phantomModel->createFromRenderableNode(node, geomInstSlotFinder_, renderContext_.get());
        models_[node->getName()] = phantomModel;
        return phantomModel;
    }
    
    // Create appropriate model type
    RiPRModelPtr model = createModelByType(cgModel);
    if (!model) {
        LOG(WARNING) << "Failed to create model for: " << node->getName();
        return nullptr;
    }
    
    // Create geometry from the renderable node
    // The model will internally allocate slots for each geometry instance
    model->createFromRenderableNode(node, geomInstSlotFinder_, renderContext_.get());
    
    // Process materials if we have a material handler
    if (materialHandler_) {
        // Get material folder from node description if available
        std::filesystem::path materialFolder;
        const auto& desc = node->description();
        if (!desc.modelPath.empty()) {
            materialFolder = desc.modelPath.parent_path();
        }
        
        // Process materials for the model
        materialHandler_->processMaterialsForModel(
            model.get(), 
            cgModel, 
            materialFolder);
    }
    
    // Store the model
    models_[node->getName()] = model;
    
    // Model created successfully (no logging needed for routine operations)
    
    return model;
}

RiPRModelPtr RiPRModelHandler::createModelByType(const sabi::CgModelPtr& cgModel)
{
    RiPRGeometryType type = determineGeometryType(cgModel);
    
    switch (type) {
        case RiPRGeometryType::Triangle:
            return RiPRTriangleModel::create();
            
        case RiPRGeometryType::Curve:
            // TODO: Implement RiPRCurveModel
            LOG(WARNING) << "Curve geometry not yet implemented";
            return nullptr;
            
        case RiPRGeometryType::TFDM:
            // TODO: Implement RiPRTFDMModel
            LOG(WARNING) << "TFDM geometry not yet implemented";
            return nullptr;
            
        case RiPRGeometryType::NRTDSM:
            // TODO: Implement RiPRNRTDSMModel
            LOG(WARNING) << "NRTDSM geometry not yet implemented";
            return nullptr;
            
        case RiPRGeometryType::Flyweight:
            return RiPRFlyweightModel::create();
            
        case RiPRGeometryType::Phantom:
            return RiPRPhantomModel::create();
            
        default:
            LOG(WARNING) << "Unknown geometry type";
            return nullptr;
    }
}

ripr::RiPRSurface* RiPRModelHandler::createRiPRSurface(RiPRModel* model)
{
    if (!model) {
        LOG(WARNING) << "Cannot create RiPRSurface from null model";
        return nullptr;
    }
    
    // Get surfaces from model to use
    const auto& modelSurfaces = model->getSurfaces();
    if (modelSurfaces.empty()) {
        LOG(WARNING) << "Model has no surfaces";
        return nullptr;
    }
    
    // Get the first surface from the model
    // Note: The model already owns the surfaces, we just return a pointer
    ripr::RiPRSurface* surface = modelSurfaces[0].get();
    
    // Notify area light handler about the new surface
    // Note: AreaLightHandler notifications are handled at scene level
    
    return surface;
}

ripr::RiPRSurfaceGroup* RiPRModelHandler::createRiPRSurfaceGroup(const std::vector<ripr::RiPRSurface*>& surfaces)
{
    if (surfaces.empty()) {
        LOG(WARNING) << "Cannot create RiPRSurfaceGroup from empty surfaces";
        return nullptr;
    }
    
    // Create new surface group
    auto group = std::make_unique<ripr::RiPRSurfaceGroup>();
    
    // Add all surfaces to the group
    for (ripr::RiPRSurface* surface : surfaces) {
        if (surface) {
            group->geomInsts.insert(surface);
        }
    }
    
    // Calculate combined AABB
    group->aabb = calculateCombinedAABB(surfaces);
    
    // Initialize other properties
    group->numEmitterPrimitives = 0;  // Will be calculated when materials are added
    group->needsReallocation = 0;
    group->needsRebuild = 1;  // Needs initial build
    group->refittable = 0;    // Static geometry by default
    
    // Store and return raw pointer (handler maintains ownership)
    ripr::RiPRSurfaceGroup* ptr = group.get();
    surfaceGroups_.push_back(std::move(group));
    
    LOG(DBUG) << "Created RiPRSurfaceGroup with " << surfaces.size() << " surfaces";
    
    return ptr;
}

ripr::RiPRNode* RiPRModelHandler::createRiPRNode(RiPRModel* model, const sabi::SpaceTime& spacetime)
{
    if (!model) {
        LOG(WARNING) << "Cannot create RiPRNode from null model";
        return nullptr;
    }
    
    // Get the model's existing surface group
    ripr::RiPRSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
    if (!surfaceGroup) {
        LOG(WARNING) << "Model has no surface group";
        return nullptr;
    }
    
    // Create new node
    auto node = std::make_unique<ripr::RiPRNode>();
    
    // Allocate instance slot
    uint32_t slot = allocateInstanceSlot();
    if (slot == SlotFinder::InvalidSlotIndex) {
        LOG(WARNING) << "Failed to allocate instance slot";
        return nullptr;
    }
    node->instSlot = slot;
    
    // Convert SpaceTime to Matrix4x4
    Matrix4x4 transform = RiPRModel::convertSpaceTimeToMatrix(spacetime);
    
    // Create surface group instance
    ripr::RiPRMesh::RiPRSurfaceGroupInstance groupInst;
    groupInst.geomGroup = surfaceGroup;
    groupInst.transform = transform;
    
    // Set up the node
    node->geomGroupInst = groupInst;
    node->matM2W = transform;
    node->nMatM2W = RiPRModel::calculateNormalMatrix(transform);
    node->prevMatM2W = node->matM2W;  // Initially same as current
    
    // Store and return raw pointer (handler maintains ownership)
    ripr::RiPRNode* ptr = node.get();
    nodes_.push_back(std::move(node));
    
    return ptr;
}

RiPRModelPtr RiPRModelHandler::getModel(const std::string& name) const
{
    auto it = models_.find(name);
    if (it != models_.end()) {
        return it->second;
    }
    return nullptr;
}

bool RiPRModelHandler::hasModel(const std::string& name) const
{
    return models_.find(name) != models_.end();
}

void RiPRModelHandler::clear()
{
    // Note: AreaLightHandler notifications are handled at scene level
    
    // Finalize all surfaces to properly destroy OptiX geometry instances
    for (auto& surface : surfaces_) {
        if (surface) {
            surface->finalize();
        }
    }
    
    // Finalize all nodes to properly destroy OptiX instances
    for (auto& node : nodes_) {
        if (node && node->optixInst) {
            node->optixInst.destroy();
        }
    }
    
    // Finalize all surface groups to properly destroy GAS and buffers
    for (auto& surfaceGroup : surfaceGroups_) {
        if (surfaceGroup) {
            if (surfaceGroup->optixGas) {
                surfaceGroup->optixGas.destroy();
            }
            if (surfaceGroup->optixGasMem.isInitialized()) {
                surfaceGroup->optixGasMem.finalize();
            }
        }
    }
    
    models_.clear();
    surfaces_.clear();
    surfaceGroups_.clear();
    nodes_.clear();
    geomInstSlotFinder_.reset();
    instanceSlotFinder_.reset();
    totalTriangles_ = 0;
    totalVertices_ = 0;
    
    // Finalize GPU buffer
    if (geometryInstanceDataBuffer_.isInitialized()) {
        geometryInstanceDataBuffer_.finalize();
    }
    
    // Handler cleared
}

RiPRGeometryType RiPRModelHandler::determineGeometryType(const sabi::CgModelPtr& model) const
{
    // For now, we only support triangle meshes
    // Future: Check for curves, displacement, etc.
    
    if (!model) {
        return RiPRGeometryType::Phantom;
    }
    
    if (model->V.cols() == 0) {
        return RiPRGeometryType::Flyweight;  // No geometry
    }
    
    // Check for particle/curve data
    // ParticleData is std::vector<Eigen::Vector4f>
    if (model->P.size() > 0) {
        return RiPRGeometryType::Curve;
    }
    
    // Check for displacement
    if (model->VD.cols() > 0) {
        // Could be TFDM or NRTDSM based on additional criteria
        return RiPRGeometryType::TFDM;
    }
    
    // Default to triangle mesh
    return RiPRGeometryType::Triangle;
}

AABB RiPRModelHandler::calculateCombinedAABB(const std::vector<ripr::RiPRSurface*>& surfaces) const
{
    AABB combined;
    combined.minP = Point3D(FLT_MAX, FLT_MAX, FLT_MAX);
    combined.maxP = Point3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const ripr::RiPRSurface* surface : surfaces) {
        if (surface) {
            combined.minP.x = std::min(combined.minP.x, surface->aabb.minP.x);
            combined.minP.y = std::min(combined.minP.y, surface->aabb.minP.y);
            combined.minP.z = std::min(combined.minP.z, surface->aabb.minP.z);
            
            combined.maxP.x = std::max(combined.maxP.x, surface->aabb.maxP.x);
            combined.maxP.y = std::max(combined.maxP.y, surface->aabb.maxP.y);
            combined.maxP.z = std::max(combined.maxP.z, surface->aabb.maxP.z);
        }
    }
    
    // Handle empty case
    if (surfaces.empty() || combined.minP.x > combined.maxP.x) {
        combined.minP = Point3D(0.0f, 0.0f, 0.0f);
        combined.maxP = Point3D(0.0f, 0.0f, 0.0f);
    }
    
    return combined;
}

size_t RiPRModelHandler::getRiPRSurfaceCount() const
{
    size_t totalCount = 0;
    for (const auto& [name, model] : models_) {
        if (model) {
            totalCount += model->getSurfaces().size();
        }
    }
    return totalCount;
}

void RiPRModelHandler::updateGeometryInstanceDataBuffer()
{
    if (!renderContext_) {
        LOG(WARNING) << "Cannot update geometry instance data buffer: no render context";
        return;
    }
    
    // Map the buffer for host access (following sample pattern)
    geometryInstanceDataBuffer_.map();
    ripr::RiPRSurfaceData* surfaceDataHost = geometryInstanceDataBuffer_.getMappedPointer();
    
    if (!surfaceDataHost) {
        LOG(WARNING) << "Failed to get mapped pointer for geometry instance data buffer";
        geometryInstanceDataBuffer_.unmap();
        return;
    }
    
    // Process all surfaces from all models
    size_t surfaceCount = 0;
    for (const auto& [name, model] : models_) {
        if (!model) continue;
        
        for (const auto& surface : model->getSurfaces()) {
            if (!surface) continue;
            
            uint32_t slot = surface->geomInstSlot;
            if (slot >= geometryInstanceDataBuffer_.numElements()) {
                LOG(WARNING) << "Surface slot " << slot << " exceeds buffer size";
                continue;
            }
            
            // Create a local RiPRSurfaceData and fill it
            ripr::RiPRSurfaceData surfaceData = {};
            
            // Set geometry buffers based on geometry type
            if (const TriangleGeometry* triGeom = std::get_if<TriangleGeometry>(&surface->geometry)) {
                surfaceData.vertexBuffer = triGeom->vertexBuffer.getROBuffer<shared::enableBufferOobCheck>();
                surfaceData.triangleBuffer = triGeom->triangleBuffer.getROBuffer<shared::enableBufferOobCheck>();
            } else if (const CurveGeometry* curveGeom = std::get_if<CurveGeometry>(&surface->geometry)) {
                // TODO: Handle curve geometry when implemented
                LOG(WARNING) << "Curve geometry not yet supported in geometry instance data buffer";
                continue;
            }
            
            // Set material slot
            if (surface->mat && materialHandler_) {
                // Find the material slot by checking each material in the handler
                const auto& allMaterials = materialHandler_->getAllMaterials();
                uint32_t matSlot = 0;
                bool found = false;
                
                for (size_t i = 0; i < allMaterials.size(); ++i) {
                    if (allMaterials[i].get() == surface->mat) {
                        matSlot = static_cast<uint32_t>(i);
                        found = true;
                        break;
                    }
                }
                
                if (found) {
                    surfaceData.disneyMaterialSlot = matSlot;
                    LOG(DBUG) << "Surface " << slot << " assigned material slot " << matSlot;
                } else {
                    surfaceData.disneyMaterialSlot = 0;
                    LOG(WARNING) << "Surface " << slot << " material not found in handler, using slot 0";
                }
            } else {
                surfaceData.disneyMaterialSlot = 0;
                LOG(WARNING) << "Surface " << slot << " has no material assigned, using slot 0";
            }
            
            // Set geometry instance slot
            surfaceData.geomInstSlot = slot;
            
            // TODO: Set emitter primitive distribution when area lights are supported
            // For now, leave emitterPrimDist as default
            
            // Write to the mapped buffer (following sample pattern)
            surfaceDataHost[slot] = surfaceData;
            surfaceCount++;
        }
    }
    
    // Unmap the buffer to sync with GPU
    geometryInstanceDataBuffer_.unmap();
    
    LOG(INFO) << "Updated geometry instance data buffer with " << surfaceCount << " surfaces";
}