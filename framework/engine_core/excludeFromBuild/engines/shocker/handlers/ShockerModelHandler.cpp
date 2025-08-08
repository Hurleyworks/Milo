#include "ShockerModelHandler.h"
#include "ShockerMaterialHandler.h"
#include "../../../RenderContext.h"
#include "../models/ShockerCore.h"


void ShockerModelHandler::initialize(RenderContextPtr context)
{
    renderContext_ = context;
    clear();
    
    // Initialize slot finders with reasonable capacity
    const uint32_t maxGeometryInstances = 10000;
    const uint32_t maxInstances = 100000;
    
    geomInstSlotFinder_.initialize(maxGeometryInstances);
    instanceSlotFinder_.initialize(maxInstances);
    
    // Handler initialized
}

ShockerModelPtr ShockerModelHandler::processRenderableNode(const sabi::RenderableNode& node)
{
    // Get the CgModel from the node
    sabi::CgModelPtr cgModel = node->getModel();
    if (!cgModel) {
        // Create phantom model for nodes without geometry
        ShockerModelPtr phantomModel = ShockerPhantomModel::create();
        phantomModel->createFromRenderableNode(node, geomInstSlotFinder_);
        models_[node->getName()] = phantomModel;
        return phantomModel;
    }
    
    // Create appropriate model type
    ShockerModelPtr model = createModelByType(cgModel);
    if (!model) {
        LOG(WARNING) << "Failed to create model for: " << node->getName();
        return nullptr;
    }
    
    // Create geometry from the renderable node
    // The model will internally allocate slots for each geometry instance
    model->createFromRenderableNode(node, geomInstSlotFinder_);
    
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

ShockerModelPtr ShockerModelHandler::createModelByType(const sabi::CgModelPtr& cgModel)
{
    ShockerGeometryType type = determineGeometryType(cgModel);
    
    switch (type) {
        case ShockerGeometryType::Triangle:
            return ShockerTriangleModel::create();
            
        case ShockerGeometryType::Curve:
            // TODO: Implement ShockerCurveModel
            LOG(WARNING) << "Curve geometry not yet implemented";
            return nullptr;
            
        case ShockerGeometryType::TFDM:
            // TODO: Implement ShockerTFDMModel
            LOG(WARNING) << "TFDM geometry not yet implemented";
            return nullptr;
            
        case ShockerGeometryType::NRTDSM:
            // TODO: Implement ShockerNRTDSMModel
            LOG(WARNING) << "NRTDSM geometry not yet implemented";
            return nullptr;
            
        case ShockerGeometryType::Flyweight:
            return ShockerFlyweightModel::create();
            
        case ShockerGeometryType::Phantom:
            return ShockerPhantomModel::create();
            
        default:
            LOG(WARNING) << "Unknown geometry type";
            return nullptr;
    }
}

shocker::ShockerSurface* ShockerModelHandler::createShockerSurface(ShockerModel* model)
{
    if (!model) {
        LOG(WARNING) << "Cannot create ShockerSurface from null model";
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
    shocker::ShockerSurface* surface = modelSurfaces[0].get();
    
    // Notify area light handler about the new surface
    // Note: AreaLightHandler notifications are handled at scene level
    
    return surface;
}

shocker::ShockerSurfaceGroup* ShockerModelHandler::createShockerSurfaceGroup(const std::vector<shocker::ShockerSurface*>& surfaces)
{
    if (surfaces.empty()) {
        LOG(WARNING) << "Cannot create ShockerSurfaceGroup from empty surfaces";
        return nullptr;
    }
    
    // Create new surface group
    auto group = std::make_unique<shocker::ShockerSurfaceGroup>();
    
    // Add all surfaces to the group
    for (shocker::ShockerSurface* surface : surfaces) {
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
    shocker::ShockerSurfaceGroup* ptr = group.get();
    surfaceGroups_.push_back(std::move(group));
    
    LOG(DBUG) << "Created ShockerSurfaceGroup with " << surfaces.size() << " surfaces";
    
    return ptr;
}

shocker::ShockerNode* ShockerModelHandler::createShockerNode(ShockerModel* model, const sabi::SpaceTime& spacetime)
{
    if (!model) {
        LOG(WARNING) << "Cannot create ShockerNode from null model";
        return nullptr;
    }
    
    // Get the model's existing surface group
    shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
    if (!surfaceGroup) {
        LOG(WARNING) << "Model has no surface group";
        return nullptr;
    }
    
    // Create new node
    auto node = std::make_unique<shocker::ShockerNode>();
    
    // Allocate instance slot
    uint32_t slot = allocateInstanceSlot();
    if (slot == SlotFinder::InvalidSlotIndex) {
        LOG(WARNING) << "Failed to allocate instance slot";
        return nullptr;
    }
    node->instSlot = slot;
    
    // Convert SpaceTime to Matrix4x4
    Matrix4x4 transform = ShockerModel::convertSpaceTimeToMatrix(spacetime);
    
    // Create surface group instance
    shocker::ShockerMesh::ShockerSurfaceGroupInstance groupInst;
    groupInst.geomGroup = surfaceGroup;
    groupInst.transform = transform;
    
    // Set up the node
    node->geomGroupInst = groupInst;
    node->matM2W = transform;
    node->nMatM2W = ShockerModel::calculateNormalMatrix(transform);
    node->prevMatM2W = node->matM2W;  // Initially same as current
    
    // Store and return raw pointer (handler maintains ownership)
    shocker::ShockerNode* ptr = node.get();
    nodes_.push_back(std::move(node));
    
    return ptr;
}

ShockerModelPtr ShockerModelHandler::getModel(const std::string& name) const
{
    auto it = models_.find(name);
    if (it != models_.end()) {
        return it->second;
    }
    return nullptr;
}

bool ShockerModelHandler::hasModel(const std::string& name) const
{
    return models_.find(name) != models_.end();
}

void ShockerModelHandler::clear()
{
    // Note: AreaLightHandler notifications are handled at scene level
    
    models_.clear();
    surfaces_.clear();
    surfaceGroups_.clear();
    nodes_.clear();
    geomInstSlotFinder_.reset();
    instanceSlotFinder_.reset();
    totalTriangles_ = 0;
    totalVertices_ = 0;
    
    // Handler cleared
}

ShockerGeometryType ShockerModelHandler::determineGeometryType(const sabi::CgModelPtr& model) const
{
    // For now, we only support triangle meshes
    // Future: Check for curves, displacement, etc.
    
    if (!model) {
        return ShockerGeometryType::Phantom;
    }
    
    if (model->V.cols() == 0) {
        return ShockerGeometryType::Flyweight;  // No geometry
    }
    
    // Check for particle/curve data
    // ParticleData is std::vector<Eigen::Vector4f>
    if (model->P.size() > 0) {
        return ShockerGeometryType::Curve;
    }
    
    // Check for displacement
    if (model->VD.cols() > 0) {
        // Could be TFDM or NRTDSM based on additional criteria
        return ShockerGeometryType::TFDM;
    }
    
    // Default to triangle mesh
    return ShockerGeometryType::Triangle;
}

AABB ShockerModelHandler::calculateCombinedAABB(const std::vector<shocker::ShockerSurface*>& surfaces) const
{
    AABB combined;
    combined.minP = Point3D(FLT_MAX, FLT_MAX, FLT_MAX);
    combined.maxP = Point3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const shocker::ShockerSurface* surface : surfaces) {
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

size_t ShockerModelHandler::getShockerSurfaceCount() const
{
    size_t totalCount = 0;
    for (const auto& [name, model] : models_) {
        if (model) {
            totalCount += model->getSurfaces().size();
        }
    }
    return totalCount;
}