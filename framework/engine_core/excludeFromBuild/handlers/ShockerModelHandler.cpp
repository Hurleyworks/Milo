#include "ShockerModelHandler.h"
#include "ShockerMaterialHandler.h"
#include "../RenderContext.h"

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
    
    // Update statistics - count geometry instances
    const auto& geomInstances = model->getGeometryInstances();
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

GeometryInstance* ShockerModelHandler::createGeometryInstance(ShockerModel* model)
{
    if (!model) {
        LOG(WARNING) << "Cannot create GeometryInstance from null model";
        return nullptr;
    }
    
    // The model already creates and owns its geometry instances
    // This method now returns the first one for compatibility
    const auto& instances = model->getGeometryInstances();
    if (instances.empty()) {
        LOG(WARNING) << "Model has no geometry instances";
        return nullptr;
    }
    
    // Return pointer to first instance
    // Callers should use model->getGeometryInstances() for all instances
    return instances[0].get();
}

GeometryGroup* ShockerModelHandler::createGeometryGroup(const std::vector<GeometryInstance*>& instances)
{
    if (instances.empty()) {
        LOG(WARNING) << "Cannot create GeometryGroup from empty instances";
        return nullptr;
    }
    
    // Create new geometry group
    auto group = std::make_unique<GeometryGroup>();
    
    // Add all instances to the group
    for (GeometryInstance* inst : instances) {
        if (inst) {
            group->geomInsts.insert(inst);
        }
    }
    
    // Calculate combined AABB
    group->aabb = calculateCombinedAABB(instances);
    
    // Initialize other properties
    group->numEmitterPrimitives = 0;  // Will be calculated when materials are added
    group->needsReallocation = 0;
    group->needsRebuild = 1;  // Needs initial build
    group->refittable = 0;    // Static geometry by default
    
    // Store and return raw pointer (handler maintains ownership)
    GeometryGroup* ptr = group.get();
    geometryGroups_.push_back(std::move(group));
    
    LOG(DBUG) << "Created GeometryGroup with " << instances.size() << " instances";
    
    return ptr;
}

Instance* ShockerModelHandler::createInstance(ShockerModel* model, const sabi::SpaceTime& spacetime)
{
    if (!model) {
        LOG(WARNING) << "Cannot create Instance from null model";
        return nullptr;
    }
    
    // Get the model's geometry group
    GeometryGroup* geomGroup = model->getGeometryGroup();
    if (!geomGroup) {
        LOG(WARNING) << "Model has no geometry group";
        return nullptr;
    }
    
    // Create new instance
    auto inst = std::make_unique<Instance>();
    
    // Allocate instance slot
    uint32_t slot = allocateInstanceSlot();
    if (slot == SlotFinder::InvalidSlotIndex) {
        LOG(WARNING) << "Failed to allocate instance slot";
        return nullptr;
    }
    inst->instSlot = slot;
    
    // Convert SpaceTime to Matrix4x4
    Matrix4x4 transform = ShockerModel::convertSpaceTimeToMatrix(spacetime);
    
    // Create geometry group instance
    Mesh::GeometryGroupInstance groupInst;
    groupInst.geomGroup = geomGroup;
    groupInst.transform = transform;
    
    // Set up the instance
    inst->geomGroupInst = groupInst;
    inst->matM2W = transform;
    inst->nMatM2W = ShockerModel::calculateNormalMatrix(transform);
    inst->prevMatM2W = inst->matM2W;  // Initially same as current
    
    // Instance created successfully (no logging needed for routine operations)
    
    // Store and return raw pointer (handler maintains ownership)
    Instance* ptr = inst.get();
    instances_.push_back(std::move(inst));
    
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
    models_.clear();
    geometryInstances_.clear();
    geometryGroups_.clear();
    instances_.clear();
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

AABB ShockerModelHandler::calculateCombinedAABB(const std::vector<GeometryInstance*>& instances) const
{
    AABB combined;
    combined.minP = Point3D(FLT_MAX, FLT_MAX, FLT_MAX);
    combined.maxP = Point3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const GeometryInstance* inst : instances) {
        if (inst) {
            combined.minP.x = std::min(combined.minP.x, inst->aabb.minP.x);
            combined.minP.y = std::min(combined.minP.y, inst->aabb.minP.y);
            combined.minP.z = std::min(combined.minP.z, inst->aabb.minP.z);
            
            combined.maxP.x = std::max(combined.maxP.x, inst->aabb.maxP.x);
            combined.maxP.y = std::max(combined.maxP.y, inst->aabb.maxP.y);
            combined.maxP.z = std::max(combined.maxP.z, inst->aabb.maxP.z);
        }
    }
    
    // Handle empty case
    if (instances.empty() || combined.minP.x > combined.maxP.x) {
        combined.minP = Point3D(0.0f, 0.0f, 0.0f);
        combined.maxP = Point3D(0.0f, 0.0f, 0.0f);
    }
    
    return combined;
}

size_t ShockerModelHandler::getGeometryInstanceCount() const
{
    size_t totalCount = 0;
    for (const auto& [name, model] : models_) {
        if (model) {
            totalCount += model->getGeometryInstances().size();
        }
    }
    return totalCount;
}