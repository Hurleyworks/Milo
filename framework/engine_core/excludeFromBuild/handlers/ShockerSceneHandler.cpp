// ShockerSceneHandler.cpp
// Implementation of scene management for the Shocker rendering system

#include "ShockerSceneHandler.h"
#include "ShockerModelHandler.h"
#include "ShockerMaterialHandler.h"
#include "../model/ShockerModel.h"

ShockerSceneHandler::ShockerSceneHandler(RenderContextPtr ctx)
    : ctx_(ctx)
{
    // Constructor - no logging needed
}

ShockerSceneHandler::~ShockerSceneHandler()
{
    // Destructor - no logging needed
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
    
    // Reserve space for instances
    instances_.reserve(1000); // Start with reasonable capacity
    
    isInitialized_ = true;
    LOG(INFO) << "ShockerSceneHandler initialized with capacity for " << MaxNumInstances << " instances";
}

Instance* ShockerSceneHandler::createInstance(RenderableWeakRef& weakNode)
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
    ShockerModelPtr model = modelHandler_->processRenderableNode(node);
    if (!model) {
        LOG(WARNING) << "Failed to create model for node: " << node->getName();
        return nullptr;
    }

    // Process materials if material handler is set
    if (materialHandler_) {
        materialHandler_->processMaterialsForModel(model.get(), node->getModel());
    }

    // Create instance from the model
    Instance* instance = modelHandler_->createInstance(model.get(), node->getSpaceTime());
    if (!instance) {
        LOG(WARNING) << "Failed to create instance for node: " << node->getName();
        return nullptr;
    }

    // Store instance
    instances_.push_back(instance);
    
    // Map instance to node
    nodeMap_[instance->instSlot] = weakNode;
    
    // Successfully created instance (no logging needed for routine operations)
    
    return instance;
}

void ShockerSceneHandler::createInstanceList(const WeakRenderableList& weakNodeList)
{
    LOG(INFO) << "Creating instances for " << weakNodeList.size() << " nodes";
    
    size_t successCount = 0;
    size_t failCount = 0;
    
    for (const auto& weakNode : weakNodeList) {
        // Create a copy to pass as non-const reference
        RenderableWeakRef weakNodeCopy = weakNode;
        Instance* instance = createInstance(weakNodeCopy);
        if (instance) {
            successCount++;
        } else {
            failCount++;
        }
    }
    
    LOG(INFO) << "Created " << successCount << " instances successfully";
    if (failCount > 0) {
        LOG(WARNING) << "Failed to create " << failCount << " instances";
    }
}

void ShockerSceneHandler::processRenderableNode(RenderableNode& node)
{
    if (!isInitialized_) {
        initialize();
    }

    // Create weak reference
    RenderableWeakRef weakNode = node;
    
    // Create instance
    createInstance(weakNode);
}

void ShockerSceneHandler::clear()
{
    // Clear instances (they're owned by model handler)
    instances_.clear();
    
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

Instance* ShockerSceneHandler::getInstance(uint32_t index) const
{
    if (index >= instances_.size()) {
        return nullptr;
    }
    return instances_[index];
}

RenderableWeakRef ShockerSceneHandler::getNode(uint32_t instanceIndex) const
{
    auto it = nodeMap_.find(instanceIndex);
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

    LOG(INFO) << "Building acceleration structures for " << instances_.size() << " instances";
    
    // Build geometry acceleration structures for all models
    for (const auto& model : modelHandler_->getAllModels()) {
        GeometryGroup* geomGroup = model.second->getGeometryGroup();
        if (geomGroup && geomGroup->needsRebuild) {
            // TODO: Build GAS when we have OptiX integration
            geomGroup->needsRebuild = 0;
            // GAS built for model
        }
    }
    
    // TODO: Build instance acceleration structure (IAS) when we have OptiX integration
    
    LOG(INFO) << "Acceleration structures built";
}

void ShockerSceneHandler::updateAccelerationStructures()
{
    if (!modelHandler_) {
        LOG(WARNING) << "Cannot update acceleration structures: model handler not set";
        return;
    }

    // Updating acceleration structures
    
    // Update any refittable geometry acceleration structures
    for (const auto& model : modelHandler_->getAllModels()) {
        GeometryGroup* geomGroup = model.second->getGeometryGroup();
        if (geomGroup && geomGroup->refittable) {
            // TODO: Refit GAS when we have OptiX integration
            // GAS refitted for model
        }
    }
    
    // TODO: Update instance acceleration structure (IAS) when we have OptiX integration
    
    // Acceleration structures updated
}

size_t ShockerSceneHandler::getGeometryInstanceCount() const
{
    if (!modelHandler_) {
        return 0;
    }
    return modelHandler_->getGeometryInstanceCount();
}

size_t ShockerSceneHandler::getMaterialCount() const
{
    if (!materialHandler_) {
        return 0;
    }
    return materialHandler_->getAllMaterials().size();
}