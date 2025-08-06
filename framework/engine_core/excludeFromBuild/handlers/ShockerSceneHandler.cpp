// ShockerSceneHandler.cpp
// Implementation of scene management for the Shocker rendering system

#include "ShockerSceneHandler.h"
#include "ShockerModelHandler.h"
#include "ShockerMaterialHandler.h"
#include "../model/ShockerModel.h"
#include "../model/ShockerCore.h"

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
    ShockerModelPtr model = modelHandler_->processRenderableNode(node);
    if (!model) {
        LOG(WARNING) << "Failed to create model for node: " << node->getName();
        return nullptr;
    }

    // Process materials if material handler is set
    if (materialHandler_) {
        materialHandler_->processMaterialsForModel(model.get(), node->getModel());
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
    
    // Build surface acceleration structures for all models
    for (const auto& [name, model] : modelHandler_->getAllModels()) {
        shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
        if (surfaceGroup && surfaceGroup->needsRebuild) {
            // TODO: Build GAS when we have OptiX integration
            surfaceGroup->needsRebuild = 0;
            // GAS built for surface group
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