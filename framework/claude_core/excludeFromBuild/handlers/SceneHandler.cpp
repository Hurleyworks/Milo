#include "SceneHandler.h"

SceneHandler::SceneHandler(RenderContextPtr ctx)
    : renderContext_(ctx)
{
    // Scene will be initialized in initialize() method
}

SceneHandler::~SceneHandler()
{
    finalize();
}

void SceneHandler::initialize()
{
    if (!renderContext_)
    {
        LOG(WARNING) << "SceneHandler::initialize() called with null RenderContext";
        return;
    }

    // Check if already initialized
    if (ias_)
    {
        LOG(DBUG) << "SceneHandler already initialized";
        return;
    }

    LOG(DBUG) << "SceneHandler lazy initialization triggered";
    
    // Initialize the OptiX scene
    scene_ = renderContext_->getScene();
    
    // Create an empty IAS
    ias_ = scene_.createInstanceAccelerationStructure();
    
    // Generate initial scene SBT layout to prevent "SBT layout generation has not been done" error
    size_t dummySize;
    scene_.generateShaderBindingTableLayout(&dummySize);
    
    // Configure the IAS with default settings (fast trace, no update)
    setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false);
    
    isDirty_ = true;
    needsRebuild_ = true;
    hasBeenBuilt_ = false;
    traversableHandle_ = 0;
}

void SceneHandler::finalize()
{
    // Clear instances from IAS
    if (ias_)
    {
        ias_.clearChildren();
    }
    
    // Clear local instance vector
    instances_.clear();
    
    // Reset traversable handle
    traversableHandle_ = 0;
    hasBeenBuilt_ = false;
    needsRebuild_ = true;
    
    // Clean up buffers
    if (optixInstanceBuffer_.isInitialized())
        optixInstanceBuffer_.finalize();
    if (accelBuffer_.isInitialized())
        accelBuffer_.finalize();
    // Note: scratch buffer is now shared and managed by RenderContext
    
    // Destroy IAS
    if (ias_)
    {
        ias_.destroy();
    }
}

void SceneHandler::setConfiguration(
    optixu::ASTradeoff tradeoff,
    bool allowUpdate,
    bool allowCompaction)
{
    if (!ias_)
    {
        LOG(WARNING) << "SceneHandler::setConfiguration() called with null IAS";
        return;
    }
    
    allowUpdate_ = allowUpdate;
    
    ias_.setConfiguration(
        tradeoff,
        allowUpdate ? optixu::AllowUpdate::Yes : optixu::AllowUpdate::No,
        allowCompaction ? optixu::AllowCompaction::Yes : optixu::AllowCompaction::No,
        optixu::AllowRandomInstanceAccess::No);
    
    // Configuration change requires rebuild
    isDirty_ = true;
    needsRebuild_ = true;
}

void SceneHandler::buildIAS()
{
    if (!isDirty_)
    {
        return; // No changes, no need to rebuild
    }

    if (instances_.empty())
    {
        // Empty scene - set traversable handle to 0
        traversableHandle_ = 0;
        isDirty_ = false;
        return;
    }

    // Note: Children are already added via addInstance/removeInstanceAt
    // No need to re-add them here
    
    // Prepare for build
    OptixAccelBufferSizes memReq;
    ias_.prepareForBuild(&memReq);
    
    // Reallocate instance buffer if needed
    if (!optixInstanceBuffer_.isInitialized() || optixInstanceBuffer_.numElements() < instances_.size())
    {
        if (optixInstanceBuffer_.isInitialized())
            optixInstanceBuffer_.finalize();
        optixInstanceBuffer_.initialize(
            renderContext_->getCudaContext(),
            cudau::BufferType::Device,
            instances_.size());
    }
    
    if (!accelBuffer_.isInitialized() || accelBuffer_.sizeInBytes() < memReq.outputSizeInBytes)
    {
        if (accelBuffer_.isInitialized())
            accelBuffer_.finalize();
        accelBuffer_.initialize(
            renderContext_->getCudaContext(),
            cudau::BufferType::Device,
            memReq.outputSizeInBytes,
            1);
    }
    
    // Use shared scratch buffer from RenderContext, resize if needed
    cudau::Buffer& scratchBuffer = renderContext_->getASScratchBuffer();
    if (!scratchBuffer.isInitialized() || scratchBuffer.sizeInBytes() < memReq.tempSizeInBytes)
    {
        if (scratchBuffer.isInitialized())
            scratchBuffer.finalize();
        scratchBuffer.initialize(
            renderContext_->getCudaContext(),
            cudau::BufferType::Device,
            memReq.tempSizeInBytes,
            1);
    }
    
    // Regenerate SBT layout BEFORE rebuild - this is required by OptiX!
    size_t hitGroupSbtSize;
    scene_.generateShaderBindingTableLayout(&hitGroupSbtSize);
    
    // Build the acceleration structure
    traversableHandle_ = ias_.rebuild(
        renderContext_->getCudaStream(),
        optixInstanceBuffer_,
        accelBuffer_,
        scratchBuffer);
    
    isDirty_ = false;
    needsRebuild_ = false;
    hasBeenBuilt_ = true;
}

void SceneHandler::updateIAS()
{
    if (!allowUpdate_)
    {
        LOG(WARNING) << "SceneHandler::updateIAS() called but updates are not enabled. Call setConfiguration with allowUpdate=true first.";
        buildIAS();  // Fall back to rebuild
        return;
    }
    
    if (!hasBeenBuilt_)
    {
        LOG(WARNING) << "SceneHandler::updateIAS() called but IAS has never been built. Building now.";
        buildIAS();
        return;
    }
    
    if (instances_.empty())
    {
        // Empty scene - nothing to update
        traversableHandle_ = 0;
        isDirty_ = false;
        return;
    }
    
    // Reallocate scratch buffer if needed for update
    OptixAccelBufferSizes memReq;
    ias_.prepareForBuild(&memReq);
    
    // Use shared scratch buffer from RenderContext, resize if needed
    cudau::Buffer& scratchBuffer = renderContext_->getASScratchBuffer();
    if (!scratchBuffer.isInitialized() || scratchBuffer.sizeInBytes() < memReq.tempUpdateSizeInBytes)
    {
        if (scratchBuffer.isInitialized())
            scratchBuffer.finalize();
        scratchBuffer.initialize(
            renderContext_->getCudaContext(),
            cudau::BufferType::Device,
            memReq.tempUpdateSizeInBytes,
            1);
    }
    
    // Update the acceleration structure
    ias_.update(renderContext_->getCudaStream(), scratchBuffer);
    
    // Note: SBT layout should remain valid after update (only transforms changed)
    
    // Note: traversableHandle remains the same after update
    isDirty_ = false;
    needsRebuild_ = false;
}

void SceneHandler::buildOrUpdateIAS()
{
    if (!isDirty_)
    {
        return;  // Nothing to do
    }
    
    // If we need a rebuild or updates aren't enabled, do a rebuild
    if (needsRebuild_ || !allowUpdate_ || !hasBeenBuilt_)
    {
        buildIAS();
    }
    else
    {
        // We can do an update (transform-only changes)
        updateIAS();
    }
}

void SceneHandler::addInstance(const optixu::Instance& instance)
{
    // Lazy initialization - only initialize when first instance is added
    if (!ias_)
    {
        initialize();
    }
    
    instances_.push_back(instance);
    ias_.addChild(instance);
    isDirty_ = true;
    needsRebuild_ = true;  // Adding instances requires rebuild
}

void SceneHandler::removeInstanceAt(uint32_t index)
{
    if (index >= instances_.size())
    {
        LOG(WARNING) << "SceneHandler::removeInstanceAt() called with invalid index: " << index;
        return;
    }
    
    instances_.erase(instances_.begin() + index);
    ias_.removeChildAt(index);
    isDirty_ = true;
    needsRebuild_ = true;  // Removing instances requires rebuild
}

void SceneHandler::clearInstances()
{
    instances_.clear();
    if (ias_)
    {
        ias_.clearChildren();
    }
    isDirty_ = true;
    needsRebuild_ = true;
    traversableHandle_ = 0;
}

void SceneHandler::updateInstanceTransform(uint32_t index, const float transform[12])
{
    if (index >= instances_.size())
    {
        LOG(WARNING) << "SceneHandler::updateInstanceTransform() called with invalid index: " << index;
        return;
    }
    
    instances_[index].setTransform(transform);
    isDirty_ = true;
    // Transform-only change - can use update if enabled
    // needsRebuild_ stays as is (not set to true)
}

void SceneHandler::markDirty()
{
    isDirty_ = true;
    if (ias_)
        ias_.markDirty();
}

bool SceneHandler::isReady() const
{
    return !isDirty_ && ias_.isReady();
}

OptixTraversableHandle SceneHandler::getTraversableHandle() const
{
    // A traversable handle of 0 is valid for an empty scene
    // Only warn if we have instances but haven't built yet
    if (isDirty_ && !instances_.empty())
    {
        LOG(WARNING) << "SceneHandler::getTraversableHandle() called with dirty scene containing instances. Call buildIAS() first.";
    }
    return traversableHandle_;
}

size_t SceneHandler::getInstanceCount() const
{
    return instances_.size();
}

optixu::Instance SceneHandler::getInstance(uint32_t index) const
{
    if (index >= instances_.size())
    {
        LOG(WARNING) << "SceneHandler::getInstance() called with invalid index: " << index;
        return optixu::Instance();
    }
    return instances_[index];
}

uint32_t SceneHandler::findInstanceIndex(const optixu::Instance& instance) const
{
    return ias_.findChildIndex(instance);
}