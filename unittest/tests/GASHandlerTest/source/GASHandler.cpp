#include "GASHandler.h"

GASHandler::GASHandler(optixu::Scene scene, CUcontext cudaContext)
    : scene_(scene), cudaContext_(cudaContext)
{
}

GASHandler::~GASHandler()
{
    finalize();
}

void GASHandler::initialize()
{
    if (initialized_)
    {
        LOG(DBUG) << "GASHandler already initialized";
        return;
    }

    if (!scene_)
    {
        LOG(WARNING) << "GASHandler::initialize() called with null Scene";
        return;
    }

    if (!cudaContext_)
    {
        LOG(WARNING) << "GASHandler::initialize() - CUDA context not available";
        return;
    }

    // Initialize scratch buffer with minimum size
    scratchBuffer_.initialize(cudaContext_, cudau::BufferType::Device, MIN_SCRATCH_SIZE, 1);
    currentScratchSize_ = MIN_SCRATCH_SIZE;

    initialized_ = true;
    LOG(DBUG) << "GASHandler initialized";
}

void GASHandler::finalize()
{
    if (!initialized_)
        return;

    // Clean up all GAS objects
    for (auto& [id, gasData] : gasMap_)
    {
        if (gasData.gas)
        {
            gasData.gas.destroy();
        }
        if (gasData.accelBuffer.isInitialized())
        {
            gasData.accelBuffer.finalize();
        }
        if (gasData.compactedBuffer.isInitialized())
        {
            gasData.compactedBuffer.finalize();
        }
    }
    gasMap_.clear();

    // Clean up scratch buffer
    if (scratchBuffer_.isInitialized())
    {
        scratchBuffer_.finalize();
    }

    totalMemoryUsage_ = 0;
    initialized_ = false;
    LOG(DBUG) << "GASHandler finalized";
}

uint32_t GASHandler::createGAS(
    const optixu::GeometryInstance& geomInst,
    optixu::ASTradeoff tradeoff,
    bool allowUpdate,
    bool allowCompaction)
{
    std::vector<optixu::GeometryInstance> geomInsts = { geomInst };
    return createGAS(geomInsts, tradeoff, allowUpdate, allowCompaction);
}

uint32_t GASHandler::createGAS(
    const std::vector<optixu::GeometryInstance>& geomInsts,
    optixu::ASTradeoff tradeoff,
    bool allowUpdate,
    bool allowCompaction)
{
    if (!initialized_)
    {
        initialize();
    }

    if (geomInsts.empty())
    {
        LOG(WARNING) << "GASHandler::createGAS() called with empty geometry instances";
        return 0;
    }

    uint32_t gasId = getNextGASId();
    GASData& gasData = gasMap_[gasId];

    // Create the GAS
    gasData.gas = scene_.createGeometryAccelerationStructure();
    
    // Configure the GAS
    gasData.gas.setConfiguration(
        tradeoff,
        allowUpdate ? optixu::AllowUpdate::Yes : optixu::AllowUpdate::No,
        allowCompaction ? optixu::AllowCompaction::Yes : optixu::AllowCompaction::No);

    // Store configuration
    gasData.tradeoff = tradeoff;
    gasData.allowUpdate = allowUpdate;
    gasData.allowCompaction = allowCompaction;

    // Add geometry instances as children
    for (const auto& geomInst : geomInsts)
    {
        gasData.gas.addChild(geomInst);
        gasData.children.push_back(geomInst);
    }

    // Set default material configuration
    gasData.materialSetCount = 1;
    gasData.rayTypeCounts = { 1 };
    gasData.gas.setMaterialSetCount(1);
    gasData.gas.setRayTypeCount(0, 1);

    // Mark as dirty (needs build)
    gasData.isDirty = true;
    gasData.hasBeenBuilt = false;

    LOG(DBUG) << "Created GAS with ID " << gasId << " containing " << geomInsts.size() << " geometry instances";
    return gasId;
}

optixu::GeometryAccelerationStructure GASHandler::getGAS(uint32_t gasId) const
{
    auto it = gasMap_.find(gasId);
    if (it != gasMap_.end())
    {
        return it->second.gas;
    }
    return optixu::GeometryAccelerationStructure();
}

OptixTraversableHandle GASHandler::getTraversableHandle(uint32_t gasId) const
{
    auto it = gasMap_.find(gasId);
    if (it != gasMap_.end())
    {
        if (!it->second.hasBeenBuilt)
        {
            LOG(WARNING) << "GASHandler::getTraversableHandle() - GAS " << gasId << " has not been built yet";
        }
        return it->second.traversableHandle;
    }
    return 0;
}

bool GASHandler::hasGAS(uint32_t gasId) const
{
    return gasMap_.find(gasId) != gasMap_.end();
}

void GASHandler::removeGAS(uint32_t gasId)
{
    auto it = gasMap_.find(gasId);
    if (it != gasMap_.end())
    {
        // Update total memory usage
        totalMemoryUsage_ -= it->second.memoryUsage;

        // Clean up resources
        if (it->second.gas)
        {
            it->second.gas.destroy();
        }
        if (it->second.accelBuffer.isInitialized())
        {
            it->second.accelBuffer.finalize();
        }
        if (it->second.compactedBuffer.isInitialized())
        {
            it->second.compactedBuffer.finalize();
        }

        gasMap_.erase(it);
        LOG(DBUG) << "Removed GAS with ID " << gasId;
    }
}

void GASHandler::buildGAS(uint32_t gasId, CUstream stream)
{
    auto it = gasMap_.find(gasId);
    if (it == gasMap_.end())
    {
        LOG(WARNING) << "GASHandler::buildGAS() - GAS " << gasId << " not found";
        return;
    }

    buildGASInternal(gasId, it->second, stream);
}

void GASHandler::buildAllDirty(CUstream stream)
{
    uint32_t builtCount = 0;
    for (auto& [gasId, gasData] : gasMap_)
    {
        if (gasData.isDirty)
        {
            buildGASInternal(gasId, gasData, stream);
            builtCount++;
        }
    }
    
    if (builtCount > 0)
    {
        LOG(INFO) << "Built " << builtCount << " dirty GAS objects";
    }
}

void GASHandler::rebuildAll(CUstream stream)
{
    for (auto& [gasId, gasData] : gasMap_)
    {
        gasData.isDirty = true;
        buildGASInternal(gasId, gasData, stream);
    }
    LOG(INFO) << "Rebuilt all " << gasMap_.size() << " GAS objects";
}

void GASHandler::compactGAS(uint32_t gasId, CUstream stream)
{
    auto it = gasMap_.find(gasId);
    if (it == gasMap_.end())
    {
        LOG(WARNING) << "GASHandler::compactGAS() - GAS " << gasId << " not found";
        return;
    }

    compactGASInternal(gasId, it->second, stream);
}

void GASHandler::compactAll(CUstream stream)
{
    uint32_t compactedCount = 0;
    size_t savedMemory = 0;

    for (auto& [gasId, gasData] : gasMap_)
    {
        if (!gasData.allowCompaction || !gasData.hasBeenBuilt || gasData.isCompacted)
            continue;

        // Check if compaction would be beneficial
        if (gasData.potentialCompactedSize < gasData.memoryUsage * COMPACTION_THRESHOLD)
        {
            size_t before = gasData.memoryUsage;
            compactGASInternal(gasId, gasData, stream);
            savedMemory += (before - gasData.memoryUsage);
            compactedCount++;
        }
    }

    if (compactedCount > 0)
    {
        LOG(INFO) << "Compacted " << compactedCount << " GAS objects, saved " << savedMemory << " bytes";
    }
}

void GASHandler::markDirty(uint32_t gasId)
{
    auto it = gasMap_.find(gasId);
    if (it != gasMap_.end())
    {
        it->second.isDirty = true;
        it->second.gas.markDirty();
    }
}

bool GASHandler::isDirty(uint32_t gasId) const
{
    auto it = gasMap_.find(gasId);
    if (it != gasMap_.end())
    {
        return it->second.isDirty;
    }
    return false;
}

uint32_t GASHandler::getGASCount() const
{
    return static_cast<uint32_t>(gasMap_.size());
}

std::vector<uint32_t> GASHandler::getAllGASIds() const
{
    std::vector<uint32_t> ids;
    ids.reserve(gasMap_.size());
    for (const auto& [id, _] : gasMap_)
    {
        ids.push_back(id);
    }
    return ids;
}

size_t GASHandler::getTotalMemoryUsage() const
{
    return totalMemoryUsage_;
}

size_t GASHandler::getGASMemoryUsage(uint32_t gasId) const
{
    auto it = gasMap_.find(gasId);
    if (it != gasMap_.end())
    {
        return it->second.memoryUsage;
    }
    return 0;
}

void GASHandler::setMaterialConfiguration(
    uint32_t gasId,
    uint32_t materialSetCount,
    const std::vector<uint32_t>& rayTypeCounts)
{
    auto it = gasMap_.find(gasId);
    if (it == gasMap_.end())
    {
        LOG(WARNING) << "GASHandler::setMaterialConfiguration() - GAS " << gasId << " not found";
        return;
    }

    GASData& gasData = it->second;
    
    if (rayTypeCounts.size() != materialSetCount)
    {
        LOG(WARNING) << "GASHandler::setMaterialConfiguration() - rayTypeCounts size doesn't match materialSetCount";
        return;
    }

    gasData.materialSetCount = materialSetCount;
    gasData.rayTypeCounts = rayTypeCounts;

    // Apply to GAS
    gasData.gas.setMaterialSetCount(materialSetCount);
    for (uint32_t i = 0; i < materialSetCount; ++i)
    {
        gasData.gas.setRayTypeCount(i, rayTypeCounts[i]);
    }

    // Configuration change requires rebuild
    gasData.isDirty = true;
}

void GASHandler::setMotionOptions(
    uint32_t gasId,
    uint32_t numKeys,
    float timeBegin,
    float timeEnd,
    OptixMotionFlags flags)
{
    auto it = gasMap_.find(gasId);
    if (it == gasMap_.end())
    {
        LOG(WARNING) << "GASHandler::setMotionOptions() - GAS " << gasId << " not found";
        return;
    }

    GASData& gasData = it->second;
    
    gasData.hasMotion = (numKeys > 1);
    gasData.motionKeyCount = numKeys;
    gasData.motionTimeBegin = timeBegin;
    gasData.motionTimeEnd = timeEnd;
    gasData.motionFlags = flags;

    // Apply to GAS
    gasData.gas.setMotionOptions(numKeys, timeBegin, timeEnd, flags);

    // Motion configuration change requires rebuild
    gasData.isDirty = true;
}

uint32_t GASHandler::findGASContaining(const optixu::GeometryInstance& geomInst) const
{
    for (const auto& [gasId, gasData] : gasMap_)
    {
        for (const auto& child : gasData.children)
        {
            if (child == geomInst)
            {
                return gasId;
            }
        }
    }
    return 0;  // Not found
}

void GASHandler::buildGASInternal(uint32_t gasId, GASData& gasData, CUstream stream)
{
    if (!gasData.isDirty && gasData.hasBeenBuilt)
    {
        return;  // Already built and not dirty
    }

    // Empty GAS special case
    if (gasData.children.empty())
    {
        gasData.traversableHandle = 0;
        gasData.isDirty = false;
        gasData.memoryUsage = 0;
        return;
    }

    // Mark GAS dirty for rebuild
    gasData.gas.markDirty();

    // Prepare for build
    OptixAccelBufferSizes memReq;
    gasData.gas.prepareForBuild(&memReq);

    // Ensure scratch buffer is large enough
    size_t scratchSize = gasData.allowUpdate 
        ? std::max(memReq.tempSizeInBytes, memReq.tempUpdateSizeInBytes)
        : memReq.tempSizeInBytes;
    ensureScratchBufferSize(scratchSize);

    // Allocate or resize acceleration buffer
    if (!gasData.accelBuffer.isInitialized() || gasData.accelBuffer.sizeInBytes() < memReq.outputSizeInBytes)
    {
        if (gasData.accelBuffer.isInitialized())
        {
            totalMemoryUsage_ -= gasData.memoryUsage;
            gasData.accelBuffer.finalize();
        }
        gasData.accelBuffer.initialize(cudaContext_, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
    }

    // Build or rebuild
    if (!gasData.hasBeenBuilt || !gasData.allowUpdate)
    {
        gasData.traversableHandle = gasData.gas.rebuild(stream, gasData.accelBuffer, scratchBuffer_);
    }
    else
    {
        gasData.gas.update(stream, scratchBuffer_);
        // Traversable handle remains the same after update
    }

    // Wait for build to complete
    cuStreamSynchronize(stream);

    // Update state
    gasData.isDirty = false;
    gasData.hasBeenBuilt = true;
    gasData.isCompacted = false;
    
    // Update memory tracking
    totalMemoryUsage_ -= gasData.memoryUsage;
    gasData.memoryUsage = memReq.outputSizeInBytes;
    totalMemoryUsage_ += gasData.memoryUsage;

    // Check potential compacted size if compaction is enabled
    if (gasData.allowCompaction)
    {
        gasData.gas.prepareForCompact(&gasData.potentialCompactedSize);
    }

    totalBuilds_++;
    LOG(DBUG) << "Built GAS " << gasId << ", memory: " << gasData.memoryUsage << " bytes";
}

void GASHandler::compactGASInternal(uint32_t gasId, GASData& gasData, CUstream stream)
{
    if (!gasData.allowCompaction)
    {
        LOG(WARNING) << "GAS " << gasId << " does not allow compaction";
        return;
    }

    if (!gasData.hasBeenBuilt)
    {
        LOG(WARNING) << "GAS " << gasId << " has not been built yet";
        return;
    }

    if (gasData.isCompacted)
    {
        LOG(DBUG) << "GAS " << gasId << " is already compacted";
        return;
    }

    // Get compacted size
    size_t compactedSize;
    gasData.gas.prepareForCompact(&compactedSize);

    // Check if compaction is worthwhile
    if (compactedSize >= gasData.memoryUsage * COMPACTION_THRESHOLD)
    {
        LOG(DBUG) << "Compaction would only save " 
                  << (100.0f * (1.0f - float(compactedSize) / float(gasData.memoryUsage)))
                  << "% for GAS " << gasId << ", skipping";
        return;
    }

    // Allocate compacted buffer
    if (gasData.compactedBuffer.isInitialized())
    {
        gasData.compactedBuffer.finalize();
    }
    gasData.compactedBuffer.initialize(cudaContext_, cudau::BufferType::Device, compactedSize, 1);

    // Perform compaction
    OptixTraversableHandle compactedHandle = gasData.gas.compact(stream, gasData.compactedBuffer);

    // Wait for compaction to complete
    cuStreamSynchronize(stream);

    // Remove uncompacted data
    gasData.gas.removeUncompacted();

    // Swap buffers
    std::swap(gasData.accelBuffer, gasData.compactedBuffer);

    // Update state
    gasData.traversableHandle = compactedHandle;
    gasData.isCompacted = true;
    
    // Update memory tracking
    totalMemoryUsage_ -= gasData.memoryUsage;
    gasData.memoryUsage = compactedSize;
    totalMemoryUsage_ += gasData.memoryUsage;

    totalCompactions_++;
    LOG(INFO) << "Compacted GAS " << gasId << " to " << compactedSize << " bytes";
}

void GASHandler::ensureScratchBufferSize(size_t requiredSize)
{
    if (requiredSize > currentScratchSize_)
    {
        // Clamp to maximum size
        requiredSize = std::min(requiredSize, MAX_SCRATCH_SIZE);
        
        if (scratchBuffer_.isInitialized())
        {
            scratchBuffer_.finalize();
        }
        
        scratchBuffer_.initialize(cudaContext_, cudau::BufferType::Device, requiredSize, 1);
        currentScratchSize_ = requiredSize;
        
        LOG(DBUG) << "Resized scratch buffer to " << requiredSize << " bytes";
    }
}

uint32_t GASHandler::getNextGASId()
{
    return nextGASId_++;
}