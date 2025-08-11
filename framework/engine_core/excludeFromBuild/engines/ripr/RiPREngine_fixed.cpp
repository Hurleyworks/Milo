// RiPREngine_fixed.cpp - Fixed SBT creation section
// This shows how the createSBTs() and related methods should be implemented

void RiPREngine::createSBTs()
{
    LOG(DBUG) << "Creating shader binding tables";
    
    if (!renderContext_)
    {
        LOG(WARNING) << "No render context for SBT creation";
        return;
    }
    
    auto cuContext = renderContext_->getCudaContext();
    
    // Get hit group SBT size from scene
    size_t hitGroupSbtSize = 0;
    scene_.generateShaderBindingTableLayout(&hitGroupSbtSize);
    LOG(DBUG) << "Scene hit group SBT size: " << hitGroupSbtSize << " bytes";
    
    // Create SBT for G-buffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline)
    {
        auto& p = gbufferPipeline_->optixPipeline;
        size_t sbtSize;
        p.generateShaderBindingTableLayout(&sbtSize);
        
        LOG(DBUG) << "G-buffer pipeline SBT size: " << sbtSize << " bytes";
        
        if (sbtSize > 0)
        {
            gbufferPipeline_->sbt.initialize(
                cuContext, cudau::BufferType::Device, sbtSize, 1);
            gbufferPipeline_->sbt.setMappedMemoryPersistent(true);
            p.setShaderBindingTable(gbufferPipeline_->sbt, gbufferPipeline_->sbt.getMappedPointer());
        }
        
        // Set hit group SBT for G-buffer pipeline
        if (!gbufferPipeline_->hitGroupSbt.isInitialized())
        {
            if (hitGroupSbtSize > 0)
            {
                gbufferPipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
                gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
            }
        }
        p.setHitGroupShaderBindingTable(gbufferPipeline_->hitGroupSbt, gbufferPipeline_->hitGroupSbt.getMappedPointer());
    }
    
    // Create SBT for path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
    {
        auto& p = pathTracePipeline_->optixPipeline;
        size_t sbtSize;
        p.generateShaderBindingTableLayout(&sbtSize);
        
        LOG(DBUG) << "Path tracing pipeline SBT size: " << sbtSize << " bytes";
        
        if (sbtSize > 0)
        {
            pathTracePipeline_->sbt.initialize(
                cuContext, cudau::BufferType::Device, sbtSize, 1);
            pathTracePipeline_->sbt.setMappedMemoryPersistent(true);
            p.setShaderBindingTable(pathTracePipeline_->sbt, pathTracePipeline_->sbt.getMappedPointer());
        }
        
        // Set hit group SBT for path tracing pipeline
        if (!pathTracePipeline_->hitGroupSbt.isInitialized())
        {
            if (hitGroupSbtSize > 0)
            {
                pathTracePipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
                pathTracePipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
            }
        }
        p.setHitGroupShaderBindingTable(pathTracePipeline_->hitGroupSbt, pathTracePipeline_->hitGroupSbt.getMappedPointer());
    }
}

void RiPREngine::updateSBTs()
{
    LOG(DBUG) << "Updating shader binding tables";
    
    if (!renderContext_)
    {
        LOG(WARNING) << "No render context for SBT update";
        return;
    }
    
    auto cuContext = renderContext_->getCudaContext();
    
    // Get updated hit group SBT size from scene
    size_t hitGroupSbtSize = 0;
    scene_.generateShaderBindingTableLayout(&hitGroupSbtSize);
    LOG(DBUG) << "Updated scene hit group SBT size: " << hitGroupSbtSize << " bytes";
    
    // Update hit group SBT for G-buffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline && hitGroupSbtSize > 0)
    {
        // Resize if needed
        if (gbufferPipeline_->hitGroupSbt.isInitialized())
        {
            size_t currentSize = gbufferPipeline_->hitGroupSbt.sizeInBytes();
            if (currentSize < hitGroupSbtSize)
            {
                LOG(DBUG) << "Resizing G-buffer pipeline hit group SBT from " << currentSize << " to " << hitGroupSbtSize << " bytes";
                gbufferPipeline_->hitGroupSbt.resize(1, hitGroupSbtSize);
            }
        }
        else
        {
            gbufferPipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
            gbufferPipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
        }
        gbufferPipeline_->optixPipeline.setHitGroupShaderBindingTable(
            gbufferPipeline_->hitGroupSbt, gbufferPipeline_->hitGroupSbt.getMappedPointer());
    }
    
    // Update hit group SBT for path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline && hitGroupSbtSize > 0)
    {
        // Resize if needed
        if (pathTracePipeline_->hitGroupSbt.isInitialized())
        {
            size_t currentSize = pathTracePipeline_->hitGroupSbt.sizeInBytes();
            if (currentSize < hitGroupSbtSize)
            {
                LOG(DBUG) << "Resizing path tracing pipeline hit group SBT from " << currentSize << " to " << hitGroupSbtSize << " bytes";
                pathTracePipeline_->hitGroupSbt.resize(1, hitGroupSbtSize);
            }
        }
        else
        {
            pathTracePipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
            pathTracePipeline_->hitGroupSbt.setMappedMemoryPersistent(true);
        }
        pathTracePipeline_->optixPipeline.setHitGroupShaderBindingTable(
            pathTracePipeline_->hitGroupSbt, pathTracePipeline_->hitGroupSbt.getMappedPointer());
    }
}

void RiPREngine::linkPipelines()
{
    LOG(DBUG) << "Linking pipelines";
    
    // Link G-buffer pipeline
    if (gbufferPipeline_ && gbufferPipeline_->optixPipeline)
    {
        OptixStackSizes stackSizes = {};
        for (const auto& [name, pg] : gbufferPipeline_->entryPoints) {
            optixu::Program::getStackSize(pg.getRawHandle(), &stackSizes);
        }
        for (const auto& [name, pg] : gbufferPipeline_->programs) {
            optixu::Program::getStackSize(pg.getRawHandle(), &stackSizes);
        }
        for (const auto& [name, hg] : gbufferPipeline_->hitPrograms) {
            optixu::HitProgramGroup::getStackSize(hg.getRawHandle(), &stackSizes);
        }
        
        uint32_t maxTraceDepth = 2;
        uint32_t stackSize = stackSizes.dssCH;
        gbufferPipeline_->config.stackSize = stackSize;
        gbufferPipeline_->optixPipeline.setStackSize(stackSize, stackSize, stackSize, maxTraceDepth);
        
        gbufferPipeline_->optixPipeline.link();
        LOG(DBUG) << "G-buffer pipeline linked with stack size: " << stackSize;
    }
    
    // Link path tracing pipeline
    if (pathTracePipeline_ && pathTracePipeline_->optixPipeline)
    {
        OptixStackSizes stackSizes = {};
        for (const auto& [name, pg] : pathTracePipeline_->entryPoints) {
            optixu::Program::getStackSize(pg.getRawHandle(), &stackSizes);
        }
        for (const auto& [name, pg] : pathTracePipeline_->programs) {
            optixu::Program::getStackSize(pg.getRawHandle(), &stackSizes);
        }
        for (const auto& [name, hg] : pathTracePipeline_->hitPrograms) {
            optixu::HitProgramGroup::getStackSize(hg.getRawHandle(), &stackSizes);
        }
        
        uint32_t maxTraceDepth = 10;  // Higher for path tracing
        uint32_t stackSize = stackSizes.dssCH;
        pathTracePipeline_->config.stackSize = stackSize;
        pathTracePipeline_->optixPipeline.setStackSize(stackSize, stackSize, stackSize, maxTraceDepth);
        
        pathTracePipeline_->optixPipeline.link();
        LOG(DBUG) << "Path tracing pipeline linked with stack size: " << stackSize;
    }
}

void RiPREngine::updateMaterialHitGroups()
{
    LOG(DBUG) << "Updating material hit groups";
    
    // Update hit groups for all materials in the scene
    // This would iterate through materials and set appropriate hit programs
    // Implementation depends on material handler interface
    
    for (const auto& model : modelHandler_->getModels())
    {
        // Set hit groups based on material properties
        // e.g., model->setHitGroups(gbufferPipeline_->hitPrograms, pathTracePipeline_->hitPrograms);
    }
}