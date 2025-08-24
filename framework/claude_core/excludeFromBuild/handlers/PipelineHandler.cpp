#include "PipelineHandler.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "../common/common_host.h"  // For LOG macro

// Private implementation structure
struct PipelineHandler::Impl {
    RenderContextPtr renderContext;
    optixu::Context optixContext;
    CUcontext cuContext = nullptr;
    
    // Storage for all pipelines
    std::unordered_map<EntryPointType, Pipeline::Ptr> pipelines;
    
    // Shared resources
    // std::unordered_map<std::string, optixu::Module> moduleCache; // REMOVED: modules cannot be shared
    optixu::Scene currentScene;
    
    explicit Impl(RenderContextPtr ctx) : renderContext(ctx) {
        // Get OptiX context from RenderContext
        if (ctx && ctx->isInitialized()) {
            optixContext = ctx->getOptiXContext();
            cuContext = ctx->getCudaContext();
        }
    }
};

// Constructor
PipelineHandler::PipelineHandler(RenderContextPtr ctx) 
    : pImpl(std::make_unique<Impl>(ctx)) {
    if (!ctx || !ctx->isInitialized()) {
        throw std::runtime_error("Invalid or uninitialized RenderContext provided to PipelineHandler");
    }
}

// Destructor
PipelineHandler::~PipelineHandler() {
    destroyAll();
}

// Setup pipeline from data and kernel name
void PipelineHandler::setupPipeline(const PipelineData& data, const std::string& kernelName) {
    LOG(INFO) << "Setting up pipeline for kernel: " << kernelName 
              << ", entry point: " << static_cast<int>(data.entryPoint);
    
    // 1. Create or get pipeline for the entry point type
    auto pipeline = createPipeline(data.entryPoint);
    
    // 2. Create pipeline if not exists
    createOrGetPipeline(pipeline, data.entryPoint);
    
    // 3. Configure pipeline options
    configurePipelineOptions(pipeline, data.config);
    
    // 4. Load module for this pipeline
    loadPipelineModule(pipeline, kernelName, data.entryPoint);
    if (!pipeline->optixModule) {
        LOG(WARNING) << "Failed to load module for pipeline";
        return;
    }
    
    // 5. Create programs
    createPipelinePrograms(pipeline, data);
    
    // 6. Setup ray types
    setupPipelineRayTypes(pipeline, data);
    
    // 7. Link and finalize pipeline
    linkAndFinalizePipeline(pipeline, data);
    
    LOG(INFO) << "Pipeline setup complete for type: " << static_cast<int>(data.entryPoint);
}

// Setup pipeline from PTX file
void PipelineHandler::setupPipeline(const PipelineData& data, const std::filesystem::path& ptxFile) {
    std::string kernelName = ptxFile.stem().string();
    setupPipeline(data, kernelName);
}

// Setup shader binding table
// Setup shader binding table - simplified like production code
void PipelineHandler::setupSBT(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline || !pipeline->optixPipeline) return;
    
    size_t sbtSize = 0;
    pipeline->optixPipeline.generateShaderBindingTableLayout(&sbtSize);
    
    if (sbtSize > 0 && !pipeline->shaderBindingTable.isInitialized()) {
        pipeline->shaderBindingTable.initialize(
            pImpl->cuContext, cudau::BufferType::Device, sbtSize, 1);
        pipeline->shaderBindingTable.setMappedMemoryPersistent(true);
        pipeline->optixPipeline.setShaderBindingTable(
            pipeline->shaderBindingTable,
            pipeline->shaderBindingTable.getMappedPointer());
    }
    
    // SBT is now ready
    if (pipeline->state >= PipelineState::Linked) {
        pipeline->transitionTo(PipelineState::SBTReady);
        if (pImpl->currentScene) {
            pipeline->transitionTo(PipelineState::Ready);
        }
    }
}

// Update SBT
void PipelineHandler::updateSBT(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline) return;
    
    // Mark as needing SBT update by going back to Linked state
    if (pipeline->state > PipelineState::Linked) {
        pipeline->transitionTo(PipelineState::Linked);
    }
    setupSBT(type);
}

// Set minimal hit group SBT - removed, now handled in setSceneDependentSBT
void PipelineHandler::setMinimalHitGroupSBT(EntryPointType type) {
    // This functionality is now integrated into setSceneDependentSBT
    // which always ensures at least 1 byte is allocated
    setSceneDependentSBT(type);
}

// Set scene-dependent SBT - simplified like production code
void PipelineHandler::setSceneDependentSBT(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline || !pipeline->optixPipeline) return;
    
    // Get hit group SBT size from scene
    size_t hitGroupSbtSize = 0;
    if (pImpl->currentScene) {
        pImpl->currentScene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    }
    
    LOG(INFO) << "Scene hit group SBT size: " << hitGroupSbtSize << " bytes";
    
    // Always allocate at least 1 byte (OptiX requirement)
    size_t bufferSize = std::max<size_t>(hitGroupSbtSize, 1);
    
    if (!pipeline->hitGroupSBT.isInitialized() || 
        pipeline->hitGroupSBT.sizeInBytes() < bufferSize) {
        
        if (pipeline->hitGroupSBT.isInitialized()) {
            pipeline->hitGroupSBT.finalize();
        }
        
        pipeline->hitGroupSBT.initialize(
            pImpl->cuContext, cudau::BufferType::Device, bufferSize, 1);
        pipeline->hitGroupSBT.setMappedMemoryPersistent(true);
    }
    
    // Set hit group SBT on pipeline
    pipeline->optixPipeline.setHitGroupShaderBindingTable(
        pipeline->hitGroupSBT,
        pipeline->hitGroupSBT.getMappedPointer());
}

// Get pipeline
Pipeline::Ptr PipelineHandler::getPipeline(EntryPointType type) {
    auto it = pImpl->pipelines.find(type);
    return (it != pImpl->pipelines.end()) ? it->second : nullptr;
}

// Check if pipeline exists
bool PipelineHandler::hasPipeline(EntryPointType type) const {
    return pImpl->pipelines.find(type) != pImpl->pipelines.end();
}

// Launch pipeline - simplified
void PipelineHandler::launch(EntryPointType type, CUstream stream, CUdeviceptr plp,
                             uint32_t width, uint32_t height, uint32_t depth) {
    auto pipeline = getPipeline(type);
    if (!pipeline || !pipeline->isReady()) {
        throw std::runtime_error("Pipeline not ready for launch");
    }
    
    pipeline->optixPipeline.launch(stream, plp, width, height, depth);
}

// Set scene for all pipelines
void PipelineHandler::setScene(optixu::Scene scene) {
    pImpl->currentScene = scene;
    LOG(INFO) << "Setting scene on all pipelines";
    
    for (auto& [type, pipeline] : pImpl->pipelines) {
        if (pipeline && pipeline->optixPipeline) {
            pipeline->optixPipeline.setScene(scene);
            LOG(INFO) << "Scene set on pipeline type: " << static_cast<int>(type);
        }
    }
}

// Update scene SBT when geometry changes
void PipelineHandler::updateSceneSBT() {
    if (!pImpl->currentScene) {
        LOG(WARNING) << "No scene set, cannot update scene SBT";
        return;
    }
    
    LOG(INFO) << "Updating scene SBT for all pipelines";
    
    // Update SBT for all active pipelines
    for (auto& [type, pipeline] : pImpl->pipelines) {
        if (pipeline && pipeline->isInitialized()) {
            setSceneDependentSBT(type);
            // Mark as ready if we have everything
            if (pipeline->state == PipelineState::SBTReady) {
                pipeline->transitionTo(PipelineState::Ready);
            }
        }
    }
}


// Set material hit groups
void PipelineHandler::setMaterialHitGroups(EntryPointType type, optixu::Material material) {
    auto pipeline = getPipeline(type);
    if (!pipeline) return;
    
    // TODO: Set hit groups on material for each ray type
    for (uint32_t rayType = 0; rayType < pipeline->numRayTypes; ++rayType) {
        // Get appropriate hit group for this ray type
        // material.setHitGroup(rayType, hitGroup);
    }
}

// Get hit group
optixu::HitProgramGroup PipelineHandler::getHitGroup(EntryPointType type, 
                                                     const std::string& name) {
    auto pipeline = getPipeline(type);
    if (!pipeline) {
        throw std::runtime_error("Pipeline not found");
    }
    
    auto it = pipeline->hitGroups.find(name);
    if (it != pipeline->hitGroups.end()) {
        return it->second;
    }
    
    throw std::runtime_error("Hit group not found: " + name);
}

// Create empty hit group
void PipelineHandler::createEmptyHitGroup(EntryPointType type, const std::string& name) {
    auto pipeline = getPipeline(type);
    if (!pipeline || !pipeline->optixPipeline) {
        LOG(WARNING) << "Pipeline not found or not initialized for type: " << static_cast<int>(type);
        return;
    }
    
    // Check if hit group already exists
    if (pipeline->hitGroups.find(name) != pipeline->hitGroups.end()) {
        LOG(INFO) << "Hit group already exists: " << name;
        return;
    }
    
    // Create empty hit group (no closest hit, no any hit, no intersection)
    auto emptyHitGroup = pipeline->optixPipeline.createEmptyHitProgramGroup();
    pipeline->hitGroups[name] = emptyHitGroup;
    LOG(INFO) << "Created empty hit group: " << name << " for pipeline type: " << static_cast<int>(type);
}

// Configure hit groups for materials on geometry instances
void PipelineHandler::configureMaterialHitGroups(optixu::GeometryInstance* geomInst,
                                                 const std::map<EntryPointType, std::vector<std::pair<uint32_t, std::string>>>& hitGroupConfig) {
    if (!geomInst) {
        LOG(WARNING) << "No geometry instance provided for hit group configuration";
        return;
    }
    
    // Material count is now handled automatically by the geometry instance
    uint32_t numMaterials = 1; // Default to 1 material
    LOG(DBUG) << "Configuring hit groups for " << numMaterials << " material(s)";
    
    // Iterate through all materials in the geometry instance
    for (uint32_t i = 0; i < numMaterials; ++i) {
        optixu::Material mat = geomInst->getMaterial(0, i); // Material set 0, index i
        if (!mat) continue;
        
        // Apply hit group configuration for each pipeline type
        for (const auto& [entryPoint, rayTypeConfigs] : hitGroupConfig) {
            try {
                // Set hit groups for each ray type
                for (const auto& [rayType, hitGroupName] : rayTypeConfigs) {
                    auto hitGroup = getHitGroup(entryPoint, hitGroupName);
                    mat.setHitGroup(rayType, hitGroup);
                    LOG(DBUG) << "Set hit group '" << hitGroupName << "' for ray type " << rayType 
                              << " on pipeline " << static_cast<int>(entryPoint);
                }
            }
            catch (const std::exception& e) {
                LOG(WARNING) << "Failed to set hit groups for pipeline " << static_cast<int>(entryPoint) 
                             << ": " << e.what();
            }
        }
    }
    
    LOG(DBUG) << "Completed hit group configuration for " << numMaterials << " material(s)";
}

// Calculate stack sizes
void PipelineHandler::calculateStackSizes(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline) return;
    
    // TODO: Calculate stack sizes based on programs
    uint32_t directCallable = 0;
    uint32_t directCallableFromState = 0;
    uint32_t continuation = 0;
    
    // Calculate max stack sizes from programs
    for (const auto& cp : pipeline->callablePrograms) {
        directCallableFromState = std::max(directCallableFromState, cp.getDCStackSize());
    }
    
    // Set calculated stack sizes
    setStackSize(type, directCallable, directCallableFromState, continuation);
}

// Set stack size
void PipelineHandler::setStackSize(EntryPointType type, uint32_t directCallable,
                                   uint32_t directCallableFromState, uint32_t continuation) {
    auto pipeline = getPipeline(type);
    if (pipeline && pipeline->optixPipeline) {
        pipeline->optixPipeline.setStackSize(
            directCallable, directCallableFromState, continuation, 
            pipeline->config.maxTraceDepth);
    }
}


// Setup callable programs
void PipelineHandler::setupCallablePrograms(EntryPointType type,
                                           const std::vector<std::string>& entryPoints) {
    auto pipeline = getPipeline(type);
    if (!pipeline) return;
    
    // TODO: Create callable programs from entry points
    // Note: OptiX pipelines don't have a setCallableProgramCount method
    // Callable programs are set individually via setCallableProgram
    
    for (size_t i = 0; i < entryPoints.size(); ++i) {
        // Create callable program
        // auto program = pipeline->optixPipeline.createCallableProgramGroup(...);
        // pipeline->callablePrograms.push_back(program);
        // pipeline->optixPipeline.setCallableProgram(i, program);
    }
}

// Check if pipeline is ready
bool PipelineHandler::isReady(EntryPointType type) const {
    auto it = pImpl->pipelines.find(type);
    return (it != pImpl->pipelines.end()) && it->second->isReady();
}

// Check if any pipeline is ready
bool PipelineHandler::isAnyPipelineReady() const {
    for (const auto& [type, pipeline] : pImpl->pipelines) {
        if (pipeline->isReady()) {
            return true;
        }
    }
    return false;
}

// Get list of active pipelines
std::vector<EntryPointType> PipelineHandler::getActivePipelines() const {
    std::vector<EntryPointType> active;
    for (const auto& [type, pipeline] : pImpl->pipelines) {
        if (pipeline->isInitialized()) {
            active.push_back(type);
        }
    }
    return active;
}

// Render sequence of pipelines
void PipelineHandler::renderSequence(const std::vector<EntryPointType>& sequence,
                                     CUstream stream, CUdeviceptr plp,
                                     uint32_t width, uint32_t height) {
    for (EntryPointType type : sequence) {
        if (isReady(type)) {
            launch(type, stream, plp, width, height);
        }
    }
}

// Destroy specific pipeline
void PipelineHandler::destroyPipeline(EntryPointType type) {
    auto it = pImpl->pipelines.find(type);
    if (it != pImpl->pipelines.end()) {
        if (it->second) {
            it->second->destroy();
        }
        pImpl->pipelines.erase(it);
    }
}

// Destroy all pipelines
void PipelineHandler::destroyAll() {
    for (auto& [type, pipeline] : pImpl->pipelines) {
        if (pipeline) {
            pipeline->destroy();
        }
    }
    pImpl->pipelines.clear();
    // Module cache removed - each pipeline owns its module
}

// Private: Create pipeline
Pipeline::Ptr PipelineHandler::createPipeline(EntryPointType type) {
    auto it = pImpl->pipelines.find(type);
    if (it != pImpl->pipelines.end()) {
        return it->second;
    }
    
    auto pipeline = std::make_shared<Pipeline>();
    pImpl->pipelines[type] = pipeline;
    return pipeline;
}

// Private: Link pipeline - simplified, now called directly in setupPipeline
void PipelineHandler::linkPipeline(EntryPointType type, uint32_t maxTraceDepth) {
    // This method is kept for compatibility but linking is now done directly in setupPipeline
    auto pipeline = getPipeline(type);
    if (pipeline && pipeline->optixPipeline) {
        pipeline->optixPipeline.link(maxTraceDepth);
    }
}

// Private: Setup ray types and miss programs
void PipelineHandler::setupRayTypes(EntryPointType type, uint32_t numRayTypes) {
    auto pipeline = getPipeline(type);
    if (!pipeline || !pipeline->optixPipeline) return;
    
    LOG(INFO) << "Setting up " << numRayTypes << " ray types for pipeline type: " << static_cast<int>(type);
    
    // Set the number of ray types
    pipeline->optixPipeline.setMissRayTypeCount(numRayTypes);
    
    // Configure miss programs based on the pipeline type
    if (type == EntryPointType::PathTrace) {
        // For path tracing: typically Closest=0, Visibility=1
        if (pipeline->programs.find(ProgramType::Miss) != pipeline->programs.end()) {
            // Set miss program for closest ray (search ray)
            pipeline->optixPipeline.setMissProgram(0, pipeline->programs[ProgramType::Miss]);
            
            // Create empty miss for visibility rays if we have multiple ray types
            if (numRayTypes > 1) {
                optixu::Module emptyModule;
                auto emptyMiss = pipeline->optixPipeline.createMissProgram(emptyModule, nullptr);
                pipeline->optixPipeline.setMissProgram(1, emptyMiss);
            }
        }
    } else if (type == EntryPointType::GBuffer) {
        // For GBuffer: typically just Primary=0
        if (pipeline->programs.find(ProgramType::Miss) != pipeline->programs.end()) {
            pipeline->optixPipeline.setMissProgram(0, pipeline->programs[ProgramType::Miss]);
        }
    }
    // Add more pipeline types as needed
}

// Private: Generate SBT layout
void PipelineHandler::generateSBTLayout(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline || !pipeline->optixPipeline) return;
    
    // Ray type configuration is now handled in setupRayTypes
    
    // Set callable programs if any
    if (!pipeline->callablePrograms.empty()) {
        // Note: OptiX pipelines don't have a setCallableProgramCount method
        // Set each callable program individually
        for (size_t i = 0; i < pipeline->callablePrograms.size(); ++i) {
            pipeline->optixPipeline.setCallableProgram(i, pipeline->callablePrograms[i]);
        }
    }
    
    // CRITICAL: Ensure hit group SBT is always allocated
    // Even with no hit groups, OptiX requires a valid (even if minimal) hit group SBT
    if (!pipeline->hitGroupSBT.isInitialized()) {
        LOG(INFO) << "[SBT_LAYOUT] Allocating minimal hit group SBT in generateSBTLayout";
        pipeline->hitGroupSBT.initialize(
            pImpl->cuContext, cudau::BufferType::Device, 1, 1);
        pipeline->hitGroupSBT.setMappedMemoryPersistent(true);
        
        // Set it on the pipeline
        pipeline->optixPipeline.setHitGroupShaderBindingTable(
            pipeline->hitGroupSBT,
            pipeline->hitGroupSBT.getMappedPointer());
        LOG(INFO) << "[SBT_LAYOUT] Minimal hit group SBT set in generateSBTLayout";
    }
}

// Private: Validate pipeline configuration
void PipelineHandler::validatePipelineConfig(const PipelineConfig& config) {
    if (config.launchParamsSize == 0) {
        throw std::runtime_error("Launch parameters size must be greater than 0");
    }
    if (config.numPayloadDwords == 0) {
        throw std::runtime_error("Payload size must be greater than 0");
    }
    // Add more validation as needed
}

// Private: Load PTX from file
std::vector<char> PipelineHandler::loadPTXFromFile(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("PTX file not found: " + path.string());
    }
    
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open PTX file: " + path.string());
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> data(size);
    if (!file.read(data.data(), size)) {
        throw std::runtime_error("Failed to read PTX file: " + path.string());
    }
    
    return data;
}

// Private: Load OptiX IR from file
std::vector<char> PipelineHandler::loadOptixIRFromFile(const std::filesystem::path& path) {
    // Same as PTX loading for now
    return loadPTXFromFile(path);
}

// Note: Pipeline::destroy() is already implemented inline in the header file
// as part of the Pipeline template struct definition. No separate implementation needed.

// ============================================================================
// Refactored helper functions for setupPipeline
// ============================================================================

void PipelineHandler::createOrGetPipeline(Pipeline::Ptr& pipeline, EntryPointType entryPoint) {
    if (!pipeline->optixPipeline) {
        LOG(INFO) << "Creating OptiX pipeline for entry point: " << static_cast<int>(entryPoint);
        pipeline->optixPipeline = pImpl->optixContext.createPipeline();
        pipeline->transitionTo(PipelineState::Uninitialized);
        LOG(INFO) << "OptiX pipeline created successfully";
    } else {
        LOG(INFO) << "Using existing OptiX pipeline for entry point: " << static_cast<int>(entryPoint);
    }
}

void PipelineHandler::configurePipelineOptions(Pipeline::Ptr& pipeline, const PipelineConfig& config) {
    pipeline->config = config;
    optixu::PipelineOptions options;
    options.payloadCountInDwords = config.numPayloadDwords;
    options.attributeCountInDwords = config.numAttributeDwords;
    options.launchParamsVariableName = config.launchParamsName.c_str();
    options.sizeOfLaunchParams = config.launchParamsSize;
    options.traversableGraphFlags = static_cast<OptixTraversableGraphFlags>(config.traversableGraphFlags);
    options.exceptionFlags = static_cast<OptixExceptionFlags>(config.exceptionFlags);
    options.supportedPrimitiveTypeFlags = static_cast<OptixPrimitiveTypeFlags>(config.primitiveTypeFlags);
    pipeline->optixPipeline.setPipelineOptions(options);
    pipeline->transitionTo(PipelineState::Configured);
}

void PipelineHandler::loadPipelineModule(Pipeline::Ptr& pipeline, const std::string& kernelName, EntryPointType entryPoint) {
    // EACH PIPELINE MUST HAVE ITS OWN MODULE - modules cannot be shared between pipelines
    PTXManager* ptxManager = pImpl->renderContext->getPTXManager();
    if (!ptxManager) {
        LOG(WARNING) << "PTXManager not available, cannot load module: " << kernelName;
        return;
    }
    
    std::vector<char> ptxData = ptxManager->getPTXData(kernelName);
    if (ptxData.empty()) {
        LOG(WARNING) << "Failed to load PTX data for kernel: " << kernelName;
        return;
    }
    
    // Create a unique module for THIS pipeline from PTX string
    std::string ptxString(ptxData.begin(), ptxData.end());
    pipeline->optixModule = pipeline->optixPipeline.createModuleFromPTXString(
        ptxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE)
    );
    
    LOG(INFO) << "Created module for pipeline " << static_cast<int>(entryPoint) 
              << " from kernel: " << kernelName;
    pipeline->transitionTo(PipelineState::ModuleLoaded);
}

void PipelineHandler::createPipelinePrograms(Pipeline::Ptr& pipeline, const PipelineData& data) {
    if (!pipeline->optixModule) {
        LOG(WARNING) << "Module not loaded, cannot create programs";
        return;
    }
    
    LOG(INFO) << "Creating programs for pipeline type: " << static_cast<int>(data.entryPoint);
    
    optixu::Module emptyModule; // For empty programs
    
    // Create ray generation program if specified
    if (!data.rayGenName.empty()) {
        auto rayGenProgram = pipeline->optixPipeline.createRayGenProgram(
            pipeline->optixModule, data.rayGenName.c_str());
        pipeline->entryPoints[data.entryPoint] = rayGenProgram;
        LOG(INFO) << "Created ray gen program: " << data.rayGenName;
    }
    
    // Create miss program if specified
    if (!data.missName.empty()) {
        auto missProgram = pipeline->optixPipeline.createMissProgram(
            pipeline->optixModule, data.missName.c_str());
        pipeline->programs[ProgramType::Miss] = missProgram;
        LOG(INFO) << "Created miss program: " << data.missName;
    }
    
    // Create closest hit program if specified
    if (!data.closestHitName.empty()) {
        auto hitGroup = pipeline->optixPipeline.createHitProgramGroupForTriangleIS(
            pipeline->optixModule, data.closestHitName.c_str(),
            emptyModule, nullptr);
        pipeline->hitGroups[data.closestHitName] = hitGroup;
        LOG(INFO) << "Created closest hit group: " << data.closestHitName;
    }
    
    // Create any hit program if specified (for visibility)
    if (!data.anyHitName.empty()) {
        auto visibilityGroup = pipeline->optixPipeline.createHitProgramGroupForTriangleIS(
            emptyModule, nullptr,
            pipeline->optixModule, data.anyHitName.c_str());
        pipeline->hitGroups[data.anyHitName] = visibilityGroup;
        LOG(INFO) << "Created any hit group: " << data.anyHitName;
    }
    
    // Set the entry point if we created one
    if (pipeline->entryPoints.find(data.entryPoint) != pipeline->entryPoints.end()) {
        pipeline->optixPipeline.setRayGenerationProgram(pipeline->entryPoints[data.entryPoint]);
        pipeline->currentEntryPoint = data.entryPoint;
    }
    
    pipeline->transitionTo(PipelineState::ProgramsCreated);
}

void PipelineHandler::setupPipelineRayTypes(Pipeline::Ptr& pipeline, const PipelineData& data) {
    // Set ray type information
    pipeline->numRayTypes = data.numRayTypes;
    pipeline->searchRayIndex = data.searchRayIndex;
    pipeline->visibilityRayIndex = data.visibilityRayIndex;
    
    // Setup ray types and miss programs (MUST be done BEFORE linking)
    setupRayTypes(data.entryPoint, data.numRayTypes);
    
    // Create empty hit group for GBuffer pipeline if needed (MUST be done BEFORE linking)
    if (data.entryPoint == EntryPointType::GBuffer) {
        // Create empty hit group for unused ray types
        auto emptyHitGroup = pipeline->optixPipeline.createEmptyHitProgramGroup();
        pipeline->hitGroups["emptyHitGroup"] = emptyHitGroup;
        LOG(INFO) << "Created empty hit group for GBuffer pipeline";
    }
}

void PipelineHandler::linkAndFinalizePipeline(Pipeline::Ptr& pipeline, const PipelineData& data) {
    // Link pipeline (must be done AFTER all programs are set)
    LOG(INFO) << "Linking pipeline for type: " << static_cast<int>(data.entryPoint);
    pipeline->optixPipeline.link(data.config.maxTraceDepth);
    pipeline->transitionTo(PipelineState::Linked);
    
    // Set the entry point
    if (pipeline->entryPoints.find(data.entryPoint) != pipeline->entryPoints.end()) {
        pipeline->setEntryPoint(data.entryPoint);
    }
    
    // Set the scene on the pipeline
    if (pImpl->renderContext) {
        optixu::Scene scene = pImpl->renderContext->getScene();
        if (scene) {
            LOG(INFO) << "Setting scene on pipeline";
            pipeline->optixPipeline.setScene(scene);
            pImpl->currentScene = scene;
        }
    }
    
    // Setup SBT (must be done after linking and scene setup)
    LOG(INFO) << "Setting up SBT for pipeline type: " << static_cast<int>(data.entryPoint);
    setupSBT(data.entryPoint);
    
    // Setup scene-dependent hit group SBT
    setSceneDependentSBT(data.entryPoint);
    pipeline->transitionTo(PipelineState::SBTReady);
    
    // Mark as ready
    pipeline->transitionTo(PipelineState::Ready);
}