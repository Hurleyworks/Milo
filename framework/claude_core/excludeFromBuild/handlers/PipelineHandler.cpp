#include "PipelineHandler.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

// Private implementation structure
struct PipelineHandler::Impl {
    RenderContextPtr renderContext;
    optixu::Context optixContext;
    CUcontext cuContext = nullptr;
    
    // Storage for all pipelines
    std::unordered_map<EntryPointType, Pipeline<EntryPointType>::Ptr> pipelines;
    
    // Shared resources
    std::unordered_map<std::string, optixu::Module> moduleCache;
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
    // TODO: Implement pipeline setup
    // 1. Create or get pipeline for the entry point type
    auto pipeline = createPipeline(data.entryPoint);
    
    // 2. Create pipeline if not exists
    if (!pipeline->optixPipeline) {
        pipeline->optixPipeline = pImpl->optixContext.createPipeline();
    }
    
    // 3. Set pipeline options from config
    pipeline->config = data.config;
    pipeline->optixPipeline.setPipelineOptions(
        data.config.numPayloadDwords,
        data.config.numAttributeDwords,
        data.config.launchParamsName.c_str(),
        data.config.launchParamsSize,
        static_cast<OptixTraversableGraphFlags>(data.config.traversableGraphFlags),
        static_cast<OptixExceptionFlags>(data.config.exceptionFlags),
        static_cast<OptixPrimitiveTypeFlags>(data.config.primitiveTypeFlags)
    );
    
    // 4. Load/create module
    // TODO: Load module from kernel name
    
    // 5. Create programs
    // TODO: Create ray gen, miss, hit programs based on data
    
    // 6. Set ray type information
    pipeline->numRayTypes = data.numRayTypes;
    pipeline->searchRayIndex = data.searchRayIndex;
    pipeline->visibilityRayIndex = data.visibilityRayIndex;
    
    // 7. Link pipeline
    linkPipeline(data.entryPoint, data.config.maxTraceDepth);
    
    pipeline->isInitialized = true;
    pipeline->sbtDirty = true;
}

// Setup pipeline from PTX file
void PipelineHandler::setupPipeline(const PipelineData& data, const std::filesystem::path& ptxFile) {
    std::string kernelName = ptxFile.stem().string();
    setupPipeline(data, kernelName);
}

// Setup shader binding table
void PipelineHandler::setupSBT(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline) {
        throw std::runtime_error("Pipeline not found for SBT setup");
    }
    
    // TODO: Implement SBT setup
    // 1. Generate SBT layout
    generateSBTLayout(type);
    
    // 2. Allocate SBT buffer if needed
    if (!pipeline->shaderBindingTable.isInitialized()) {
        size_t sbtSize = 0;
        pipeline->optixPipeline.generateShaderBindingTableLayout(&sbtSize);
        pipeline->shaderBindingTable.initialize(
            pImpl->cuContext, cudau::BufferType::Device, sbtSize, 1);
        pipeline->shaderBindingTable.setMappedMemoryPersistent(true);
    }
    
    // 3. Set SBT on pipeline
    pipeline->optixPipeline.setShaderBindingTable(
        pipeline->shaderBindingTable, 
        pipeline->shaderBindingTable.getMappedPointer());
    
    pipeline->sbtDirty = false;
}

// Update SBT
void PipelineHandler::updateSBT(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline) return;
    
    pipeline->sbtDirty = true;
    setupSBT(type);
}

// Set scene-dependent SBT
void PipelineHandler::setSceneDependentSBT(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline || !pImpl->currentScene) return;
    
    // TODO: Implement scene-dependent SBT setup
    // 1. Get SBT size from scene
    size_t hitGroupSbtSize = 0;
    pImpl->currentScene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    
    // 2. Allocate hit group SBT if needed
    if (!pipeline->hitGroupSBT.isInitialized() || 
        pipeline->hitGroupSBT.sizeInBytes() != hitGroupSbtSize) {
        pipeline->hitGroupSBT.finalize();
        pipeline->hitGroupSBT.initialize(
            pImpl->cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
        pipeline->hitGroupSBT.setMappedMemoryPersistent(true);
    }
    
    // 3. Set hit group SBT on pipeline
    pipeline->optixPipeline.setHitGroupShaderBindingTable(
        pipeline->hitGroupSBT,
        pipeline->hitGroupSBT.getMappedPointer());
}

// Get pipeline
Pipeline<EntryPointType>::Ptr PipelineHandler::getPipeline(EntryPointType type) {
    auto it = pImpl->pipelines.find(type);
    return (it != pImpl->pipelines.end()) ? it->second : nullptr;
}

// Check if pipeline exists
bool PipelineHandler::hasPipeline(EntryPointType type) const {
    return pImpl->pipelines.find(type) != pImpl->pipelines.end();
}

// Launch pipeline
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
    for (auto& [type, pipeline] : pImpl->pipelines) {
        if (pipeline && pipeline->optixPipeline) {
            pipeline->optixPipeline.setScene(scene);
        }
    }
}

// Set scene for specific pipeline
void PipelineHandler::setScene(EntryPointType type, optixu::Scene scene) {
    auto pipeline = getPipeline(type);
    if (pipeline && pipeline->optixPipeline) {
        pipeline->optixPipeline.setScene(scene);
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

// Load module from data
optixu::Module PipelineHandler::loadModule(const std::string& name, 
                                           const std::vector<char>& data) {
    // Check cache first
    auto it = pImpl->moduleCache.find(name);
    if (it != pImpl->moduleCache.end()) {
        return it->second;
    }
    
    // Create new module
    // TODO: Determine if data is PTX or OptiX IR and create accordingly
    optixu::Module module;
    // module = pipeline->optixPipeline.createModuleFromOptixIR(...);
    
    pImpl->moduleCache[name] = module;
    return module;
}

// Load module from file
optixu::Module PipelineHandler::loadModule(const std::filesystem::path& ptxFile) {
    auto data = loadPTXFromFile(ptxFile);
    return loadModule(ptxFile.stem().string(), data);
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
        if (pipeline->isInitialized) {
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
    pImpl->moduleCache.clear();
}

// Private: Create pipeline
Pipeline<EntryPointType>::Ptr PipelineHandler::createPipeline(EntryPointType type) {
    auto it = pImpl->pipelines.find(type);
    if (it != pImpl->pipelines.end()) {
        return it->second;
    }
    
    auto pipeline = std::make_shared<Pipeline<EntryPointType>>();
    pImpl->pipelines[type] = pipeline;
    return pipeline;
}

// Private: Link pipeline
void PipelineHandler::linkPipeline(EntryPointType type, uint32_t maxTraceDepth) {
    auto pipeline = getPipeline(type);
    if (pipeline && pipeline->optixPipeline) {
        pipeline->optixPipeline.link(maxTraceDepth);
    }
}

// Private: Generate SBT layout
void PipelineHandler::generateSBTLayout(EntryPointType type) {
    auto pipeline = getPipeline(type);
    if (!pipeline || !pipeline->optixPipeline) return;
    
    // Set ray type configuration
    pipeline->optixPipeline.setNumMissRayTypes(pipeline->numRayTypes);
    
    // Set miss programs for each ray type
    for (const auto& [progType, program] : pipeline->programs) {
        if (progType == ProgramType::Miss) {
            // TODO: Set miss program for appropriate ray types
            // pipeline->optixPipeline.setMissProgram(rayType, program);
        }
    }
    
    // Set callable programs if any
    if (!pipeline->callablePrograms.empty()) {
        // Note: OptiX pipelines don't have a setCallableProgramCount method
        // Set each callable program individually
        for (size_t i = 0; i < pipeline->callablePrograms.size(); ++i) {
            pipeline->optixPipeline.setCallableProgram(i, pipeline->callablePrograms[i]);
        }
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