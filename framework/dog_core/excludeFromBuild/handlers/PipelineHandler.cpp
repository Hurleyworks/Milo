#include "PipelineHandler.h"
#include "../RenderContext.h"
#include <fstream>
#include <sstream>

namespace dog
{

// Define callable program entry points for material evaluation
static const char* g_callableProgramNames[] = {
    "__direct_callable__setupPrimaryRay",
    "__direct_callable__sampleDiffuseBSDF",
    "__direct_callable__evaluateDiffuseBSDF",
    "__direct_callable__sampleSpecularBSDF",
    "__direct_callable__evaluateSpecularBSDF",
    "__direct_callable__sampleCoatBSDF",
    "__direct_callable__evaluateCoatBSDF",
    "__direct_callable__setupBSDF",
    "__direct_callable__sampleBSDF",
    "__direct_callable__evaluateBSDF",
    "__direct_callable__evaluateDirectionalPDF",
    "__direct_callable__sampleEmitter",
    nullptr
};

static constexpr int NumCallablePrograms = sizeof(g_callableProgramNames) / sizeof(char*) - 1;

PipelineHandler::PipelineHandler(RenderContextPtr ctx)
    : render_context_(ctx)
{
    // Initialize callable program names
    for (int i = 0; g_callableProgramNames[i] != nullptr; ++i)
    {
        callable_program_names_.push_back(g_callableProgramNames[i]);
    }
}

PipelineHandler::~PipelineHandler()
{
    finalize();
}

bool PipelineHandler::initialize(const std::filesystem::path& gbufferPtxPath,
                                 const std::filesystem::path& pathTracingPtxPath)
{
    if (initialized_)
    {
        LOG(WARNING) << "PipelineHandler already initialized";
        return true;
    }

    if (!render_context_ || !render_context_->isInitialized())
    {
        LOG(WARNING) << "RenderContext not initialized";
        return false;
    }

    // Initialize G-buffer pipeline
    if (!initializeGBufferPipeline(gbufferPtxPath))
    {
        LOG(WARNING) << "Failed to initialize G-buffer pipeline";
        return false;
    }

    // Initialize path tracing pipeline
    if (!initializePathTracingPipeline(pathTracingPtxPath))
    {
        LOG(WARNING) << "Failed to initialize path tracing pipeline";
        gbuffer_pipeline_.finalize();
        return false;
    }

    initialized_ = true;
    LOG(INFO) << "PipelineHandler initialized successfully";
    return true;
}

void PipelineHandler::finalize()
{
    if (!initialized_)
        return;

    pathtracing_pipeline_.finalize();
    gbuffer_pipeline_.finalize();

    initialized_ = false;
    LOG(INFO) << "PipelineHandler finalized";
}

bool PipelineHandler::initializeGBufferPipeline(const std::filesystem::path& ptxPath)
{
    CUcontext cuContext = render_context_->getCudaContext();
    optixu::Context optixContext = render_context_->getOptixContext();

    Pipeline<GBufferEntryPoint>& pipeline = gbuffer_pipeline_;
    optixu::Pipeline& p = pipeline.optixPipeline;
    optixu::Module& m = pipeline.optixModule;
    optixu::Module emptyModule;

    // Create pipeline
    p = optixContext.createPipeline();

    // Set pipeline options
    p.setPipelineOptions(
        std::max({DogShared::PrimaryRayPayloadSignature::numDwords}),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(DogShared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    // Load PTX module
    std::string ptxString = readPtxFile(ptxPath);
    if (ptxString.empty())
    {
        LOG(WARNING) << "Failed to read G-buffer PTX file: " << ptxPath.string();
        return false;
    }

    m = p.createModuleFromPTXString(
        ptxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        OPTIX_COMPILE_DEBUG_LEVEL_NONE);

    // Create ray generation program
    pipeline.entryPoints[GBufferEntryPoint::setupGBuffers] = 
        p.createRayGenProgram(m, "__raygen__setupGBuffers");

    // Create hit program groups
    pipeline.hitPrograms["hitgroup"] = p.createHitProgramGroupForTriangleIS(
        m, "__closesthit__setupGBuffers",
        emptyModule, nullptr);
    
    pipeline.programs["miss"] = p.createMissProgram(
        m, "__miss__setupGBuffers");

    pipeline.hitPrograms["emptyHitGroup"] = p.createEmptyHitProgramGroup();

    // Set initial entry point
    pipeline.setEntryPoint(GBufferEntryPoint::setupGBuffers);
    
    // Configure ray types
    p.setNumMissRayTypes(DogShared::GBufferRayType::NumTypes);
    p.setMissProgram(DogShared::GBufferRayType::Primary, pipeline.programs.at("miss"));

    // Setup callable programs
    if (!setupCallablePrograms(p, m, pipeline.callablePrograms))
    {
        LOG(WARNING) << "Failed to setup callable programs for G-buffer pipeline";
        return false;
    }

    // Link pipeline
    p.link(1);

    // Calculate stack sizes
    uint32_t maxDcStackSize = 0;
    for (size_t i = 0; i < pipeline.callablePrograms.size(); ++i)
    {
        if (pipeline.callablePrograms[i])
        {
            maxDcStackSize = std::max(maxDcStackSize, 
                                     pipeline.callablePrograms[i].getDCStackSize());
        }
    }

    uint32_t maxCcStackSize =
        pipeline.entryPoints.at(GBufferEntryPoint::setupGBuffers).getStackSize() +
        std::max({pipeline.hitPrograms.at("hitgroup").getCHStackSize(),
                 pipeline.programs.at("miss").getStackSize()});

    p.setStackSize(0, maxDcStackSize, maxCcStackSize, 2);

    // Initialize shader binding table
    size_t sbtSize;
    p.generateShaderBindingTableLayout(&sbtSize);
    pipeline.sbt.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    pipeline.sbt.setMappedMemoryPersistent(true);
    p.setShaderBindingTable(pipeline.sbt, pipeline.sbt.getMappedPointer());

    LOG(DBUG) << "G-buffer pipeline initialized successfully";
    return true;
}

bool PipelineHandler::initializePathTracingPipeline(const std::filesystem::path& ptxPath)
{
    CUcontext cuContext = render_context_->getCudaContext();
    optixu::Context optixContext = render_context_->getOptixContext();

    Pipeline<PathTracingEntryPoint>& pipeline = pathtracing_pipeline_;
    optixu::Pipeline& p = pipeline.optixPipeline;
    optixu::Module& m = pipeline.optixModule;
    optixu::Module emptyModule;

    // Create pipeline
    p = optixContext.createPipeline();

    // Set pipeline options
    p.setPipelineOptions(
        std::max({DogShared::PathTraceRayPayloadSignature::numDwords,
                 DogShared::VisibilityRayPayloadSignature::numDwords}),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(DogShared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    // Load PTX module
    std::string ptxString = readPtxFile(ptxPath);
    if (ptxString.empty())
    {
        LOG(WARNING) << "Failed to read path tracing PTX file: " << ptxPath;
        return false;
    }

    m = p.createModuleFromPTXString(
        ptxString,
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        OPTIX_COMPILE_DEBUG_LEVEL_NONE);

    // Create ray generation program
    pipeline.entryPoints[PathTracingEntryPoint::pathTraceBaseline] =
        p.createRayGenProgram(m, "__raygen__pathTraceBaseline");

    // Create miss program
    pipeline.programs["pathTraceMiss"] = p.createMissProgram(
        m, "__miss__pathTraceBaseline");

    // Create hit program groups
    pipeline.hitPrograms["pathTraceHit"] = p.createHitProgramGroupForTriangleIS(
        m, "__closesthit__pathTraceBaseline",
        emptyModule, nullptr);

    pipeline.hitPrograms["visibility"] = p.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        m, "__anyhit__visibility");

    pipeline.programs["emptyMiss"] = p.createMissProgram(emptyModule, nullptr);

    // Configure ray types
    p.setNumMissRayTypes(DogShared::PathTracingRayType::NumTypes);
    p.setMissProgram(DogShared::PathTracingRayType::Closest, 
                    pipeline.programs.at("pathTraceMiss"));
    p.setMissProgram(DogShared::PathTracingRayType::Visibility, 
                    pipeline.programs.at("emptyMiss"));

    // Setup callable programs
    if (!setupCallablePrograms(p, m, pipeline.callablePrograms))
    {
        LOG(WARNING) << "Failed to setup callable programs for path tracing pipeline";
        return false;
    }

    // Link pipeline
    p.link(2);

    // Calculate stack sizes
    uint32_t maxDcStackSize = 0;
    for (size_t i = 0; i < pipeline.callablePrograms.size(); ++i)
    {
        if (pipeline.callablePrograms[i])
        {
            maxDcStackSize = std::max(maxDcStackSize,
                                     pipeline.callablePrograms[i].getDCStackSize());
        }
    }

    uint32_t maxCcStackSize =
        pipeline.entryPoints.at(PathTracingEntryPoint::pathTraceBaseline).getStackSize() +
        std::max({pipeline.hitPrograms.at("pathTraceHit").getCHStackSize() +
                 pipeline.hitPrograms.at("visibility").getAHStackSize(),
                 pipeline.programs.at("pathTraceMiss").getStackSize()});

    p.setStackSize(0, maxDcStackSize, maxCcStackSize, 2);

    // Initialize shader binding table
    size_t sbtSize;
    p.generateShaderBindingTableLayout(&sbtSize);
    pipeline.sbt.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    pipeline.sbt.setMappedMemoryPersistent(true);
    p.setShaderBindingTable(pipeline.sbt, pipeline.sbt.getMappedPointer());

    LOG(DBUG) << "Path tracing pipeline initialized successfully";
    return true;
}

bool PipelineHandler::setupCallablePrograms(optixu::Pipeline& pipeline, optixu::Module& module,
                                           std::vector<optixu::CallableProgramGroup>& callablePrograms)
{
    optixu::Module emptyModule;

    pipeline.setNumCallablePrograms(static_cast<uint32_t>(callable_program_names_.size()));
    callablePrograms.resize(callable_program_names_.size());

    for (size_t i = 0; i < callable_program_names_.size(); ++i)
    {
        if (callable_program_names_[i] != nullptr)
        {
            try
            {
                optixu::CallableProgramGroup program = pipeline.createCallableProgramGroup(
                    module, callable_program_names_[i],
                    emptyModule, nullptr);
                callablePrograms[i] = program;
                pipeline.setCallableProgram(static_cast<uint32_t>(i), program);
            }
            catch (const std::exception& e)
            {
                LOG(WARNING) << "Failed to create callable program " << callable_program_names_[i]
                           << ": " << e.what();
                // Continue - some callable programs may be optional
            }
        }
    }

    return true;
}

std::string PipelineHandler::readPtxFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        LOG(WARNING) << "Failed to open PTX file: " << path;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// G-buffer pipeline operations
void PipelineHandler::launchGBufferPipeline(CUstream stream, CUdeviceptr plpOnDevice,
                                           uint32_t width, uint32_t height)
{
    if (gbuffer_pipeline_.optixPipeline)
    {
        gbuffer_pipeline_.optixPipeline.launch(stream, plpOnDevice, width, height, 1);
    }
}

bool PipelineHandler::hasGBufferHitGroupSbt() const
{
    return gbuffer_pipeline_.hitGroupSbt.isInitialized();
}

size_t PipelineHandler::getGBufferHitGroupSbtSize() const
{
    return gbuffer_pipeline_.hitGroupSbt.sizeInBytes();
}

void PipelineHandler::initializeGBufferHitGroupSbt(size_t sbtSize)
{
    CUcontext cuContext = render_context_->getCudaContext();
    gbuffer_pipeline_.hitGroupSbt.finalize();
    gbuffer_pipeline_.hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    gbuffer_pipeline_.hitGroupSbt.setMappedMemoryPersistent(true);
}

void PipelineHandler::setGBufferScene(optixu::Scene scene)
{
    if (gbuffer_pipeline_.optixPipeline)
    {
        gbuffer_pipeline_.optixPipeline.setScene(scene);
        gbuffer_pipeline_.optixPipeline.setHitGroupShaderBindingTable(
            gbuffer_pipeline_.hitGroupSbt, gbuffer_pipeline_.hitGroupSbt.getMappedPointer());
    }
}

void PipelineHandler::setGBufferEntryPoint(GBufferEntryPoint entryPoint)
{
    gbuffer_pipeline_.setEntryPoint(entryPoint);
}

// Path tracing pipeline operations
void PipelineHandler::launchPathTracingPipeline(CUstream stream, CUdeviceptr plpOnDevice,
                                               uint32_t width, uint32_t height)
{
    if (pathtracing_pipeline_.optixPipeline)
    {
        pathtracing_pipeline_.setEntryPoint(PathTracingEntryPoint::pathTraceBaseline);
        pathtracing_pipeline_.optixPipeline.launch(stream, plpOnDevice, width, height, 1);
    }
}

bool PipelineHandler::hasPathTracingHitGroupSbt() const
{
    return pathtracing_pipeline_.hitGroupSbt.isInitialized();
}

size_t PipelineHandler::getPathTracingHitGroupSbtSize() const
{
    return pathtracing_pipeline_.hitGroupSbt.sizeInBytes();
}

void PipelineHandler::initializePathTracingHitGroupSbt(size_t sbtSize)
{
    CUcontext cuContext = render_context_->getCudaContext();
    pathtracing_pipeline_.hitGroupSbt.finalize();
    pathtracing_pipeline_.hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    pathtracing_pipeline_.hitGroupSbt.setMappedMemoryPersistent(true);
}

void PipelineHandler::setPathTracingScene(optixu::Scene scene)
{
    if (pathtracing_pipeline_.optixPipeline)
    {
        pathtracing_pipeline_.optixPipeline.setScene(scene);
        pathtracing_pipeline_.optixPipeline.setHitGroupShaderBindingTable(
            pathtracing_pipeline_.hitGroupSbt, pathtracing_pipeline_.hitGroupSbt.getMappedPointer());
    }
}

void PipelineHandler::setPathTracingEntryPoint(PathTracingEntryPoint entryPoint)
{
    pathtracing_pipeline_.setEntryPoint(entryPoint);
}

// Update material hit groups
void PipelineHandler::updateMaterialHitGroups(optixu::Material& material)
{
    // Configure material hit groups for G-buffer pipeline
    if (gbuffer_pipeline_.hitPrograms.count("hitgroup") > 0)
    {
        material.setHitGroup(DogShared::GBufferRayType::Primary,
                           gbuffer_pipeline_.hitPrograms.at("hitgroup"));
    }
    
    // Set empty hit groups for unused ray types
    if (gbuffer_pipeline_.hitPrograms.count("emptyHitGroup") > 0)
    {
        for (uint32_t rayType = DogShared::GBufferRayType::NumTypes; 
             rayType < DogShared::maxNumRayTypes; ++rayType)
        {
            material.setHitGroup(rayType, gbuffer_pipeline_.hitPrograms.at("emptyHitGroup"));
        }
    }

    // Configure material hit groups for path tracing pipeline
    if (pathtracing_pipeline_.hitPrograms.count("pathTraceHit") > 0)
    {
        material.setHitGroup(DogShared::PathTracingRayType::Closest,
                           pathtracing_pipeline_.hitPrograms.at("pathTraceHit"));
    }
    
    if (pathtracing_pipeline_.hitPrograms.count("visibility") > 0)
    {
        material.setHitGroup(DogShared::PathTracingRayType::Visibility,
                           pathtracing_pipeline_.hitPrograms.at("visibility"));
    }
}

} // namespace dog