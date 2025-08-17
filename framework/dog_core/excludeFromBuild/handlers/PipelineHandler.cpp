#include "PipelineHandler.h"
#include "../RenderContext.h"

namespace dog
{

PipelineHandler::PipelineHandler(RenderContextPtr ctx)
    : render_context_(ctx)
{
    // Initialize PTXManager
    ptx_manager_ = std::make_unique<PTXManager>();
    if (render_context_)
    {
        // Initialize with resource path from render context
        std::filesystem::path resourcePath = render_context_->getResourcePath();
        if (!resourcePath.empty())
        {
            ptx_manager_->initialize(resourcePath / "ptx");
            LOG(INFO) << "PTXManager initialized with resource path: " << (resourcePath / "ptx").string();
        }
        else
        {
            LOG(WARNING) << "Resource path not available from RenderContext";
        }
    }
}

PipelineHandler::~PipelineHandler()
{
    finalize();
}

bool PipelineHandler::initialize(const std::string& gbufferKernelName,
                                 const std::string& pathTracingKernelName)
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

    if (!ptx_manager_)
    {
        LOG(WARNING) << "PTXManager not initialized";
        return false;
    }

    // Initialize G-buffer pipeline
    if (!initializeGBufferPipeline(gbufferKernelName))
    {
        LOG(WARNING) << "Failed to initialize G-buffer pipeline";
        return false;
    }

    // Initialize path tracing pipeline
    if (!initializePathTracingPipeline(pathTracingKernelName))
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

bool PipelineHandler::initializeGBufferPipeline(const std::string& kernelName)
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

    // Load PTX using PTXManager
    std::vector<char> ptxData = getPTXData(kernelName);
    if (ptxData.empty())
    {
        LOG(WARNING) << "Failed to load PTX for G-buffer kernel: " << kernelName;
        return false;
    }

    // Convert to string for OptiX
    std::string ptxString(ptxData.begin(), ptxData.end());

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

    // Link pipeline
    p.link(1);

    // Calculate stack sizes
    uint32_t maxDcStackSize = 0;

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

bool PipelineHandler::initializePathTracingPipeline(const std::string& kernelName)
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

    // Load PTX using PTXManager
    std::vector<char> ptxData = getPTXData(kernelName);
    if (ptxData.empty())
    {
        LOG(WARNING) << "Failed to load PTX for path tracing kernel: " << kernelName;
        return false;
    }

    // Convert to string for OptiX
    std::string ptxString(ptxData.begin(), ptxData.end());

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

    // Link pipeline
    p.link(2);

    // Calculate stack sizes
    uint32_t maxDcStackSize = 0;

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


std::vector<char> PipelineHandler::getPTXData(const std::string& kernelName, bool useEmbedded)
{
    if (!ptx_manager_)
    {
        LOG(WARNING) << "PTXManager not initialized";
        return {};
    }
    
    try
    {
        return ptx_manager_->getPTXData(kernelName, useEmbedded);
    }
    catch (const std::exception& e)
    {
        LOG(WARNING) << "Failed to get PTX data for kernel " << kernelName << ": " << e.what();
        return {};
    }
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