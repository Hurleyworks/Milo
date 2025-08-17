// PipelineHandler manages OptiX ray tracing pipelines for the Dog rendering system.
// It provides centralized management of rendering pipelines, shader programs, and shader binding tables.
//
// Pipeline Management:
// - Manages G-buffer pipeline for geometry and material data extraction
// - Manages path tracing pipeline for Monte Carlo ray tracing
// - Handles shader binding table (SBT) generation and updates
// - Coordinates pipeline compilation and linking
//
// Program Organization:
// - Ray generation programs for different rendering modes
// - Hit programs for surface interactions
// - Miss programs for environment sampling
// - Callable programs for material evaluation
//
// Memory Management:
// - Automatic SBT memory allocation and management
// - Pipeline state cleanup on destruction
// - Support for dynamic pipeline reconfiguration
// - RAII principles for resource safety
//
// Integration:
// - Works with GPUContext for OptiX access
// - Integrates with material and scene systems
// - Supports multiple rendering modes
// - Provides launch methods for pipeline execution
//
// Usage:
// - Create via factory method PipelineHandler::create()
// - Initialize with render context
// - Set active pipeline and entry points
// - Launch pipelines for rendering
//
// Thread Safety:
// - Not thread-safe by default
// - Requires external synchronization for multi-threaded access
// - Pipeline operations should be synchronized with rendering

#pragma once

#include "../common/common_host.h"
#include "../DogShared.h"

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;

namespace dog
{

using PipelineHandlerPtr = std::shared_ptr<class PipelineHandler>;

// PipelineHandler manages OptiX rendering pipelines and shader binding tables
// Provides centralized pipeline lifecycle management with automatic cleanup
class PipelineHandler
{
public:
    // Entry point enums for different pipeline modes
    enum class GBufferEntryPoint
    {
        setupGBuffers = 0,
    };

    enum class PathTracingEntryPoint
    {
        pathTraceBaseline = 0,
    };

    // Factory method to create a shared PipelineHandler instance
    static PipelineHandlerPtr create(RenderContextPtr ctx)
    {
        return std::make_shared<PipelineHandler>(ctx);
    }

    PipelineHandler(RenderContextPtr ctx);
    ~PipelineHandler();

    PipelineHandler(const PipelineHandler&) = delete;
    PipelineHandler& operator=(const PipelineHandler&) = delete;
    PipelineHandler(PipelineHandler&&) = default;
    PipelineHandler& operator=(PipelineHandler&&) = default;

    // Initialize pipelines with PTX module paths
    // Returns true if successful, false otherwise
    bool initialize(const std::filesystem::path& gbufferPtxPath,
                   const std::filesystem::path& pathTracingPtxPath);

    // Clean up all pipeline resources
    void finalize();

    // Check if pipelines are initialized
    bool isInitialized() const { return initialized_; }

    // G-buffer pipeline operations
    void launchGBufferPipeline(CUstream stream, CUdeviceptr plpOnDevice, 
                              uint32_t width, uint32_t height);
    bool hasGBufferHitGroupSbt() const;
    size_t getGBufferHitGroupSbtSize() const;
    void initializeGBufferHitGroupSbt(size_t sbtSize);
    void setGBufferScene(optixu::Scene scene);
    void setGBufferEntryPoint(GBufferEntryPoint entryPoint);

    // Path tracing pipeline operations
    void launchPathTracingPipeline(CUstream stream, CUdeviceptr plpOnDevice,
                                  uint32_t width, uint32_t height);
    bool hasPathTracingHitGroupSbt() const;
    size_t getPathTracingHitGroupSbtSize() const;
    void initializePathTracingHitGroupSbt(size_t sbtSize);
    void setPathTracingScene(optixu::Scene scene);
    void setPathTracingEntryPoint(PathTracingEntryPoint entryPoint);

    // Get callable program count for material system
    size_t getNumCallablePrograms() const { return callable_program_names_.size(); }

    // Update default material hit groups
    void updateMaterialHitGroups(optixu::Material& material);

private:
    // Pipeline template structure for code reuse
    template <typename EntryPointType>
    struct Pipeline
    {
        optixu::Pipeline optixPipeline;
        optixu::Module optixModule;
        std::unordered_map<EntryPointType, optixu::Program> entryPoints;
        std::unordered_map<std::string, optixu::Program> programs;
        std::unordered_map<std::string, optixu::HitProgramGroup> hitPrograms;
        std::vector<optixu::CallableProgramGroup> callablePrograms;
        cudau::Buffer sbt;
        cudau::Buffer hitGroupSbt;

        void setEntryPoint(EntryPointType et)
        {
            if (optixPipeline && entryPoints.count(et) > 0)
            {
                optixPipeline.setRayGenerationProgram(entryPoints.at(et));
            }
        }

        void finalize();
    };

    // Internal initialization methods
    bool initializeGBufferPipeline(const std::filesystem::path& ptxPath);
    bool initializePathTracingPipeline(const std::filesystem::path& ptxPath);
    bool setupCallablePrograms(optixu::Pipeline& pipeline, optixu::Module& module,
                              std::vector<optixu::CallableProgramGroup>& callablePrograms);
    std::string readPtxFile(const std::filesystem::path& path);

    // Member variables
    RenderContextPtr render_context_;
    bool initialized_ = false;

    Pipeline<GBufferEntryPoint> gbuffer_pipeline_;
    Pipeline<PathTracingEntryPoint> pathtracing_pipeline_;

    // Callable program names for material evaluation
    std::vector<const char*> callable_program_names_;
};

// Template implementation
template <typename EntryPointType>
void PipelineHandler::Pipeline<EntryPointType>::finalize()
{
    hitGroupSbt.finalize();
    sbt.finalize();
    
    for (auto& program : callablePrograms)
    {
        if (program)
            program.destroy();
    }
    callablePrograms.clear();
    
    for (auto& [name, program] : programs)
    {
        if (program)
            program.destroy();
    }
    programs.clear();
    
    for (auto& [name, hitProgram] : hitPrograms)
    {
        if (hitProgram)
            hitProgram.destroy();
    }
    hitPrograms.clear();
    
    for (auto& [type, entryPoint] : entryPoints)
    {
        if (entryPoint)
            entryPoint.destroy();
    }
    entryPoints.clear();
    
    if (optixModule)
        optixModule.destroy();
    
    if (optixPipeline)
        optixPipeline.destroy();
}

} // namespace dog