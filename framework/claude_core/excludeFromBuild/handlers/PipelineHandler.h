#pragma once

// ====================================================================================
// OptiX 9 Pipeline Overview
// ====================================================================================
//
// In OptiX 9, a Pipeline is the core object that encapsulates the entire ray tracing
// program execution configuration. It represents a complete ray tracing "shader program"
// that defines how rays are generated, traced, and shaded in your application.
//
// Key Components of an OptiX Pipeline:
//
// 1. MODULES
//    - Contain compiled GPU code (PTX or OptiX IR format)
//    - Similar to shader objects in traditional graphics APIs
//    - Multiple modules can be linked into a single pipeline
//
// 2. PROGRAMS
//    - Ray Generation Programs: Entry points that spawn primary rays
//    - Miss Programs: Execute when rays don't hit any geometry
//    - Exception Programs: Handle errors during traversal
//    - Closest Hit Programs: Execute at ray-geometry intersection points
//    - Any Hit Programs: Execute at any potential intersection (for alpha testing)
//    - Intersection Programs: Custom primitive intersection tests
//    - Direct/Continuation Callable Programs: Callable functions for shader reuse
//
// 3. HIT PROGRAM GROUPS
//    - Combine Closest Hit, Any Hit, and Intersection programs
//    - Define the complete behavior for a ray-geometry interaction
//    - Associated with materials in the scene
//
// 4. SHADER BINDING TABLE (SBT)
//    - GPU memory layout that maps programs to their data
//    - Contains program headers and per-instance data
//    - Split into sections: RayGen, Miss, HitGroup, Callable
//    - Critical for performance - proper alignment and layout matters
//
// 5. PIPELINE CONFIGURATION
//    - Payload size: Data carried by rays (colors, recursion depth, etc.)
//    - Attribute size: Data passed from intersection to shading
//    - Max trace depth: Recursion limit for ray tracing
//    - Traversable graph flags: What acceleration structures to support
//    - Exception flags: Which errors to catch
//    - Primitive type flags: Triangle, curves, custom primitives, spheres
//
// 6. LAUNCH PARAMETERS
//    - User-defined struct passed to all programs
//    - Contains global data like camera parameters, light data, buffer pointers
//    - Accessed via optixGetLaunchParams() in device code
//
// Pipeline Lifecycle:
// 1. Create Pipeline object from Context
// 2. Set pipeline options (payload size, traversal settings, etc.)
// 3. Create and attach Modules containing GPU code
// 4. Create Programs from module entry points
// 5. Create HitProgramGroups combining hit/intersection programs
// 6. Link the pipeline (compile and optimize)
// 7. Calculate stack sizes for recursive ray tracing
// 8. Generate and fill Shader Binding Table
// 9. Associate with Scene containing geometry
// 10. Launch with dimensions and parameters
//
// Why Multiple Pipelines?
// Different rendering techniques often require different pipeline configurations:
// - GBuffer pass: Simple primary ray intersection, writes geometric data
// - Path tracing: Complex recursive tracing with material evaluation
// - Picking: Minimal payload for object selection
// - Shadow rays: Optimized any-hit programs for visibility
//
// Each pipeline can be optimized for its specific use case, with only the
// necessary programs and minimal payload sizes for better performance.
//
// ====================================================================================

// PipelineHandler.h
// Manages multiple OptiX pipelines with a clean, unified interface
// 
// Key responsibilities:
// - Creates and manages multiple pipelines for different rendering modes (GBuffer, PathTrace, etc.)
// - Ensures each pipeline has its own module (modules cannot be shared between pipelines)
// - Manages pipeline state transitions through a clear state machine
// - Handles shader binding table (SBT) setup and updates
// - Coordinates scene assignment across all pipelines
// - Provides hit group configuration for materials
//
// Design principles:
// - Each pipeline owns its resources (module, programs, SBT)
// - State transitions are explicit and trackable via PipelineState enum
// - Pipeline setup is broken into focused, testable steps
// - Scene is shared across all pipelines (one scene per OptiX context)



#include "../RenderContext.h"

// Forward declarations
class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
class PipelineHandler;
using PipelineHandlerPtr = std::shared_ptr<PipelineHandler>;

// Shared enums for all pipelines
enum class EntryPointType {
    GBuffer,
    PathTrace,
    PathTraceProgressive,
    Pick,
    Debug,
    // Add more as needed
};

enum class ProgramType {
    Shading,
    Visibility,
    Miss,
    Exception,
    // Add more as needed
};

// Pipeline configuration (similar to ShockerEngine)
struct PipelineConfig {
    uint32_t maxTraceDepth = 2;
    uint32_t numPayloadDwords = 3;
    uint32_t numAttributeDwords = 2;
    std::string launchParamsName = "plp";
    size_t launchParamsSize = 0;
    uint32_t traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    uint32_t exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    uint32_t primitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    
    // Compilation options
    uint32_t maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    OptixCompileOptimizationLevel optimizationLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    OptixCompileDebugLevel debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
};

// Data for setting up a complete pipeline
struct PipelineData {
    EntryPointType entryPoint;
    std::string rayGenName;
    std::string missName;
    std::string closestHitName;
    std::string anyHitName;
    std::string exceptionName;
    uint32_t numRayTypes = 1;
    uint32_t searchRayIndex = 0;
    uint32_t visibilityRayIndex = 1;
    PipelineConfig config;
};

// Pipeline state enum for clearer state management
enum class PipelineState {
    Uninitialized,    // Pipeline created but not configured
    Configured,       // Pipeline options set
    ModuleLoaded,     // PTX module loaded
    ProgramsCreated,  // Programs created from module
    Linked,           // Pipeline linked
    SBTReady,         // SBT allocated and configured
    Ready            // Ready to launch
};

// Individual pipeline structure (simplified without template)
struct Pipeline {
    using Ptr = std::shared_ptr<Pipeline>;
    
    // Core OptiX objects
    optixu::Pipeline optixPipeline;
    optixu::Module optixModule;
    std::vector<optixu::Module> additionalModules;  // Support multiple modules
    
    // Programs organized by type
    std::unordered_map<EntryPointType, optixu::Program> entryPoints;
    std::unordered_map<ProgramType, optixu::Program> programs;
    std::unordered_map<std::string, optixu::HitProgramGroup> hitGroups;
    std::vector<optixu::CallableProgramGroup> callablePrograms;
    
    // Shader binding tables
    cudau::Buffer shaderBindingTable;
    cudau::Buffer hitGroupSBT;
    
    // Configuration and state
    PipelineConfig config;
    EntryPointType currentEntryPoint;
    PipelineState state = PipelineState::Uninitialized;
    
    // Ray type information
    uint32_t numRayTypes = 1;
    uint32_t searchRayIndex = 0;
    uint32_t visibilityRayIndex = 1;
    
    // Set the active entry point
    void setEntryPoint(EntryPointType ep) {
        auto it = entryPoints.find(ep);
        if (it != entryPoints.end() && optixPipeline) {
            optixPipeline.setRayGenerationProgram(it->second);
            currentEntryPoint = ep;
        }
    }
    
    // Check if ready to launch
    bool isReady() const {
        return state == PipelineState::Ready && optixPipeline;
    }
    
    // Check if initialized (for compatibility)
    bool isInitialized() const {
        return state >= PipelineState::Linked;
    }
    
    // State transition helpers
    void transitionTo(PipelineState newState) {
        // Only allow forward transitions (except reset to Uninitialized)
        if (newState == PipelineState::Uninitialized || newState > state) {
            state = newState;
        }
    }
    
    // Clean up resources
    void destroy() {
        // Clean up in reverse order of creation
        if (hitGroupSBT.isInitialized()) {
            hitGroupSBT.finalize();
        }
        if (shaderBindingTable.isInitialized()) {
            shaderBindingTable.finalize();
        }
        
        for (auto& cp : callablePrograms) {
            if (cp) cp.destroy();
        }
        callablePrograms.clear();
        
        for (auto& [name, hg] : hitGroups) {
            if (hg) hg.destroy();
        }
        hitGroups.clear();
        
        for (auto& [type, prog] : programs) {
            if (prog) prog.destroy();
        }
        programs.clear();
        
        for (auto& [ep, prog] : entryPoints) {
            if (prog) prog.destroy();
        }
        entryPoints.clear();
        
        for (auto& module : additionalModules) {
            if (module) module.destroy();
        }
        additionalModules.clear();
        
        if (optixModule) {
            optixModule.destroy();
        }
        
        if (optixPipeline) {
            optixPipeline.destroy();
        }
        
        state = PipelineState::Uninitialized;
    }
};

// Main PipelineHandler class
class PipelineHandler {
public:
    // Factory method following standard handler pattern
    static PipelineHandlerPtr create(RenderContextPtr ctx) {
        return std::make_shared<PipelineHandler>(ctx);
    }
    
    // Constructor/Destructor
    explicit PipelineHandler(RenderContextPtr ctx);
    ~PipelineHandler();
    
    // Pipeline creation and setup (production-style API)
    void setupPipeline(const PipelineData& data, const std::string& kernelName);
    void setupPipeline(const PipelineData& data, const std::filesystem::path& ptxFile);
    
    // SBT management
    void setupSBT(EntryPointType type);
    void updateSBT(EntryPointType type);
    void setSceneDependentSBT(EntryPointType type);
    
    // Pipeline access
    Pipeline::Ptr getPipeline(EntryPointType type);
    bool hasPipeline(EntryPointType type) const;
    
    // Clean launch API (ShockerEngine-style)
    void launch(EntryPointType type, CUstream stream, CUdeviceptr plp, 
                uint32_t width, uint32_t height, uint32_t depth = 1);
    
    // Entry point management
    template <typename SpecificEntryPointType>
    void setEntryPoint(EntryPointType pipelineType, SpecificEntryPointType entryPoint) {
        // TODO: Type-safe entry point setting
    }
    
    // Scene management
    void setScene(optixu::Scene scene);
    void updateSceneSBT();  // Call this when scene geometry changes
    
    // Material and hit group management
    void setMaterialHitGroups(EntryPointType type, optixu::Material material);
    optixu::HitProgramGroup getHitGroup(EntryPointType type, const std::string& name);
    void createEmptyHitGroup(EntryPointType type, const std::string& name);
    
    // Configure hit groups for materials on geometry instances
    void configureMaterialHitGroups(optixu::GeometryInstance* geomInst,
                                    const std::map<EntryPointType, std::vector<std::pair<uint32_t, std::string>>>& hitGroupConfig);
    
    // Stack size management
    void calculateStackSizes(EntryPointType type);
    void setStackSize(EntryPointType type, uint32_t directCallable, 
                      uint32_t directCallableFromState, uint32_t continuation);
    
    // Module management - removed as modules are now loaded per-pipeline internally
    
    // Callable program support
    void setupCallablePrograms(EntryPointType type, 
                               const std::vector<std::string>& entryPoints);
    
    // State queries
    bool isReady(EntryPointType type) const;
    bool isAnyPipelineReady() const;
    std::vector<EntryPointType> getActivePipelines() const;
    
    // Multi-pipeline coordination
    void renderSequence(const std::vector<EntryPointType>& sequence,
                       CUstream stream, CUdeviceptr plp,
                       uint32_t width, uint32_t height);
    
    // Resource management
    void destroyPipeline(EntryPointType type);
    void destroyAll();
    
private:
    // Private implementation
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Internal methods
    Pipeline::Ptr createPipeline(EntryPointType type);
    void linkPipeline(EntryPointType type, uint32_t maxTraceDepth);
    void setupRayTypes(EntryPointType type, uint32_t numRayTypes);
    void generateSBTLayout(EntryPointType type);
    void setMinimalHitGroupSBT(EntryPointType type);
    void validatePipelineConfig(const PipelineConfig& config);
    
    // Refactored pipeline setup steps
    void createOrGetPipeline(Pipeline::Ptr& pipeline, EntryPointType entryPoint);
    void configurePipelineOptions(Pipeline::Ptr& pipeline, const PipelineConfig& config);
    void loadPipelineModule(Pipeline::Ptr& pipeline, const std::string& kernelName, EntryPointType entryPoint);
    void createPipelinePrograms(Pipeline::Ptr& pipeline, const PipelineData& data);
    void setupPipelineRayTypes(Pipeline::Ptr& pipeline, const PipelineData& data);
    void linkAndFinalizePipeline(Pipeline::Ptr& pipeline, const PipelineData& data);
    
    // Module loading helpers
    std::vector<char> loadPTXFromFile(const std::filesystem::path& path);
    std::vector<char> loadOptixIRFromFile(const std::filesystem::path& path);
};

// Inline implementations for templates
template <>
inline void PipelineHandler::setEntryPoint<EntryPointType>(
    EntryPointType pipelineType, EntryPointType entryPoint) {
    if (auto pipeline = getPipeline(pipelineType)) {
        pipeline->setEntryPoint(entryPoint);
    }
}