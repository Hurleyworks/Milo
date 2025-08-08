#pragma once

// RenderPipeline.h
// Template pipeline structure for managing OptiX pipelines in rendering engines.
// Provides type-safe entry point management and standardized pipeline configuration.

#include <unordered_map>
#include <vector>
#include <string>

// Forward declarations
namespace optixu {
    class Pipeline;
    class Module;
    class Program;
    class HitProgramGroup;
    class CallableProgramGroup;
}

namespace cudau {
    class Buffer;
}

class RenderContext;

namespace engine_core {

// Pipeline configuration constants
struct PipelineConfig {
    uint32_t maxTraceDepth = 2;
    uint32_t numPayloadDwords = 3;      // Typical for RGB payload
    uint32_t numAttributeDwords = 2;    // Typically calculated using optixu::calcSumDwords<float2>()
    uint32_t stackSize = 0;             // Will be calculated
    
    // OptiX pipeline compile options
    uint32_t maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    OptixCompileOptimizationLevel optimizationLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    OptixCompileDebugLevel debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
};

// Template pipeline class for managing OptiX pipelines
template <typename EntryPointType>
class RenderPipeline {
public:
    RenderPipeline() :
        currentEntryPoint(static_cast<EntryPointType>(-1)) {
    }
    
    ~RenderPipeline() {
        // Cleanup is handled in destroy() method
    }
    
    // Initialize the pipeline with given configuration
    void initialize(RenderContext* context, const PipelineConfig& config) {
        this->config = config;
        // Actual initialization will happen in derived engine
    }
    
    // Set the active entry point
    void setEntryPoint(EntryPointType entryPoint) {
        auto it = entryPoints.find(entryPoint);
        if (it != entryPoints.end() && optixPipeline) {
            optixPipeline.setRayGenerationProgram(it->second);
            currentEntryPoint = entryPoint;
        }
    }
    
    // Get current entry point
    EntryPointType getCurrentEntryPoint() const {
        return currentEntryPoint;
    }
    
    // Check if entry point exists
    bool hasEntryPoint(EntryPointType entryPoint) const {
        return entryPoints.find(entryPoint) != entryPoints.end();
    }
    
    // Clean up resources
    void destroy() {
        // Clean up in reverse order of creation
        for (auto& callable : callablePrograms) {
            callable.destroy();
        }
        callablePrograms.clear();
        
        for (auto& [name, program] : hitPrograms) {
            program.destroy();
        }
        hitPrograms.clear();
        
        for (auto& [name, program] : programs) {
            program.destroy();
        }
        programs.clear();
        
        for (auto& [type, program] : entryPoints) {
            program.destroy();
        }
        entryPoints.clear();
        
        // Module and pipeline cleanup
        if (optixModule) {
            optixModule.destroy();
        }
        if (optixPipeline) {
            optixPipeline.destroy();
        }
        
        // Buffer cleanup
        if (sbt.isInitialized()) {
            sbt.finalize();
        }
        if (hitGroupSbt.isInitialized()) {
            hitGroupSbt.finalize();
        }
    }
    
public:
    // OptiX components (stored as values, not pointers)
    optixu::Pipeline optixPipeline;
    optixu::Module optixModule;
    
    // Program collections
    std::unordered_map<EntryPointType, optixu::Program> entryPoints;
    std::unordered_map<std::string, optixu::Program> programs;
    std::unordered_map<std::string, optixu::HitProgramGroup> hitPrograms;
    std::vector<optixu::CallableProgramGroup> callablePrograms;
    
    // Shader binding tables
    cudau::Buffer sbt;
    cudau::Buffer hitGroupSbt;
    
    // Launch parameters buffer
    cudau::Buffer launchParamsBuffer;
    
    // Configuration
    PipelineConfig config;
    
private:
    EntryPointType currentEntryPoint;
};

} // namespace dog_core