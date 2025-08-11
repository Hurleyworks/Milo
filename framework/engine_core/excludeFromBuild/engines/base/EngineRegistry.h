#pragma once

// EngineRegistry.h
// Helper functions to register all available rendering engines

#include "../RenderEngineManager.h"
//#include "TestEngine.h"
//#include "BasicPathTracingEngine.h"
//#include "EnvironmentRenderEngine.h"
//#include "ShockerRenderEngine.h"
//#include "RiPREngine.h"
#include "../milo/MiloEngine.h"
#include "../shocker/ShockerEngine.h"
#include "../ripr/RiPREngine.h"

// Register all built-in rendering engines
inline void registerBuiltInEngines(RenderEngineManager& manager)
{
    #if 0
    // Register test engine
    manager.registerEngine("test", 
                          "Test Engine",
                          "Simple test engine for debugging and development",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<TestEngine>();
                          });
    
    // Register basic path tracer
    manager.registerEngine("basic_pathtracer",
                          "Basic Path Tracer", 
                          "Monte Carlo path tracing with HDR environment mapping",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<BasicPathTracingEngine>();
                          });
    
    // Register environment renderer
    manager.registerEngine("environment_renderer",
                          "Environment Renderer",
                          "Specialized engine for HDR environment and sky rendering",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<EnvironmentRenderEngine>();
                          });
    
    
    // Register Shocker render engine
    manager.registerEngine("shocker",
                          "Shocker Render Engine",
                          "Shocker-based path tracing with optimized scene management",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<ShockerRenderEngine>();
                          });
    
    // Register RiPR engine
    manager.registerEngine("ripr",
                          "RiPR Engine",
                          "ReSTIR Path Tracing with adaptive sampling and improved convergence",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<RiPREngine>();
                          });
    #endif
    // Register Milo engine
    manager.registerEngine("milo",
                          "Milo Engine",
                          "High-performance path tracing engine based on RiPR architecture",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<MiloEngine>();
                          });
    
    // Register Shocker engine
    manager.registerEngine("shocker",
                          "Shocker Engine",
                          "Dual-pipeline ray tracing engine with G-buffer and path tracing modes",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<ShockerEngine>();
                          });
    
    // Register RiPR engine
    manager.registerEngine("ripr",
                          "RiPR Engine",
                          "Dual-pipeline ray tracing engine with G-buffer and path tracing modes",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<RiPREngine>();
                          });
    
    // TODO: Register more engines as they are implemented
    // manager.registerEngine("svgf", []() {
    //     return std::make_unique<SVGFEngine>();
    // });
}
