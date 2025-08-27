#pragma once

// EngineRegistry.h
// Helper functions to register all available rendering engines

#include "../RenderEngineManager.h"
#include "../shocker/ShockerEngine.h"
#include "../ripr/RiPREngine.h"
#include "../claudia/ClaudiaEngine.h"

    // Register all built-in rendering engines
inline void registerBuiltInEngines(RenderEngineManager& manager)
{

    // Register Shocker engine
    manager.registerEngine("shocker",
                          "Shocker Engine",
                          "Next-generation path tracing engine with optimized performance",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<ShockerEngine>();
                          });
    
    // Register RiPR engine
    manager.registerEngine("ripr",
                          "RiPR Engine",
                          "RiPR path tracing engine with area light support and improved handlers",
                          []() -> std::unique_ptr<IRenderingEngine> {
                              return std::make_unique<RiPREngine>();
                          });

    // Register Claudia engine
    manager.registerEngine ("claudia",
                            "Claudia Engine",
                            "Claudia path tracing engine with area light support and improved handlers",
                            []() -> std::unique_ptr<IRenderingEngine>
                            {
                                return std::make_unique<ClaudiaEngine>();
                            });
    
    // TODO: Register more engines as they are implemented
    // manager.registerEngine("svgf", []() {
    //     return std::make_unique<SVGFEngine>();
    // });
}
