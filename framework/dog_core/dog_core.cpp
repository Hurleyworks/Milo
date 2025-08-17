// Stub implementation for dog_core framework
// Starting from scratch - add includes as implementations are added

#include "dog_core.h"

// Include the implementations
#include "excludeFromBuild/common/common_host.cpp"
#include "excludeFromBuild/common/dds_loader.cpp"
#include "excludeFromBuild/ActiveRender.cpp"
#include "excludeFromBuild/Renderer.cpp"
#include "excludeFromBuild/GPUContext.cpp"
#include "excludeFromBuild/RenderContext.cpp"

// Handler implementations
#include "excludeFromBuild/handlers/ScreenBufferHandler.cpp"
#include "excludeFromBuild/handlers/PipelineHandler.cpp"

// Tools implementations
#include "excludeFromBuild/tools/PTXManager.cpp"
#include "excludeFromBuild/tools/GPUManager.cpp"
#include "excludeFromBuild/tools/GPUMemoryMonitor.cpp"
#include "excludeFromBuild/tools/GPUTimerManager.cpp"

// Additional includes will be added here as the framework is built out
// For example:
// #include "excludeFromBuild/handlers/Handlers.cpp"
// etc.