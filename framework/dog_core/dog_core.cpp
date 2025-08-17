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

// Additional includes will be added here as the framework is built out
// For example:
// #include "excludeFromBuild/handlers/Handlers.cpp"
// #include "excludeFromBuild/tools/PTXManager.cpp"
// etc.