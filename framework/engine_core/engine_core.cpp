

#include "engine_core.h"

#include "excludeFromBuild/ActiveRender.cpp"
#include "excludeFromBuild/Renderer.cpp"
#include "excludeFromBuild/GPUContext.cpp"
#include "excludeFromBuild/RenderContext.cpp"

// tools
#include "excludeFromBuild/tools/PTXManager.cpp"
#include "excludeFromBuild/tools/GPUManager.cpp"
#include "excludeFromBuild/tools/GPUMemoryMonitor.cpp"
#include "excludeFromBuild/tools/GPUTimerManager.cpp"

// common
#include "excludeFromBuild/common/common_host.cpp"
#include "excludeFromBuild/common/dds_loader.cpp"

// nvcc
#include "excludeFromBuild/nvcc/CudaCompiler.cpp"

// commmon handlers
#include "excludeFromBuild/handlers/SkyDomeHandler.cpp"
#include "excludeFromBuild/handlers/TextureHandler.cpp"
#include "excludeFromBuild/handlers/AreaLightHandler.cpp"

// milo handlers
#include "excludeFromBuild/engines/milo/handlers/MiloSceneHandler.cpp"
#include "excludeFromBuild/engines/milo/handlers/MiloMaterialHandler.cpp"
#include "excludeFromBuild/engines/milo/handlers/MiloModelHandler.cpp"
#include "excludeFromBuild/engines/milo/handlers/MiloRenderHandler.cpp"
#include "excludeFromBuild/engines/milo/handlers/MiloDenoiserHandler.cpp"

// claudia handlers
#include "excludeFromBuild/engines/claudia/handlers/ClaudiaSceneHandler.cpp"
#include "excludeFromBuild/engines/claudia/handlers/ClaudiaMaterialHandler.cpp"
#include "excludeFromBuild/engines/claudia/handlers/ClaudiaModelHandler.cpp"
#include "excludeFromBuild/engines/claudia/handlers/ClaudiaRenderHandler.cpp"
#include "excludeFromBuild/engines/claudia/handlers/ClaudiaDenoiserHandler.cpp"

// ripr handlers
#include "excludeFromBuild/engines/ripr/handlers/RiPRModelHandler.cpp"
#include "excludeFromBuild/engines/ripr/handlers/RiPRMaterialHandler.cpp"
#include "excludeFromBuild/engines/ripr/handlers/RiPRSceneHandler.cpp"
#include "excludeFromBuild/engines/ripr/handlers/RiPRRenderHandler.cpp"
#include "excludeFromBuild/engines/ripr/handlers/RiPRDenoiserHandler.cpp"

// shocker handlers
#include "excludeFromBuild/engines/shocker/handlers/ShockerModelHandler.cpp"
#include "excludeFromBuild/engines/shocker/handlers/ShockerMaterialHandler.cpp"
#include "excludeFromBuild/engines/shocker/handlers/ShockerSceneHandler.cpp"
#include "excludeFromBuild/engines/shocker/handlers/ShockerRenderHandler.cpp"

// engines
#include "excludeFromBuild/engines/base/BaseRenderingEngine.cpp"
#include "excludeFromBuild/engines/RenderEngineManager.cpp"
#include "excludeFromBuild/engines/milo/MiloEngine.cpp"
#include "excludeFromBuild/engines/claudia/ClaudiaEngine.cpp"
#include "excludeFromBuild/engines/ripr/RiPREngine.cpp"
#include "excludeFromBuild/engines/shocker/ShockerEngine.cpp"

// models
#include "excludeFromBuild/engines/milo/models/MiloModel.cpp"
#include "excludeFromBuild/engines/claudia/models/ClaudiaModel.cpp"
#include "excludeFromBuild/engines/ripr/models/RiPRModel.cpp"
#include "excludeFromBuild/engines/shocker/models/ShockerModel.cpp"
