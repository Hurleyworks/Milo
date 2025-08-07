

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
#include "excludeFromBuild/handlers/MiloSceneHandler.cpp"
#include "excludeFromBuild/handlers/MiloMaterialHandler.cpp"
#include "excludeFromBuild/handlers/MiloModelHandler.cpp"
#include "excludeFromBuild/handlers/MiloRenderHandler.cpp"
#include "excludeFromBuild/handlers/MiloDenoiserHandler.cpp"

// shocker handlers
#include "excludeFromBuild/handlers/ShockerModelHandler.cpp"
#include "excludeFromBuild/handlers/ShockerMaterialHandler.cpp"
#include "excludeFromBuild/handlers/ShockerSceneHandler.cpp"

// engines
#include "excludeFromBuild/engines/BaseRenderingEngine.cpp"
#include "excludeFromBuild/engines/RenderEngineManager.cpp"
#include "excludeFromBuild/engines/MiloEngine.cpp"

// models
#include "excludeFromBuild/model/MiloModel.cpp"
#include "excludeFromBuild/model/ShockerModel.cpp"
