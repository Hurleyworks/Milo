

#include "claude_core.h"

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
#include "excludeFromBuild/handlers/MaterialHandler.cpp"
#include "excludeFromBuild/handlers/PipelineHandler.cpp"
#include "excludeFromBuild/handlers/DenoiserHandler.cpp"
#include "excludeFromBuild/handlers/ScreenBufferHandler.cpp"

// shocker handlers
#include "excludeFromBuild/engines/shocker/handlers/ShockerModelHandler.cpp"
#include "excludeFromBuild/engines/shocker/handlers/ShockerMaterialHandler.cpp"
#include "excludeFromBuild/engines/shocker/handlers/ShockerSceneHandler.cpp"


//// ripr handlers
//#include "excludeFromBuild/engines/ripr/handlers/RiPRModelHandler.cpp"
//#include "excludeFromBuild/engines/ripr/handlers/RiPRMaterialHandler.cpp"
//#include "excludeFromBuild/engines/ripr/handlers/RiPRSceneHandler.cpp"
//#include "excludeFromBuild/engines/ripr/handlers/RiPRRenderHandler.cpp"
//#include "excludeFromBuild/engines/ripr/handlers/RiPRDenoiserHandler.cpp"
//#include "excludeFromBuild/engines/ripr/handlers/ModelHandler.cpp"

// engines
#include "excludeFromBuild/engines/base/BaseRenderingEngine.cpp"
#include "excludeFromBuild/engines/RenderEngineManager.cpp"
#include "excludeFromBuild/engines/shocker/ShockerEngine.cpp"
//#include "excludeFromBuild/engines/ripr/RiPREngine.cpp"

// models

#include "excludeFromBuild/engines/shocker/models/ShockerModel.cpp"
//#include "excludeFromBuild/engines/ripr/models/RiPRModel.cpp"
