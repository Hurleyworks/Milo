

#include "claude_core.h"

#include "excludeFromBuild/ActiveRender.cpp"
#include "excludeFromBuild/Renderer.cpp"
#include "excludeFromBuild/GPUContext.cpp"

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
#include "excludeFromBuild/handlers/DisneyMaterialHandler.cpp"
#include "excludeFromBuild/handlers/PipelineHandler.cpp"
#include "excludeFromBuild/handlers/DenoiserHandler.cpp"
#include "excludeFromBuild/handlers/ScreenBufferHandler.cpp"
#include "excludeFromBuild/handlers/InstanceHandler.cpp"
#include "excludeFromBuild/handlers/TriangleMeshHandler.cpp"

// shocker handlers
#include "excludeFromBuild/engines/shocker/handlers/ShockerModelHandler.cpp"
#include "excludeFromBuild/engines/shocker/handlers/ShockerSceneHandler.cpp"

// ripr handlers  
#include "excludeFromBuild/engines/ripr/handlers/RiPRModelHandler.cpp"
#include "excludeFromBuild/engines/ripr/handlers/RiPRSceneHandler.cpp"

// claudia handlers  
#include "excludeFromBuild/engines/claudia/handlers/ClaudiaModelHandler.cpp"
#include "excludeFromBuild/engines/claudia/handlers/ClaudiaSceneHandler.cpp"

// engines
#include "excludeFromBuild/engines/base/BaseRenderingEngine.cpp"
#include "excludeFromBuild/engines/RenderEngineManager.cpp"
#include "excludeFromBuild/engines/shocker/ShockerEngine.cpp"
#include "excludeFromBuild/engines/ripr/RiPREngine.cpp"
#include "excludeFromBuild/engines/claudia/ClaudiaEngine.cpp"

// models
#include "excludeFromBuild/engines/shocker/models/ShockerModel.cpp"
#include "excludeFromBuild/engines/ripr/models/RiPRModel.cpp"
#include "excludeFromBuild/engines/claudia/models/ClaudiaModel.cpp"
