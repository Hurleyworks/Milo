

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

// handlers
#include "excludeFromBuild/handlers/SkyDomeHandler.cpp"
#include "excludeFromBuild/handlers/DenoiserHandler.cpp"
#include "excludeFromBuild/handlers/MiloSceneHandler.cpp"
#include "excludeFromBuild/handlers/MiloMaterialHandler.cpp"
#include "excludeFromBuild/handlers/MiloModelHandler.cpp"
#include "excludeFromBuild/handlers/MiloRenderHandler.cpp"
#include "excludeFromBuild/handlers/TextureHandler.cpp"
#include "excludeFromBuild/handlers/MiloDenoiserHandler.cpp"


#if 0
#include "excludeFromBuild/handlers/RenderHandler.cpp"
#include "excludeFromBuild/handlers/ShockerRenderHandler.cpp"
#include "excludeFromBuild/handlers/EngineSceneHandler.cpp"

#include "excludeFromBuild/handlers/ShockerSceneHandler.cpp"
#include "excludeFromBuild/handlers/ShockerMaterialHandler.cpp"
#include "excludeFromBuild/handlers/ShockerGeometryHandler.cpp"
#include "excludeFromBuild/handlers/RiPRSceneHandler.cpp"
#include "excludeFromBuild/handlers/RiPRMaterialHandler.cpp"
#include "excludeFromBuild/handlers/RiPRModelHandler.cpp"
#include "excludeFromBuild/handlers/RiPRRenderHandler.cpp"
#endif


// engines
#include "excludeFromBuild/engines/BaseRenderingEngine.cpp"
#include "excludeFromBuild/engines/RenderEngineManager.cpp"
#include "excludeFromBuild/engines/MiloEngine.cpp"

//#include "excludeFromBuild/engines/BasicPathTracingEngine.cpp"
//#include "excludeFromBuild/engines/EnvironmentRenderEngine.cpp"
//#include "excludeFromBuild/engines/ShockerRenderEngine.cpp"
//#include "excludeFromBuild/engines/RiPREngine.cpp"
//#include "excludeFromBuild/engines/TestEngine.cpp"



// models
//#include "excludeFromBuild/model/RiPRModel.cpp"
#include "excludeFromBuild/model/MiloModel.cpp"

//#include "excludeFromBuild/geometry/EngineTriangleMesh.cpp"




