#pragma once


#include "../../../engine_core.h"
// EngineEntryPoints.h
// Defines common entry point enums for all rendering engines.
// Each engine can extend the base enum with its specific entry points.

namespace engine_core {

// Base entry points common to all engines
enum class CommonEntryPoint {
    // Reserved common entry points that might be shared
    Invalid = -1,
    
    // Engine-specific entry points start from here
    EngineSpecificStart = 0
};

// Environment rendering engine entry points
enum class EnvironmentEntryPoint {
    RenderEnvironment = static_cast<int>(CommonEntryPoint::EngineSpecificStart),
    NumEntryPoints
};

// Basic path tracing engine entry points
enum class PathTracingEntryPoint {
    PathTrace = static_cast<int>(CommonEntryPoint::EngineSpecificStart),
    PathTraceProgressive,
    NumEntryPoints
};

// Test engine entry points
enum class TestEngineEntryPoint {
    SimpleRender = static_cast<int>(CommonEntryPoint::EngineSpecificStart),
    NumEntryPoints
};

// Picking engine entry points
enum class PickingEntryPoint {
    PickAndGBuffer = static_cast<int>(CommonEntryPoint::EngineSpecificStart),
    NumEntryPoints
};

// GBuffer engine entry points
enum class GBufferEntryPoint {
    setupGBuffers = static_cast<int>(CommonEntryPoint::EngineSpecificStart),
    NumEntryPoints
};


} // namespace engine_core