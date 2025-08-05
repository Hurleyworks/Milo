# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Milo is an interactive physics-based content creation framework built primarily in C++ with CUDA/OptiX rendering capabilities. It uses a custom application framework called "Jahley" and includes sophisticated GPU-accelerated path tracing.

## Build Commands

### Generate Visual Studio Solution
```batch
generateVS2022.bat
```

### Build the Project
```batch
b
```
This runs the automated build script that captures errors and formats them for Claude. If the build fails, errors will be saved to `build_errors.txt`.

### Continuous Build Loop
For iterative development with Claude fixing errors:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/claude_build_command.ps1 -Action loop
```

### Manual Build
```batch
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" builds\VisualStudio2022\Apps.sln /p:Configuration=Debug /p:Platform=x64
```

### Code Formatting
The project uses clang-format. Format files with:
```batch
clang-format -i <filename>
```

## Architecture Overview

### Core Framework Structure

1. **appCore/** - Jahley Framework
   - Base application framework providing:
     - Application lifecycle management (`App.h`)
     - Logging system (`Log.h`)
     - OpenGL rendering (`OpenglRenderer.h`)
     - Entry point abstraction (`EntryPoint.h`)
   - Uses precompiled headers (`berserkpch.h`)

2. **framework/** - Core Systems
   - **engine_core**: CUDA/OptiX rendering engine
     - Material, model, scene, and texture handlers
     - PTX management for GPU kernels
     - Denoiser integration
   - **properties_core**: Property system for configuration
   - **qms_core**: QuickSilver Messenger Service for event handling
   - **server_core**: Socket server for remote communication

3. **modules/** - Extended Functionality
   - **mace_core**: Basic utilities and input handling
   - **oiio_core**: OpenImageIO integration
   - **sabi_core**: Scene graph, animation, I/O (GLTF, LWO3)
   - **wabi_core**: Math utilities and mesh processing

### Key Design Patterns

1. **Message-Driven Architecture**: Uses QMS (QuickSilver Messenger Service) for decoupled communication between components.

2. **Handler Pattern**: Major subsystems use handlers (e.g., MaterialHandler, ModelHandler, SceneHandler) for managing resources.

3. **PTX Embedding**: CUDA kernels are compiled to PTX and embedded as C arrays for runtime compilation.

4. **excludeFromBuild Pattern**: Work-in-progress or optional code is organized in `excludeFromBuild/` directories.

### Application Entry Point

Applications inherit from `Jahley::App` and use the `JahleyEntryPoint` macro. Example:
```cpp
class MyApp : public Jahley::App {
    // Implementation
};
JahleyEntryPoint(MyApp);
```

### Resource Paths

The framework automatically creates these directories:
- cache/
- logs/
- output/
- resources/
- scripts/
- temp/

### GPU/CUDA Development

- CUDA 12.9 and OptiX 9.0 are required
- PTX files are embedded using scripts in `scripts/` directory
- GPU kernels support multiple architectures (sm_50 through sm_90a)

### Dependencies

Major dependencies managed through vcpkg and direct inclusion:
- Rendering: GLFW, OpenGL, CUDA, OptiX
- UI: NanoGUI
- Math: Eigen
- Image I/O: OpenImageIO, STB
- Serialization: Cereal, nlohmann/json
- Physics: Newton Dynamics
- Logging: g3log

### Sample Applications

1. **HelloWorld**: Minimal application demonstrating framework usage
2. **RenderDog**: Advanced rendering application with MVC architecture

## Important Notes

- The project is Windows-only with Visual Studio 2022 support
- x64 architecture only
- Uses C++20 standard for VS2022 builds
- No automated testing framework is currently set up
- The `b.bat` command is the primary build tool and includes error formatting for Claude

## Module System

### Module Dependency Chain
```
mace_core (base utilities, input handling)
    ↓
wabi_core (math) and oiio_core (image I/O) - parallel
    ↓
sabi_core (scene management, file I/O)
    ↓
properties_core (configuration system)
    ↓
qms_core (messaging system)
    ↓
engine_core (CUDA/OptiX rendering)
```

### Key Implications
- **No standard headers needed**: mace_core includes all STL headers
- **Upstream access**: Each module has access to all upstream module types
- **Amalgamated pattern**: Each module's main .cpp includes all implementation files from `excludeFromBuild/`


## Critical Reminders

### Never Include These Headers
- `#include "OptixUtil/optixu_on_cudau.h"`
- `#include "CUDAUtil/cudau.h"`
These headers are not part of this codebase and will cause compilation errors.

### Build Workflow
- **Build automatically after making changes** - by running `b.bat`
- **Build errors**: Check `E:\1Dog\Dog\build_errors.txt`
- **Fix errors**: Read the error file and fix issues when requested

## PTX File Management

- **PTX Generation**: CUDA kernels are compiled to PTX files during a separate build by the user
- **PTX Locations**: `resources/RenderDog/ptx/[Debug|Release]/sm_86/`
- **Python Requirement**: Python 3.11+ required for PTX embedding

## Development Principles

- **Always Study APIs Thoroughly**
  - Never make assumptions about an API. Study the API first so you get it right the first time!!!
  
  Please keep the following guidelines in mind throughout our interaction:


# Development Guidelines

## 1. Communication Style
- Use clear and simple language in your explanations.
- If you lack information or knowledge, ask for clarification instead of guessing.
- Balance comprehensive explanations with concise code examples.

## 2. Code Standards
- Write well-commented, simple, elegant, and production-ready code.
- Avoid overly complicated constructs.
- Only include headers that are part of the project.
- Always use `#pragma once` instead of include guards.
- Use `.h` extension for headers, never `.hpp`.

## 3. Documentation Standards

### a. Code Comments
- Use only single-line comments with `//` style.
- Place comments above each function/method, explaining its purpose.
- Keep comments concise but informative.
- Don't comment obvious member variables or add decorative comments/separators.
- Maintain minimal spacing between functions.

### b. Class Documentation
- For each class, provide a detailed summary in `//` comments at the top of the header file.

## 4. Specific Requirements
- Implement appropriate safety and error checks.

## 5. Testing Requirements
- Use C++ doctest for unit testing.
- Be prepared to demonstrate doctest usage within the {{SDK_NAME}} SDK context.

## 6. Error Handling and Logging
- Use g3log for logging.
- Use DBUG level for debug information (not DEBUG).
- Use WARNING level for non-fatal issues (instead of ERROR or CRITICAL).

## 7. Development Process
- Thoroughly verify your knowledge before responding.
- Ensure all code is suitable for production environments.
- Consider common edge cases and provide appropriate error handling.