# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Milo is an interactive physics-based content creation framework built primarily in C++ with CUDA/OptiX rendering capabilities. It uses a custom application framework called "Jahley" and includes sophisticated GPU-accelerated path tracing with multiple rendering engines (Milo, Shocker, RiPR, Claudia).

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

### Unit Testing

#### Compile and run all unit tests (PREFERRED METHOD)
```batch
cd unittest
test
```
This command will:
- Build all unit tests
- Run all test executables
- Save results to `unittest/test_results.txt` if tests fail
- Save build errors to `unittest/build_errors.txt` if compilation fails

#### Compile and run specific test
```batch
cd unittest
test CgModel2Shocker
```

#### Build only (without running)
```batch
cd unittest
b
```

#### Run specific test scripts
```batch
scripts\test_shocker_model.bat
scripts\test_area_light_handler.bat
```

#### Manual test compilation and execution
```batch
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" unittest\builds\VisualStudio2022\projects\[TestName].vcxproj /p:Configuration=Debug /p:Platform=x64
builds\bin\Debug-windows-x86_64\[TestName]\[TestName].exe
```

#### Available tests
- HelloTest
- GeometryTest  
- ShockerModelTest
- AreaLightHandlerTest
- Eigen2Shocker

#### Creating a new test script
To create a test runner script for a new test, follow the pattern in `scripts/test_*.bat`

### Code Formatting
The project uses clang-format with custom settings. Format files with:
```batch
clang-format -i <filename>
```
Key formatting rules:
- 4 spaces indentation (no tabs)
- Allman brace style (braces on new line)
- No column limit
- Pointer alignment to left (`char* ptr` not `char *ptr`)
- Don't sort includes (maintains specific order)

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
   - **engine_core**: CUDA/OptiX rendering engines (Milo, Shocker, RiPR, Claudia)
     - Base engine class: `BaseRenderingEngine`
     - Common handlers: Material, Model, Scene, Render, Denoiser, Texture, SkyDome, AreaLight
     - PTX management for GPU kernels
     - GPU context and timer management
   - **dog_core**: New lightweight rendering framework (under development)
     - Simplified pipeline architecture
     - ScreenBuffer, Pipeline, PipelineParameter, and Scene handlers
     - Message-based integration with QMS system
   - **properties_core**: Property system for configuration
   - **qms_core**: QuickSilver Messenger Service for event handling
   - **server_core**: Socket server for remote communication

3. **modules/** - Extended Functionality
   - **mace_core**: Basic utilities and input handling (includes all STL headers)
   - **oiio_core**: OpenImageIO integration for image I/O
   - **sabi_core**: Scene graph, animation, I/O (GLTF, LWO3)
   - **wabi_core**: Math utilities and mesh processing

### Key Design Patterns

1. **Message-Driven Architecture**: Uses QMS (QuickSilver Messenger Service) for decoupled communication between components.

2. **Handler Pattern**: Each rendering engine uses specialized handlers:
   - `SceneHandler`: Manages scene graph and traversal
   - `MaterialHandler`: Material creation and management
   - `ModelHandler`: Geometry and model management
   - `RenderHandler`: Rendering pipeline and buffer management
   - `DenoiserHandler`: OptiX AI denoiser integration

3. **PTX Embedding**: CUDA kernels are compiled to PTX and embedded as C arrays for runtime compilation.

4. **excludeFromBuild Pattern**: Implementation files are organized in `excludeFromBuild/` directories and included by the module's main .cpp file via amalgamation.

### Application Entry Point

Applications inherit from `Jahley::App` and define `Jahley::CreateApplication()`. The framework provides the main() function in `EntryPoint.h`. Example:
```cpp
class MyApp : public Jahley::App {
    // Implementation
};

Jahley::App* Jahley::CreateApplication() {
    return new MyApp();
}
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
- PTX files are located in `resources/RenderDog/ptx/[Debug|Release]/sm_86/`

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
- Unit tests use doctest framework in `unittest/` directory
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
engine_core / dog_core (CUDA/OptiX rendering)
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
- **Build errors**: Saved to `build_errors.txt` in project root when build fails
- **Fix errors**: Read the error file and fix issues when requested
- **Current branch context**: The current branch is often a feature branch - check git status for context

## PTX File Management

- **PTX Generation**: CUDA kernels are compiled to PTX files during a separate build by the user
- **PTX Embedding Scripts**: `scripts/desktop_dog_embed_ptx_debug.bat` and `scripts/desktop_dog_embed_ptx_release.bat`
- **Python Requirement**: Python 3.11+ required for PTX embedding (`scripts/embed_ptx.py`)
- **Embedded PTX Location**: `framework/engine_core/generated/embedded_ptx.h`

## Development Principles

### Code Standards
- Use `.h` extension for headers, never `.hpp`
- Always use `#pragma once` instead of include guards
- Use single-line comments with `//` style
- Don't add decorative comments or separators
- Maintain minimal spacing between functions

### Testing
- Tests are located in `unittest/tests/` directory
- Each test project follows the pattern: `[Component]Test`
- Use doctest framework for unit testing
- Tests are built separately from main project using `unittest/generateVS2022.bat`
- Test solution: `unittest/builds/VisualStudio2022/UnitTests.sln`

### Logging
- Use g3log for logging
- Use DBUG level for debug information (not DEBUG)
- Use WARNING level for non-fatal issues

## Rendering Engines

The project includes multiple rendering engines in `framework/engine_core/excludeFromBuild/engines/`:

### Engine Architecture
All engines inherit from `BaseRenderingEngine` and implement:
- **Milo**: Primary rendering engine
- **Shocker**: Alternative rendering implementation
- **RiPR**: Another rendering variant  
- **Claudia**: Additional rendering engine

### Engine Components
Each engine has its own:
- Handler classes (Material, Model, Scene, Render, Denoiser)
- CUDA kernels (.cu files) and PTX files
- Shared header for GPU/host communication (`[engine]_shared.h`)
- Models directory with specialized model implementations

### Common Handler Interfaces
- `TextureHandler`: Texture loading and management
- `SkyDomeHandler`: Environment lighting
- `AreaLightHandler`: Area light management
- `Handlers`: Central handler registry

## Important Implementation Details

- **Scene Traversable Handle**: Scene traversable handle can be set to 0 in an empty scene. It's a feature, not a bug.
- **Always Study APIs Thoroughly**: Never make assumptions about an API. Study the API first so you get it right the first time.
- **Premake Build System**: The project uses Premake5 for generating Visual Studio solutions
- **Memory System**: The project may have memory files stored by Serena MCP for context across conversations
- **Git Workflow**: Avoid making git commits unless explicitly requested - provide commit messages for user to execute
- **Handler System**: ALWAYS use the centralized Handler system in RenderContext. All handlers are accessible via `renderContext_->getHandlers()`. This provides:
  - Centralized initialization and cleanup through the Handlers struct
  - Shared access between all handlers (handlers can access each other)
  - Consistent factory pattern with `::create(RenderContextPtr)` methods
  - Proper lifecycle management with initialize/finalize patterns
  - Example: Access ScreenBufferHandler via `renderContext_->getHandlers().screenBufferHandler`
  - Never create handlers independently - always go through the Handlers struct


## Core Development Philosophy

### KISS (Keep It Simple, Stupid)
Simplicity should be a key goal in design. Choose straightforward solutions over complex ones whenever possible. Simple solutions are easier to understand, maintain, and debug.

### YAGNI (You Aren't Gonna Need It)
Avoid building functionality on speculation. Implement features only when they are needed, not when you anticipate they might be useful in the future.

### Design Principles
- **Dependency Inversion**: High-level modules should not depend on low-level modules. Both should depend on abstractions (interfaces/abstract classes).
- **Open/Closed Principle**: Software entities should be open for extension but closed for modification.
- **Single Responsibility**: Each function, class, and translation unit should have one clear purpose.
- **Fail Fast**: Use assertions, exceptions, and early validation to catch errors immediately when issues occur.

## 🧱 Code Structure & Modularity

### File and Function Limits
- **Header files should be under 300 lines** and source files **under 500 lines of code**. If approaching these limits, refactor by splitting into multiple translation units.
- **Functions should be under 50 lines** with a single, clear responsibility.
- **Classes should be under 150 lines** and represent a single concept or entity.
- **Organize code into clearly separated header/source pairs**, grouped by feature or responsibility.
- **Line length should be max 100 characters** - configure clang-format or your formatter accordingly.
- **Use the project's build system** (CMake/Make/etc.) whenever compiling and testing, including for unit tests.

### C++ Specific Guidelines
- **Follow RAII principles** - manage resources through constructors/destructors
- **Prefer stack allocation** over dynamic allocation when possible
- **Use smart pointers** (`std::unique_ptr`, `std::shared_ptr`) for dynamic memory management
- **Include guards or `#pragma once`** in all header files
- **Forward declarations** in headers when possible to reduce compilation dependencies
- **Const-correctness** - mark methods and parameters const when they don't modify state

## HTML Documentation Guidelines

### Code Block Styling
When creating HTML documentation with code blocks, always use clean, readable styling:
- **Background**: Light gray (#f8f8f8) with 1px solid #ddd border
- **Text Color**: Pure black (#000) for maximum contrast
- **Font**: 'Consolas', 'Courier New', monospace at 14px
- **Padding**: 20px for comfortable spacing
- **Line Height**: 1.6 for readability
- **Border Radius**: 4px for subtle rounding

Example CSS for code blocks:
```css
.code-block, pre {
    background: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 20px;
    margin: 24px 0;
    overflow-x: auto;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.6;
    color: #000;
}
```

### General HTML Documentation Design
- **Use minimal, clean design** with white backgrounds and black text
- **Avoid dark backgrounds** for code blocks - they reduce readability
- **Keep layouts spacious** with generous margins and padding
- **Maximum content width** around 900px for optimal reading
- **Simple color scheme** - use colors sparingly, mainly for links (#0066cc)