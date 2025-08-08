#pragma once

// IRenderingEngine.h
// Base interface for all rendering engines (path tracing, ReSTIR, SVGF, etc.)
// Each engine is a complete, self-contained rendering system that can be
// swapped at runtime based on scene characteristics or user preference.

// Forward declarations
class RenderContext;
namespace sabi {
    class Renderable;
    using RenderableNode = std::shared_ptr<Renderable>;
}

class IRenderingEngine
{
public:
    virtual ~IRenderingEngine() = default;
    
    // Lifecycle Management
    // Initialize the engine with the render context
    // This is where all engine-specific resources are created
    virtual void initialize(RenderContext* ctx) = 0;
    
    // Complete cleanup of all engine resources
    // After this call, the engine should consume zero GPU/CPU resources
    virtual void cleanup() = 0;
    
    // Scene Management
    // Add geometry from a renderable node to the engine's scene
    virtual void addGeometry(sabi::RenderableNode node) = 0;
    
    // Clear all geometry from the scene
    virtual void clearScene() = 0;
    
    // Rendering
    // Execute the engine's rendering pipeline for one frame
    virtual void render(const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber) = 0;
    
    // Environment update notification
    // Called when the environment map has changed (e.g., new HDR loaded)
    virtual void onEnvironmentChanged() = 0;
    
    // Engine Information
    // Get the engine's display name (e.g., "ReSTIR", "SVGF")
    virtual std::string getName() const = 0;
    
    // Get a description of what this engine does
    virtual std::string getDescription() const = 0;
};

// Shared pointer type for engines
using RenderingEnginePtr = std::shared_ptr<IRenderingEngine>;