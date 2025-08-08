#pragma once

// RenderEngineManager.h
// Manages the lifecycle of rendering engines and handles switching between them.
// Only one engine is active at a time to minimize resource usage.

#include "base/IRenderingEngine.h"

// Forward declarations
class RenderContext;
class GPUTimerManager;

// Factory function type for creating rendering engines
using EngineFactory = std::function<std::unique_ptr<IRenderingEngine>()>;

// Information about a rendering engine
struct EngineInfo
{
    std::string name;
    std::string description;
    std::vector<std::string> capabilities;
};

// Registration data for an engine (factory + metadata)
struct EngineRegistration
{
    EngineInfo info;
    EngineFactory factory;
};

class RenderEngineManager
{
public:
    RenderEngineManager();
    ~RenderEngineManager();
    
    // Initialize the manager with render context
    void initialize(RenderContext* ctx);
    
    // Set GPU timer manager (optional - for performance monitoring)
    void setGPUTimerManager(GPUTimerManager* timerManager);
    
    // Cleanup all resources
    void cleanup();
    
    // Engine Registration
    // Register a new engine type with its factory function and metadata
    void registerEngine(const std::string& name, 
                       const std::string& displayName,
                       const std::string& description,
                       EngineFactory factory);
    
    // Legacy registration (deprecated - for backward compatibility)
    void registerEngine(const std::string& name, EngineFactory factory);
    
    // Engine Management
    // Switch to a different engine (destroys current engine first)
    void switchEngine(const std::string& engineName);
    
    // Get list of all registered engines
    std::vector<std::string> getAvailableEngines() const;
    
    // Get information about a specific engine
    EngineInfo getEngineInfo(const std::string& engineName) const;
    
    // Get current active engine name (empty if none)
    std::string getCurrentEngineName() const { return currentEngineName_; }
    
    // Scene Management
    // Add geometry to the active engine
    void addGeometry(sabi::RenderableNode node);
    
    // Clear scene in the active engine
    void clearScene();
    
    // Notify the active engine that the environment has changed
    void onEnvironmentChanged();
    
    // Rendering
    // Render a frame using the active engine
    void render(const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber);
    
    // Check if an engine is currently active
    bool hasActiveEngine() const { return activeEngine_ != nullptr; }
    
    // Engine-specific controls
    void setShockerRenderMode(int mode);
    
private:
    // The currently active rendering engine (only one at a time)
    std::unique_ptr<IRenderingEngine> activeEngine_;
    
    // Registry of available engines (factory + metadata)
    std::map<std::string, EngineRegistration> engineRegistry_;
    
    // Render context (shared by all engines)
    RenderContext* renderContext_;
    
    // Name of the current active engine
    std::string currentEngineName_;
    
    // Flag to track initialization state
    bool isInitialized_;
    
    // GPU timer manager (optional)
    GPUTimerManager* gpuTimerManager_;
};