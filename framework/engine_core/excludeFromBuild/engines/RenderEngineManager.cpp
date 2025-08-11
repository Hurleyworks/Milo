#include "RenderEngineManager.h"
#include "base/BaseRenderingEngine.h"
#include "ripr/RiPREngine.h"
#include "../RenderContext.h"

RenderEngineManager::RenderEngineManager() :
    renderContext_ (nullptr), isInitialized_ (false), gpuTimerManager_(nullptr)
{
    LOG (INFO) << "RenderEngineManager created";
}

RenderEngineManager::~RenderEngineManager()
{
    cleanup();
}

void RenderEngineManager::initialize (RenderContext* ctx)
{
    LOG (INFO) << "RenderEngineManager::initialize()";

    if (isInitialized_)
    {
        LOG (WARNING) << "RenderEngineManager already initialized";
        return;
    }

    if (!ctx)
    {
        LOG (WARNING) << "RenderEngineManager::initialize() called with null context";
        return;
    }

    renderContext_ = ctx;
    isInitialized_ = true;
}

void RenderEngineManager::setGPUTimerManager(GPUTimerManager* timerManager)
{
    gpuTimerManager_ = timerManager;
}

void RenderEngineManager::cleanup()
{
    LOG (INFO) << "RenderEngineManager::cleanup()";

    // Destroy active engine if any
    if (activeEngine_)
    {
        LOG (INFO) << "Cleaning up active engine: " << currentEngineName_;
        activeEngine_->cleanup();
        activeEngine_.reset();
        currentEngineName_.clear();
    }

    // Clear registry (factories remain valid)
    // Note: We don't clear engineFactories_ as they can be reused

    renderContext_ = nullptr;
    isInitialized_ = false;
}

void RenderEngineManager::registerEngine(const std::string& name,
                                       const std::string& displayName,
                                       const std::string& description,
                                       EngineFactory factory)
{
    LOG(INFO) << "Registering engine: " << name;

    if (name.empty())
    {
        LOG(WARNING) << "Cannot register engine with empty name";
        return;
    }

    if (!factory)
    {
        LOG(WARNING) << "Cannot register engine '" << name << "' with null factory";
        return;
    }

    // Create registration with metadata
    EngineRegistration registration;
    registration.info.name = displayName.empty() ? name : displayName;
    registration.info.description = description;
    registration.factory = factory;

    // Overwrite if already exists
    engineRegistry_[name] = registration;
}

// Legacy registration method (deprecated)
void RenderEngineManager::registerEngine(const std::string& name, EngineFactory factory)
{
    // For backward compatibility, use the name as display name and empty description
    registerEngine(name, name, "No description available", factory);
}

void RenderEngineManager::switchEngine (const std::string& engineName)
{
    LOG (INFO) << "Switching to engine: " << engineName;

    if (!isInitialized_)
    {
        LOG (WARNING) << "RenderEngineManager not initialized";
        return;
    }

    if (!renderContext_)
    {
        LOG (WARNING) << "RenderContext is null";
        return;
    }

    // Check if engine exists
    auto it = engineRegistry_.find(engineName);
    if (it == engineRegistry_.end())
    {
        LOG(WARNING) << "Engine '" << engineName << "' not found in registry";
        return;
    }

    // If same engine, do nothing
    if (engineName == currentEngineName_ && activeEngine_)
    {
        LOG (INFO) << "Already using engine: " << engineName;
        return;
    }

    // Step 1: Cleanup current engine completely
    if (activeEngine_)
    {
        LOG (INFO) << "Cleaning up current engine: " << currentEngineName_;
        activeEngine_->cleanup();
        activeEngine_.reset();
        currentEngineName_.clear();
        
    }

    // Step 2: Create new engine
    try
    {
        LOG (INFO) << "Creating new engine: " << engineName;
        activeEngine_ = it->second.factory();

        if (!activeEngine_)
        {
            LOG (WARNING) << "Factory returned null engine for: " << engineName;
            return;
        }

        // Step 3: Initialize new engine
        LOG (INFO) << "Initializing engine: " << engineName;
        activeEngine_->initialize (renderContext_);
        
        // Step 4: Set GPU timer manager if engine inherits from BaseRenderingEngine
        if (gpuTimerManager_)
        {
            auto* baseEngine = dynamic_cast<BaseRenderingEngine*>(activeEngine_.get());
            if (baseEngine)
            {
                baseEngine->setGPUTimerManager(gpuTimerManager_);
                LOG (DBUG) << "Set GPU timer manager for engine: " << engineName;
            }
        }

        currentEngineName_ = engineName;
        LOG (INFO) << "Successfully switched to engine: " << engineName;
        
        // Mark camera as dirty to trigger a render update
        auto camera = renderContext_->getCamera();
        if (camera)
        {
            camera->setDirty(true);
            LOG (DBUG) << "Marked camera as dirty after engine switch";
        }
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Failed to create/initialize engine '" << engineName << "': " << e.what();
        activeEngine_.reset();
        currentEngineName_.clear();
    }
}

std::vector<std::string> RenderEngineManager::getAvailableEngines() const
{
    std::vector<std::string> engines;
    engines.reserve(engineRegistry_.size());

    for (const auto& [name, registration] : engineRegistry_)
    {
        engines.push_back(name);
    }

    return engines;
}

EngineInfo RenderEngineManager::getEngineInfo(const std::string& engineName) const
{
    EngineInfo info;

    // Simply return the stored metadata - no instantiation needed!
    auto it = engineRegistry_.find(engineName);
    if (it != engineRegistry_.end())
    {
        info = it->second.info;
    }
    else
    {
        // Engine not found, return empty info
        info.name = engineName;
        info.description = "Engine not found";
    }

    return info;
}

void RenderEngineManager::render (const mace::InputEvent& input, bool updateMotion, uint32_t frameNumber)
{
    if (!activeEngine_)
    {
        // No active engine, nothing to render
        return;
    }

    activeEngine_->render (input, updateMotion, frameNumber);
}

void RenderEngineManager::addGeometry(sabi::RenderableNode node)
{
    if (!activeEngine_)
    {
        LOG(WARNING) << "No active engine to add geometry to";
        return;
    }
    
    activeEngine_->addGeometry(node);
}

void RenderEngineManager::clearScene()
{
    if (!activeEngine_)
    {
        LOG(WARNING) << "No active engine to clear scene from";
        return;
    }
    
    activeEngine_->clearScene();
}

void RenderEngineManager::onEnvironmentChanged()
{
    if (!activeEngine_)
    {
        LOG(WARNING) << "No active engine to notify about environment change";
        return;
    }
    
    activeEngine_->onEnvironmentChanged();
    LOG(DBUG) << "Notified active engine about environment change";
}

void RenderEngineManager::setRiPRRenderMode(int mode)
{
    if (!activeEngine_)
    {
        LOG(WARNING) << "No active engine to set render mode";
        return;
    }
    
    // Check if it's the RiPREngine
    if (currentEngineName_ == "ripr")
    {
        // Cast to RiPREngine and set render mode
        if (auto riprEngine = dynamic_cast<RiPREngine*>(activeEngine_.get()))
        {
            riprEngine->setRenderMode(static_cast<RiPREngine::RenderMode>(mode));
            LOG(INFO) << "Set RiPR render mode to: " << mode;
        }
        else
        {
            LOG(WARNING) << "Failed to cast to RiPREngine";
        }
    }
    else
    {
        LOG(WARNING) << "RiPR render mode can only be set when RiPREngine is active. Current engine: " << currentEngineName_;
    }
}