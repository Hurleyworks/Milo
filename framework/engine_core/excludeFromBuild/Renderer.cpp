#include "Renderer.h"
#include "nvcc/CudaCompiler.h"
#include "tools/PTXManager.h"
#include "tools/GPUTimerManager.h"
#include "engines/base/EngineRegistry.h"
#include <OpenImageIO/imagebuf.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

OIIO::ImageBuf createSkyWithSun (int width, int height,
                                 float sunAltitude = 0.4f,   // 0.0 = horizon, 1.0 = zenith
                                 float sunAzimuth = 0.5f,    // 0.0-1.0 maps to 0-360 degrees
                                 float sunSize = 0.01f,      // relative size of sun disk
                                 float sunIntensity = 50.0f) // HDR intensity of sun center
{
    // Create an ImageSpec with 3 channels (RGB) and float type for HDR
    OIIO::ImageSpec spec (width, height, 3, OIIO::TypeDesc::FLOAT);

    // Create the ImageBuf with that specification
    OIIO::ImageBuf buf (spec);

    // Sky colors - more realistic values
    float zenith_color[3] = {0.3f, 0.5f, 0.9f};  // Blue sky at zenith
    float horizon_color[3] = {0.7f, 0.8f, 1.0f}; // Brighter blue at horizon

    // Sun colors
    float sun_color[3] = {1.0f, 1.0f, 0.9f};      // Slightly yellow sun
    float sun_glow_color[3] = {1.0f, 0.9f, 0.7f}; // Warmer glow around sun

    // Calculate sun position in image space
    float sun_x = width * sunAzimuth;
    float sun_y = height * (1.0f - sunAltitude); // Invert Y to match altitude
    float sun_radius = std::min (width, height) * sunSize;

    // Calculate atmosphere thickness at horizon
    const float rayleigh_strength = 2.5f;

    // Process each pixel
    for (int y = 0; y < height; ++y)
    {
        // Vertical position factor (0 at horizon, 1 at zenith)
        float altitude = static_cast<float> (height - y - 1) / (height - 1);
        float thickness = 1.0f / (std::max (0.05f, altitude) * 0.5f + 0.5f);

        for (int x = 0; x < width; ++x)
        {
            // Base sky color - gradient from horizon to zenith
            float pixel_color[3];
            for (int c = 0; c < 3; ++c)
            {
                pixel_color[c] = horizon_color[c] * (1.0f - altitude) +
                                 zenith_color[c] * altitude;
            }

            // Atmospheric scattering (more reddish at horizon)
            float scatter = std::pow (1.0f - altitude, rayleigh_strength);
            pixel_color[0] = std::min (1.0f, pixel_color[0] + scatter * 0.2f);
            pixel_color[1] = std::min (1.0f, pixel_color[1] + scatter * 0.05f);
            pixel_color[2] = std::max (0.1f, pixel_color[2] - scatter * 0.2f);

            // Distance to sun (for drawing sun disk and glow)
            float dx = x - sun_x;
            float dy = y - sun_y;
            float dist_to_sun = std::sqrt (dx * dx + dy * dy);

            // Add sun disk
            if (dist_to_sun < sun_radius)
            {
                // Smooth edge for the sun
                float sun_factor = 1.0f - (dist_to_sun / sun_radius);
                sun_factor = std::pow (sun_factor, 0.5f); // Soften edge

                // Apply sun color and intensity (HDR value > 1.0)
                for (int c = 0; c < 3; ++c)
                {
                    pixel_color[c] = pixel_color[c] * (1.0f - sun_factor) +
                                     sun_color[c] * sun_factor * sunIntensity;
                }
            }
            // Add sun glow/halo
            else if (dist_to_sun < sun_radius * 10.0f)
            {
                float glow_factor = 1.0f - (dist_to_sun / (sun_radius * 10.0f));
                glow_factor = std::pow (glow_factor, 2.0f); // Squared for more natural falloff

                // Apply glow with softer intensity
                float glow_intensity = sunIntensity * 0.1f * glow_factor;
                for (int c = 0; c < 3; ++c)
                {
                    pixel_color[c] = pixel_color[c] + sun_glow_color[c] * glow_intensity;
                }
            }

            // Set pixel value
            buf.setpixel (x, y, pixel_color);
        }
    }

    return buf;
}

using sabi::RenderableNode;

// Default constructor initializes member variables to safe defaults
Renderer::Renderer() :
    renderContext_ (std::make_shared<RenderContext>())
{
    LOG (DBUG) << "Renderer constructor called";
}

// Destructor ensures proper cleanup even if Shutdown wasn't called
Renderer::~Renderer()
{
    LOG (DBUG) << "Renderer destructor called";
    // finalize();
}

void Renderer::init (MessageService messengers, const PropertyService& properties)
{
    LOG (DBUG) << _FN_;
    this->messengers = messengers;
    this->properties = properties;
}

void Renderer::initializeEngine (CameraHandle camera, ImageCacheHandlerPtr imageCache)
{
    LOG (DBUG) << _FN_;

    try
    {
        std::filesystem::path resourceFolder = properties.renderProps->getVal<std::string> (RenderKey::ResourceFolder);
        std::filesystem::path repoFolder = properties.renderProps->getVal<std::string> (RenderKey::RepoFolder);

        // Check build configuration for CUDA kernel compilation strategy
        bool softwareReleaseMode = properties.renderProps->getVal<bool> (RenderKey::SoftwareReleaseMode);
        bool embeddedPTX = properties.renderProps->getVal<bool> (RenderKey::UseEmbeddedPTX);

        // NB: Determine whether to compile CUDA kernels at runtime
        // Development workflow:
        // 1. Change compileCuda to true to force recompilation when CUDA source changes
        // 2. Run application once to compile fresh PTX files
        // 3. Run ptx_embed.bat to update generated/embedded_ptx.h with new binaries
        // 4. Restore original condition and rebuild application
        // 5. Application will then use the embedded PTX files

        bool compileCuda = false;
        std::string engineFilter = "all";  // Can be "all", "milo", or "shocker"

        if (compileCuda || (!softwareReleaseMode && !embeddedPTX))
        {
            CudaCompiler nvcc;

            // Define the architectures to compile for
            std::vector<std::string> targetArchitectures;

            // Try to load from properties if available
            try
            {
                std::string archList = properties.renderProps->getVal<std::string> (RenderKey::CudaTargetArchitectures);
                if (!archList.empty())
                {
                    // Parse comma-separated list of architectures
                    size_t pos = 0;
                    while ((pos = archList.find (',')) != std::string::npos)
                    {
                        targetArchitectures.push_back (archList.substr (0, pos));
                        archList.erase (0, pos + 1);
                    }
                    if (!archList.empty())
                    {
                        targetArchitectures.push_back (archList);
                    }
                }
            }
            catch (...)
            {
                // Property not found, using defaults
            }

            // If no architectures specified in properties, use defaults
            if (targetArchitectures.empty())
            {
                targetArchitectures = {"sm_60", "sm_75", "sm_80", "sm_86", "sm_90"};
            }

            // Log which architectures we're compiling for
            LOG (DBUG) << "Compiling CUDA kernels for the following architectures:";
            for (const auto& arch : targetArchitectures)
            {
                LOG (DBUG) << "  - " << arch;
            }

            // Compile for all target architectures with engine filter
            nvcc.compile (resourceFolder, repoFolder, targetArchitectures, engineFilter);
        }

        if(compileCuda)
            return;

        // Step 1: Initialize Render Context with system resources
        // Skip pipeline initialization - we're using engine system only
        bool skipPipelineInit = true;
        if (!renderContext_->initialize (camera, imageCache, properties, skipPipelineInit))
        {
            throw std::runtime_error ("Failed to initialize render context");
        }

        // Step 3: Initialize PTX Manager
        ptxManager_ = std::make_unique<PTXManager>();
        ptxManager_->initialize (resourceFolder);

        // Step 4: Set PTX manager in render context
        renderContext_->setPTXManager (ptxManager_.get());

        // Step 5: Initialize GPU Timer Manager
        gpuTimerManager_ = std::make_unique<GPUTimerManager>();
        if (!gpuTimerManager_->initialize (renderContext_))
        {
            LOG (WARNING) << "Failed to initialize GPU Timer Manager";
            // Continue without GPU timing support
            gpuTimerManager_.reset();
        }

        // Step 6: Initialize Rendering Engine Manager
        engineManager_ = std::make_unique<RenderEngineManager>();
        engineManager_->initialize(renderContext_.get());
        
        // Pass GPU timer manager to engine manager
        if (gpuTimerManager_)
        {
            engineManager_->setGPUTimerManager(gpuTimerManager_.get());
        }
        
        // Register all built-in engines
        registerBuiltInEngines(*engineManager_);
        
        // Log available engines
        auto engines = engineManager_->getAvailableEngines();
        LOG(INFO) << "Available rendering engines:";
        for (const auto& engine : engines) {
            auto info = engineManager_->getEngineInfo(engine);
            LOG(INFO) << "  - " << info.name << ": " << info.description;
        }
        
        // Start with basic path tracer as default
        engineManager_->switchEngine("milo");

        // Step 7: Pipeline Handler is already initialized for geometry support
        // The scene pipeline is active by default for model creation
        // Engines will be used for actual rendering
        LOG(INFO) << "Using rendering engine system for rendering with pipeline geometry support";

        initialized_ = true;
        LOG (INFO) << "Milo rendering engine initialized successfully";
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Failed to initialize rendering engine: " << e.what();
        finalize();
        throw;
    }
}

void Renderer::render (const InputEvent& input, bool updateMotion, uint32_t frameNumber)
{
    try
    {
        if (!initialized_)
        {
            LOG (WARNING) << "Renderer not initialized";
            return;
        }

        // Use the new engine system if available
        if (engineManager_ && engineManager_->hasActiveEngine()) {
            engineManager_->render(input, updateMotion, frameNumber);
            return;
        }


    }
    catch (std::exception& e)
    {
        LOG (WARNING) << e.what();
        throw;
    }
}

void Renderer::addSkyDomeHDR (const std::filesystem::path& hdrPath)
{
    LOG (DBUG) << _FN_ << "   " << hdrPath.generic_string();

    if (!renderContext_)
    {
        LOG (WARNING) << "Render context not initialized";
        return;
    }

   

    if (!std::filesystem::exists (hdrPath))
    {
        LOG (WARNING) << "Environment HDR file not found: " << hdrPath.generic_string();

         OIIO::ImageBuf sky = createSkyWithSun (2048, 1024);

        return;
    }

    try
    {
        OIIO::ImageBuf image (hdrPath.generic_string());

        if (!image.has_error())
        {
            // Get the SkyDomeHandler from the render context
            auto& handlers = renderContext_->getHandlers();
            if (handlers.skyDomeHandler)
            {
                // Add the sky dome image to the handler
                handlers.skyDomeHandler->addSkyDomeImage (std::move (image));
                LOG (INFO) << "Successfully loaded sky dome HDR: " << hdrPath.generic_string();

                // Store the HDR path for pipeline switching
                currentSkyDomeHDR_ = hdrPath;
                
                // Notify the active render engine that the environment has changed
                if (engineManager_)
                {
                    engineManager_->onEnvironmentChanged();
                }
            }
            else
            {
                LOG (WARNING) << "SkyDomeHandler not initialized";
            }
        }
        else
        {
            LOG (WARNING) << "Failed to load HDR image: " << image.geterror();
        }
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Exception loading HDR file: " << e.what();
    }
}

// already checked upstream for expiredness
void Renderer::addRenderableNode (RenderableWeakRef& weakNode)
{
    RenderableNode node = weakNode.lock();
    if (!node)
    {
        LOG (WARNING) << "Cannot add renderable node: weak reference is expired";
        return;
    }

    // Store the node for pipeline switching
    renderableNodes_.push_back (weakNode);

    // Check if we have a valid render context
    if (!renderContext_)
    {
        LOG (WARNING) << "Cannot add renderable node: render context not initialized";
        return;
    }


    if (engineManager_ && engineManager_->hasActiveEngine())
    {
        engineManager_->addGeometry(node);
        LOG (INFO) << "Added geometry '" << node->getName() << "' to active engine";
    }
    else
    {
        LOG (WARNING) << "No active engine to add geometry to";
    }

}

void Renderer::finalize()
{
    if (initialized_)
    {
        // Finalize Rendering Engine Manager
        if (engineManager_)
        {
            engineManager_->cleanup();
            engineManager_.reset();
        }
        
        // Finalize Pipeline Handler
        // Pipeline handler cleanup is handled by RenderContext handlers

        // Finalize GPU Timer Manager
        if (gpuTimerManager_)
        {
            gpuTimerManager_->finalize();
            gpuTimerManager_.reset();
        }

        if (renderContext_)
        {
            renderContext_->cleanup();
        }
        initialized_ = false;
        LOG (INFO) << "Milo rendering engine finalized";
    }
}





// Engine management methods (new system)
bool Renderer::setEngine(const std::string& engineName)
{
    if (!engineManager_)
    {
        LOG(WARNING) << "Cannot set engine: Engine manager not initialized";
        return false;
    }
    
    // Switch to the new engine
    engineManager_->switchEngine(engineName);
    
    // Check if switch was successful
    if (engineManager_->getCurrentEngineName() != engineName)
    {
        return false;
    }
    
    // Re-add all stored geometry to the new engine
    if (engineManager_->hasActiveEngine())
    {
        // Re-add sky dome if one was set
        if (!currentSkyDomeHDR_.empty())
        {
            // Re-add the sky dome HDR to the engine
            addSkyDomeHDR(currentSkyDomeHDR_);
        }
        
        // Re-add all geometry nodes using the engine manager
        for (auto& weakNode : renderableNodes_)
        {
            auto node = weakNode.lock();
            if (node)
            {
                engineManager_->addGeometry(node);
            }
        }
        
        LOG(INFO) << "Re-added " << renderableNodes_.size() << " geometry nodes to engine: " << engineName;
    }
    
    return true;
}

std::string Renderer::getCurrentEngineName() const
{
    if (engineManager_)
    {
        return engineManager_->getCurrentEngineName();
    }
    return "";
}

std::vector<std::string> Renderer::getAvailableEngines() const
{
    if (engineManager_)
    {
        return engineManager_->getAvailableEngines();
    }
    return {};
}

void Renderer::setShockerRenderMode(int mode)
{
    if (!engineManager_)
    {
        LOG(WARNING) << "Cannot set Shocker render mode: Engine manager not initialized";
        return;
    }
    
    // Delegate to the engine manager
    engineManager_->setShockerRenderMode(mode);
}
