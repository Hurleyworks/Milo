#include "Jahley.h"

const std::string APP_NAME = "AreaLightHandlerTest";

#ifdef CHECK
#undef CHECK
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;

#include <engine_core/engine_core.h>

// Include AreaLightHandler and dependencies
#include "engine_core/excludeFromBuild/handlers/AreaLightHandler.h"
#include "engine_core/excludeFromBuild/handlers/ShockerSceneHandler.h"
#include "engine_core/excludeFromBuild/handlers/ShockerModelHandler.h"
#include "engine_core/excludeFromBuild/handlers/ShockerMaterialHandler.h"
#include "engine_core/excludeFromBuild/model/ShockerModel.h"
#include "engine_core/excludeFromBuild/model/ShockerCore.h"
#include "engine_core/excludeFromBuild/material/HostDisneyMaterial.h"
#include "engine_core/excludeFromBuild/common/common_host.h"

using sabi::RenderableNode;

static RenderableNode loadGLTF (const std::filesystem::path& gltfPath)
{
    GLTFImporter gltf;
    //  std::vector<Animation> animations;
    auto [cgModel, animations] = gltf.importModel (gltfPath.generic_string());
    if (!cgModel)
    {
        LOG (WARNING) << "Load failed " << gltfPath.string();
        return nullptr;
    }

    RenderableNode node = sabi::WorldItem::create();
    node->setClientID (node->getID());
    node->setModel (cgModel);
    node->getState().state |= sabi::PRenderableState::Visible;

    // Set the model path in the description so texture loading can find the content folder
    sabi::RenderableDesc desc = node->description();
    desc.modelPath = gltfPath;
    node->setDescription (desc);

    std::string modelName = getFileNameWithoutExtension (gltfPath);

    for (auto& s : cgModel->S)
    {
        s.vertexCount = cgModel->vertexCount();
    }

    return node;
}

static RenderableNode createWarmLight()
{
    // Create a warm white mesh light with custom size and intensity
    RenderableNode warmLight = sabi::MeshOps::createLuminousRectangleNode (
        "WarmMeshLight",
        4.0f, 2.0f,                         // 4x2 units
        Eigen::Vector3f (1.0f, 0.9f, 0.8f), // Warm white color
        10.0f                               // High intensity
    );

      warmLight->setClientID (warmLight->getID());
    sabi::SpaceTime& st = warmLight->getSpaceTime();
    st.worldTransform.translation() = Eigen::Vector3f (0.0f, 1.0f, -4.0f);

    return warmLight;
}

// Helper class to initialize GPU context for testing
class GPUTestContext
{
 public:
    RenderContextPtr renderContext;
    TextureHandlerPtr textureHandler;
    bool initialized = false;

    GPUTestContext()
    {
        try
        {
            // Create RenderContext (it will initialize CUDA and OptiX internally)
            renderContext = std::make_shared<RenderContext>();
            if (!renderContext->initialize())
            {
                LOG (WARNING) << "Failed to initialize RenderContext, GPU tests will be skipped";
                return;
            }

            // Create and initialize TextureHandler
            textureHandler = TextureHandler::create (renderContext);

            // Set up handlers
            renderContext->getHandlers().textureHandler = textureHandler;

            initialized = true;
            LOG (INFO) << "GPU test context initialized successfully";
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Failed to initialize GPU test context: " << e.what();
            initialized = false;
        }
    }

    ~GPUTestContext()
    {
        if (initialized)
        {
            // Clean up in reverse order
            textureHandler.reset();

            if (renderContext)
            {
                renderContext->cleanup();
                renderContext.reset();
            }

            LOG (INFO) << "GPU test context cleaned up";
        }
    }

    bool isAvailable() const { return initialized; }
};






TEST_SUITE("AreaLightHandler")
{
    TEST_CASE("Simple WarmLight Test")
    {
        GPUTestContext gpuContext;
        if (!gpuContext.isAvailable())
        {
            WARN("GPU not available, skipping GPU tests");
            return;
        }
        
        // Create handlers with proper shared pointers
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        auto materialHandler = ShockerMaterialHandler::create();
        auto sceneHandler = ShockerSceneHandler::create(gpuContext.renderContext);
        auto areaLightHandler = std::make_shared<AreaLightHandler>();
        
        // Initialize handlers
        modelHandler->initialize(gpuContext.renderContext);
        materialHandler->initialize(gpuContext.renderContext.get());
        sceneHandler->initialize();
        areaLightHandler->initialize(gpuContext.renderContext->getCudaContext(), 100);
        
        // Set dependencies
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        sceneHandler->setAreaLightHandler(areaLightHandler);
        
        modelHandler->setMaterialHandler(materialHandler.get());
        modelHandler->setAreaLightHandler(areaLightHandler);
        
        materialHandler->setAreaLightHandler(areaLightHandler);
        
        areaLightHandler->setSceneHandler(sceneHandler.get());
        areaLightHandler->setModelHandler(modelHandler.get());
        areaLightHandler->setMaterialHandler(materialHandler.get());
        
        // Create warm light
        RenderableNode warmLight = createWarmLight();
        CHECK(warmLight != nullptr);
        CHECK(warmLight->getModel() != nullptr);
        
        // Process the warm light through the scene handler
        sceneHandler->processRenderableNode(warmLight);
        
        // Check statistics
        CHECK(sceneHandler->getNodeCount() > 0);
        
        // Since the handlers don't automatically notify AreaLightHandler yet,
        // we need to manually check for emissive surfaces and notify
        
        // Get all models from the model handler
        const auto& models = modelHandler->getAllModels();
        CHECK(!models.empty());
        
        // Check each model's surfaces for emissive materials
        for (const auto& [name, model] : models)
        {
            const auto& surfaces = model->getSurfaces();
            for (const auto& surface : surfaces)
            {
                if (surface && surface->mat && surface->mat->emissive)
                {
                    // Found an emissive surface - notify the area light handler
                    areaLightHandler->onSurfaceAdded(surface.get());
                }
            }
        }
        
        // Now check if area lights were detected
        CHECK(areaLightHandler->hasAreaLights());
        CHECK(areaLightHandler->getNumAreaLights() == 1);
        
        // Clean up
        sceneHandler->clear();
        
        // Finalize handlers
        areaLightHandler->finalize();
        materialHandler->clear();
        modelHandler->clear();
    }
    
    TEST_CASE("Multiple Area Lights and Removal")
    {
        GPUTestContext gpuContext;
        if (!gpuContext.isAvailable())
        {
            WARN("GPU not available, skipping GPU tests");
            return;
        }
        
        // Create handlers with proper shared pointers
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        auto materialHandler = ShockerMaterialHandler::create();
        auto sceneHandler = ShockerSceneHandler::create(gpuContext.renderContext);
        auto areaLightHandler = std::make_shared<AreaLightHandler>();
        
        // Initialize handlers
        modelHandler->initialize(gpuContext.renderContext);
        materialHandler->initialize(gpuContext.renderContext.get());
        sceneHandler->initialize();
        areaLightHandler->initialize(gpuContext.renderContext->getCudaContext(), 100);
        
        // Set dependencies
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        sceneHandler->setAreaLightHandler(areaLightHandler);
        
        modelHandler->setMaterialHandler(materialHandler.get());
        modelHandler->setAreaLightHandler(areaLightHandler);
        
        materialHandler->setAreaLightHandler(areaLightHandler);
        
        areaLightHandler->setSceneHandler(sceneHandler.get());
        areaLightHandler->setModelHandler(modelHandler.get());
        areaLightHandler->setMaterialHandler(materialHandler.get());
        
        // Create multiple light sources with different properties
        RenderableNode warmLight1 = sabi::MeshOps::createLuminousRectangleNode(
            "WarmLight1", 
            2.0f, 1.0f,                          // 2x1 units
            Eigen::Vector3f(1.0f, 0.8f, 0.6f),  // Warm orange
            5.0f                                 // Medium intensity
        );
        
        RenderableNode coolLight = sabi::MeshOps::createLuminousRectangleNode(
            "CoolLight",
            3.0f, 3.0f,                          // 3x3 units
            Eigen::Vector3f(0.6f, 0.8f, 1.0f),  // Cool blue
            8.0f                                 // Higher intensity
        );
        
        RenderableNode brightLight = sabi::MeshOps::createLuminousRectangleNode(
            "BrightLight",
            1.0f, 1.0f,                          // 1x1 units
            Eigen::Vector3f(1.0f, 1.0f, 1.0f),  // Pure white
            20.0f                                // Very high intensity
        );
        
        // Position the lights differently
        warmLight1->getSpaceTime().worldTransform.translation() = Eigen::Vector3f(-3.0f, 2.0f, 0.0f);
        coolLight->getSpaceTime().worldTransform.translation() = Eigen::Vector3f(3.0f, 2.0f, 0.0f);
        brightLight->getSpaceTime().worldTransform.translation() = Eigen::Vector3f(0.0f, 4.0f, 0.0f);
        
        // Process all lights through the scene handler
        sceneHandler->processRenderableNode(warmLight1);
        sceneHandler->processRenderableNode(coolLight);
        sceneHandler->processRenderableNode(brightLight);
        
        CHECK(sceneHandler->getNodeCount() == 3);
        
        // Collect all emissive surfaces and notify area light handler
        std::vector<shocker::ShockerSurface*> emissiveSurfaces;
        
        const auto& models = modelHandler->getAllModels();
        for (const auto& [name, model] : models)
        {
            const auto& surfaces = model->getSurfaces();
            for (const auto& surface : surfaces)
            {
                if (surface && surface->mat && surface->mat->emissive)
                {
                    emissiveSurfaces.push_back(surface.get());
                    areaLightHandler->onSurfaceAdded(surface.get());
                }
            }
        }
        
        // Verify all three lights were detected
        CHECK(areaLightHandler->hasAreaLights());
        CHECK(areaLightHandler->getNumAreaLights() == 3);
        CHECK(areaLightHandler->isDirty());
        
        // Update distributions
        CUstream stream = 0;
        areaLightHandler->updateAreaLightDistributions(stream);
        CHECK(!areaLightHandler->isDirty());
        
        // Remove one light (the middle one)
        if (emissiveSurfaces.size() > 1)
        {
            areaLightHandler->onSurfaceRemoved(emissiveSurfaces[1]);
            CHECK(areaLightHandler->getNumAreaLights() == 2);
            CHECK(areaLightHandler->isDirty());
        }
        
        // Test material change - convert an emissive surface to non-emissive
        if (!emissiveSurfaces.empty())
        {
            shocker::ShockerSurface* surface = emissiveSurfaces[0];
            DisneyMaterial* oldMaterial = const_cast<DisneyMaterial*>(surface->mat);
            
            // Create a non-emissive material (using default material)
            DisneyMaterial* defaultMat = materialHandler->getMaterial(0);
            
            // Notify about material change
            areaLightHandler->onMaterialAssigned(surface, oldMaterial, defaultMat);
            
            // Should have one less area light
            CHECK(areaLightHandler->getNumAreaLights() == 1);
        }
        
        // Clean up remaining surfaces
        for (auto* surface : emissiveSurfaces)
        {
            // Only remove if still tracked (we already removed some)
            areaLightHandler->onSurfaceRemoved(surface);
        }
        
        CHECK(!areaLightHandler->hasAreaLights());
        CHECK(areaLightHandler->getNumAreaLights() == 0);
        
        // Clean up
        sceneHandler->clear();
        
        // Finalize handlers
        areaLightHandler->finalize();
        materialHandler->clear();
        modelHandler->clear();
    }
}

class Application : public Jahley::App
{
public:
    Application(DesktopWindowSettings settings = DesktopWindowSettings(), bool windowApp = false) :
        Jahley::App()
    {
        doctest::Context().run();
    }

private:
};

Jahley::App* Jahley::CreateApplication()
{
    return new Application();
}

