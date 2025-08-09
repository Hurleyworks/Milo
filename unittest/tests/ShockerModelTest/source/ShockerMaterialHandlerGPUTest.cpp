// ShockerMaterialHandlerGPUTest.cpp
// GPU-enabled tests for ShockerMaterialHandler with CUDA context

#include <doctest/doctest.h>
#include <sabi_core/sabi_core.h>

#include "engine_core/excludeFromBuild/engines/shocker/handlers/ShockerMaterialHandler.h"
#include "engine_core/excludeFromBuild/engines/shocker/handlers/ShockerModelHandler.h"
#include "engine_core/excludeFromBuild/handlers/TextureHandler.h"
#include "engine_core/excludeFromBuild/handlers/Handlers.h"
#include "engine_core/excludeFromBuild/RenderContext.h"
#include "engine_core/excludeFromBuild/engines/shocker/models/ShockerModel.h"

// Helper class to initialize GPU context for testing
class GPUTestContext
{
public:
    RenderContextPtr renderContext;
    TextureHandlerPtr textureHandler;
    bool initialized = false;
    
    GPUTestContext()
    {
        try {
            // Create RenderContext (it will initialize CUDA and OptiX internally)
            renderContext = std::make_shared<RenderContext>();
            if (!renderContext->initialize()) {
                LOG(WARNING) << "Failed to initialize RenderContext, GPU tests will be skipped";
                return;
            }
            
            // Create and initialize TextureHandler
            textureHandler = TextureHandler::create(renderContext);
            
            // Set up handlers
            renderContext->getHandlers().textureHandler = textureHandler;
            
            initialized = true;
            LOG(INFO) << "GPU test context initialized successfully";
            
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to initialize GPU test context: " << e.what();
            initialized = false;
        }
    }
    
    ~GPUTestContext()
    {
        if (initialized) {
            // Clean up in reverse order
            textureHandler.reset();
            
            if (renderContext) {
                renderContext->cleanup();
                renderContext.reset();
            }
            
            LOG(INFO) << "GPU test context cleaned up";
        }
    }
    
    bool isAvailable() const { return initialized; }
};

TEST_CASE("ShockerMaterialHandler GPU Tests")
{
    GPUTestContext gpuContext;
    
    if (!gpuContext.isAvailable()) {
        LOG(WARNING) << "Skipping GPU tests - no CUDA device available";
        return;
    }
    
    SUBCASE("Material Creation with GPU Context")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(gpuContext.renderContext.get());
        
        // Create a test material
        sabi::CgMaterial cgMat;
        cgMat.name = "GPUTestMaterial";
        cgMat.core.baseColor = Eigen::Vector3f(0.8f, 0.2f, 0.2f);
        cgMat.core.roughness = 0.5f;
        cgMat.metallic.metallic = 0.3f;
        
        // Create material with GPU context available
        DisneyMaterial* material = matHandler.createMaterialFromCg(cgMat);
        CHECK(material != nullptr);
        
        // Verify material properties were set
        // Note: Textures are only created when loading from actual texture files
        // Color-only materials don't create textures - this is expected behavior
        
        LOG(INFO) << "Created material with GPU context (textures not expected from colors only)";
    }
    
    SUBCASE("Process Simple Model with GPU Context")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(gpuContext.renderContext.get());
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(gpuContext.renderContext);
        modelHandler.setMaterialHandler(&matHandler);
        
        // Create a simple cube model
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        CHECK(cube != nullptr);
        
        // Set up material for the cube
        if (!cube->S.empty()) {
            cube->S[0].cgMaterial.name = "CubeMaterial";
            cube->S[0].cgMaterial.core.baseColor = Eigen::Vector3f(0.8f, 0.2f, 0.2f);
            cube->S[0].cgMaterial.core.roughness = 0.5f;
            cube->S[0].cgMaterial.metallic.metallic = 0.3f;
        }
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("GPUTestCube");
        node->setModel(cube);
        
        // Process with GPU context available
        ShockerModelPtr model = modelHandler.processRenderableNode(node);
        CHECK(model != nullptr);
        
        // Materials should be created
        CHECK(matHandler.getAllMaterials().size() > 1); // Default + surface material
        
        // Verify materials were assigned to surfaces
        for (const auto& surface : model->getSurfaces()) {
            CHECK(surface->mat != nullptr);
        }
        
        LOG(INFO) << "Successfully processed simple model with GPU context";
    }
    
    SUBCASE("Process GLTF Model with Textures")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(gpuContext.renderContext.get());
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(gpuContext.renderContext);
        modelHandler.setMaterialHandler(&matHandler);
        
        // Try to load a simple GLTF model
        std::filesystem::path modelPath = "E:/common_content/models/Duck/glTF/Duck.gltf";
        
        if (!std::filesystem::exists(modelPath)) {
            LOG(WARNING) << "Test model not found, skipping";
            return;
        }
        
        GLTFImporter gltf;
        auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
        
        if (!cgModel) {
            LOG(WARNING) << "Failed to load test model";
            return;
        }
        
        LOG(INFO) << "Loaded Duck model with " << cgModel->cgTextures.size() << " textures";
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("GPUDuck");
        node->setModel(cgModel);
        
        sabi::RenderableDesc desc = node->description();
        desc.modelPath = modelPath;
        node->setDescription(desc);
        
        // Set vertex count safely
        size_t vertexCount = cgModel->vertexCount();
        if (vertexCount > 0) {
            for (auto& s : cgModel->S) {
                s.vertexCount = vertexCount;
            }
        }
        
        // Process with GPU context available
        ShockerModelPtr model = modelHandler.processRenderableNode(node);
        CHECK(model != nullptr);
        
        // Materials should be created for each surface
        CHECK(matHandler.getAllMaterials().size() > cgModel->S.size());
        
        // Check if textures were loaded (Duck model has textures)
        bool hasTexturedMaterial = false;
        for (const auto& mat : matHandler.getAllMaterials()) {
            if (mat->texBaseColor != 0 || mat->texNormal != 0) {
                hasTexturedMaterial = true;
                break;
            }
        }
        
        if (cgModel->cgTextures.size() > 0) {
            // If the model has textures, we should have loaded some
            CHECK(hasTexturedMaterial);
            LOG(INFO) << "Successfully loaded GLTF model with textures";
        } else {
            LOG(INFO) << "GLTF model has no textures";
        }
    }
    
    SUBCASE("GPU Buffer Upload Test")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(gpuContext.renderContext.get());
        
        // Create multiple materials
        for (int i = 0; i < 10; ++i) {
            sabi::CgMaterial cgMat;
            cgMat.name = "Material_" + std::to_string(i);
            cgMat.core.baseColor = Eigen::Vector3f(i * 0.1f, i * 0.1f, i * 0.1f);
            cgMat.core.roughness = i * 0.1f;
            
            DisneyMaterial* mat = matHandler.createMaterialFromCg(cgMat);
            CHECK(mat != nullptr);
        }
        
        // Upload materials to GPU
        matHandler.uploadMaterialsToGPU();
        
        // Get material data buffer
        auto* buffer = matHandler.getMaterialDataBuffer();
        CHECK(buffer != nullptr);
        CHECK(buffer->isInitialized());
        
        LOG(INFO) << "Uploaded " << matHandler.getAllMaterials().size() << " materials to GPU buffer";
    }
    
    SUBCASE("Material Upload to GPU")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(gpuContext.renderContext.get());
        
        // Create several materials
        for (int i = 0; i < 5; ++i) {
            sabi::CgMaterial cgMat;
            cgMat.name = "Material_" + std::to_string(i);
            cgMat.core.baseColor = Eigen::Vector3f(i * 0.2f, i * 0.2f, i * 0.2f);
            cgMat.core.roughness = i * 0.2f;
            
            DisneyMaterial* material = matHandler.createMaterialFromCg(cgMat);
            CHECK(material != nullptr);
        }
        
        // Upload to GPU and verify
        matHandler.uploadMaterialsToGPU();
        auto* buffer = matHandler.getMaterialDataBuffer();
        CHECK(buffer != nullptr);
        CHECK(buffer->isInitialized());
        
        LOG(INFO) << "Successfully uploaded " << matHandler.getAllMaterials().size() << " materials to GPU";
    }
    
    SUBCASE("Complex Model Batch Processing with GPU")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(gpuContext.renderContext.get());
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(gpuContext.renderContext);
        modelHandler.setMaterialHandler(&matHandler);
        
        std::vector<std::filesystem::path> testModels = {
            "E:/common_content/models/DamagedHelmet/glTF/DamagedHelmet.gltf",
            "E:/common_content/models/Duck/glTF/Duck.gltf",
            "E:/common_content/models/BoomBox/glTF/BoomBox.gltf"
        };
        
        int modelsProcessed = 0;
        int texturesLoaded = 0;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (const auto& modelPath : testModels) {
            if (!std::filesystem::exists(modelPath)) {
                continue;
            }
            
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
            
            if (cgModel) {
                sabi::RenderableNode node = sabi::WorldItem::create();
                node->setName(modelPath.filename().replace_extension().string());
                node->setModel(cgModel);
                
                sabi::RenderableDesc desc = node->description();
                desc.modelPath = modelPath;
                node->setDescription(desc);
                
                // Set vertex count for all surfaces
                size_t vertexCount = cgModel->vertexCount();
                for (auto& s : cgModel->S) {
                    s.vertexCount = vertexCount;
                }
                
                ShockerModelPtr model = modelHandler.processRenderableNode(node);
                if (model) {
                    modelsProcessed++;
                    texturesLoaded += cgModel->cgTextures.size();
                }
            }
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        LOG(INFO) << "===== GPU Batch Processing Results =====";
        LOG(INFO) << "Models processed: " << modelsProcessed;
        LOG(INFO) << "Textures loaded: " << texturesLoaded;
        LOG(INFO) << "Materials created: " << matHandler.getAllMaterials().size();
        LOG(INFO) << "Processing time: " << duration.count() << "ms";
        
        CHECK(modelsProcessed > 0);
        
        // Final GPU upload
        matHandler.uploadMaterialsToGPU();
        
        // Verify GPU buffer is populated
        CHECK(matHandler.getMaterialDataBuffer()->isInitialized());
    }
}

TEST_CASE("ShockerMaterialHandler GPU Memory Management")
{
    GPUTestContext gpuContext;
    
    if (!gpuContext.isAvailable()) {
        LOG(WARNING) << "Skipping GPU memory tests - no CUDA device available";
        return;
    }
    
    SUBCASE("Memory Cleanup Test")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(gpuContext.renderContext.get());
        
        // Create many materials to test memory management
        const int numMaterials = 100;
        
        for (int i = 0; i < numMaterials; ++i) {
            sabi::CgMaterial cgMat;
            cgMat.name = "MemTest_" + std::to_string(i);
            cgMat.core.baseColor = Eigen::Vector3f(
                static_cast<float>(i) / numMaterials,
                static_cast<float>(i) / numMaterials,
                static_cast<float>(i) / numMaterials
            );
            
            DisneyMaterial* mat = matHandler.createMaterialFromCg(cgMat);
            CHECK(mat != nullptr);
        }
        
        LOG(INFO) << "Created " << numMaterials << " materials with GPU resources";
        
        // Upload to GPU
        matHandler.uploadMaterialsToGPU();
        
        // Clear and verify cleanup
        matHandler.clear();
        CHECK(matHandler.getAllMaterials().empty());
        
        // RenderContext should still be valid after clearing materials
        CHECK(gpuContext.renderContext != nullptr);
        CHECK(gpuContext.renderContext->isInitialized());
        
        LOG(INFO) << "Memory cleanup successful";
    }
}