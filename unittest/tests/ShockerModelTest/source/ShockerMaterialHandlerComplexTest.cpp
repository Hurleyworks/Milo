// ShockerMaterialHandlerComplexTest.cpp
// Complex tests for ShockerMaterialHandler with real-world GLTF models

#include <doctest/doctest.h>
#include <sabi_core/sabi_core.h>

#include "engine_core/excludeFromBuild/handlers/ShockerMaterialHandler.h"
#include "engine_core/excludeFromBuild/handlers/ShockerModelHandler.h"
#include "engine_core/excludeFromBuild/model/ShockerModel.h"

TEST_CASE("ShockerMaterialHandler Complex GLTF Models")
{
    SUBCASE("Process DamagedHelmet with PBR Materials")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        modelHandler.setMaterialHandler(&matHandler);
        
        std::filesystem::path modelPath = "E:/common_content/models/DamagedHelmet/glTF/DamagedHelmet.gltf";
        
        if (std::filesystem::exists(modelPath)) {
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
            
            if (cgModel) {
                LOG(INFO) << "Testing DamagedHelmet with " << cgModel->S.size() << " surfaces";
                
                // Create RenderableNode
                sabi::RenderableNode node = sabi::WorldItem::create();
                node->setName("DamagedHelmet");
                node->setModel(cgModel);
                
                // Set model path for texture loading
                sabi::RenderableDesc desc = node->description();
                desc.modelPath = modelPath;
                node->setDescription(desc);
                
                // Update surface vertex counts
                for (auto& s : cgModel->S) {
                    s.vertexCount = cgModel->vertexCount();
                }
                
                // Process through handlers
                ShockerModelPtr model = modelHandler.processRenderableNode(node);
                CHECK(model != nullptr);
                
                // Verify materials were created for each surface
                CHECK(matHandler.getAllMaterials().size() > cgModel->S.size()); // Should have default + surface materials
                
                // Check each geometry instance has a material
                for (const auto& geomInst : model->getGeometryInstances()) {
                    CHECK(geomInst->mat != nullptr);
                }
                
                // Log material statistics
                LOG(INFO) << "Created " << matHandler.getAllMaterials().size() << " materials";
                LOG(INFO) << "Model has " << model->getGeometryInstances().size() << " geometry instances";
                
                // Check material properties from GLTF
                for (size_t i = 0; i < cgModel->S.size(); ++i) {
                    const auto& surface = cgModel->S[i];
                    const auto& cgMat = surface.cgMaterial;
                    
                    LOG(DBUG) << "Surface " << i << " material: " << cgMat.name;
                    LOG(DBUG) << "  Base color: " << cgMat.core.baseColor.transpose();
                    LOG(DBUG) << "  Roughness: " << cgMat.core.roughness;
                    LOG(DBUG) << "  Metallic: " << cgMat.metallic.metallic;
                    
                    // Check for textures
                    if (cgMat.core.baseColorTexture) {
                        LOG(DBUG) << "  Has base color texture";
                    }
                    if (cgMat.core.roughnessTexture) {
                        LOG(DBUG) << "  Has roughness texture";
                    }
                    if (cgMat.metallic.metallicTexture) {
                        LOG(DBUG) << "  Has metallic texture";
                    }
                    if (cgMat.normalTexture) {
                        LOG(DBUG) << "  Has normal texture";
                    }
                }
            }
        } else {
            LOG(INFO) << "DamagedHelmet model not found, skipping test";
        }
    }
    
    SUBCASE("Process FlightHelmet with Complex Materials")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        modelHandler.setMaterialHandler(&matHandler);
        
        std::filesystem::path modelPath = "E:/common_content/models/FlightHelmet/glTF/FlightHelmet.gltf";
        
        if (std::filesystem::exists(modelPath)) {
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
            
            if (cgModel) {
                LOG(INFO) << "Testing FlightHelmet with " << cgModel->S.size() << " surfaces";
                LOG(INFO) << "Model has " << cgModel->cgTextures.size() << " textures";
                LOG(INFO) << "Model has " << cgModel->cgImages.size() << " images";
                
                sabi::RenderableNode node = sabi::WorldItem::create();
                node->setName("FlightHelmet");
                node->setModel(cgModel);
                
                sabi::RenderableDesc desc = node->description();
                desc.modelPath = modelPath;
                node->setDescription(desc);
                
                for (auto& s : cgModel->S) {
                    s.vertexCount = cgModel->vertexCount();
                }
                
                ShockerModelPtr model = modelHandler.processRenderableNode(node);
                CHECK(model != nullptr);
                
                // FlightHelmet typically has many surfaces with different materials
                CHECK(model->getGeometryInstances().size() == cgModel->S.size());
                
                // Verify each surface got a unique material
                size_t uniqueMaterials = matHandler.getAllMaterials().size() - 1; // Subtract default
                LOG(INFO) << "Created " << uniqueMaterials << " unique materials for " << cgModel->S.size() << " surfaces";
                
                // Check material variety
                int metallicCount = 0;
                int transparentCount = 0;
                int emissiveCount = 0;
                
                for (const auto& surface : cgModel->S) {
                    const auto& mat = surface.cgMaterial;
                    if (mat.metallic.metallic > 0.5f) metallicCount++;
                    if (mat.transparency.transparency > 0.0f) transparentCount++;
                    if (mat.emission.luminous > 0.0f) emissiveCount++;
                }
                
                LOG(INFO) << "Material variety - Metallic: " << metallicCount 
                          << ", Transparent: " << transparentCount
                          << ", Emissive: " << emissiveCount;
            }
        } else {
            LOG(INFO) << "FlightHelmet model not found, skipping test";
        }
    }
    
    SUBCASE("Process BMW Bike with Many Materials")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        modelHandler.setMaterialHandler(&matHandler);
        
        std::filesystem::path modelPath = "E:/common_content/models/bmw_bike/scene.gltf";
        
        if (std::filesystem::exists(modelPath)) {
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
            
            if (cgModel) {
                LOG(INFO) << "===== BMW Bike Material Test =====";
                LOG(INFO) << "Surfaces: " << cgModel->S.size();
                LOG(INFO) << "Textures: " << cgModel->cgTextures.size();
                LOG(INFO) << "Images: " << cgModel->cgImages.size();
                
                sabi::RenderableNode node = sabi::WorldItem::create();
                node->setName("BMW_Bike");
                node->setModel(cgModel);
                
                sabi::RenderableDesc desc = node->description();
                desc.modelPath = modelPath;
                node->setDescription(desc);
                
                for (auto& s : cgModel->S) {
                    s.vertexCount = cgModel->vertexCount();
                }
                
                auto startTime = std::chrono::high_resolution_clock::now();
                
                ShockerModelPtr model = modelHandler.processRenderableNode(node);
                
                auto endTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
                
                CHECK(model != nullptr);
                CHECK(model->getGeometryInstances().size() == cgModel->S.size());
                
                LOG(INFO) << "Processing time: " << duration.count() << "ms";
                LOG(INFO) << "Materials created: " << matHandler.getAllMaterials().size();
                
                // Analyze material complexity
                std::map<std::string, int> materialTypes;
                for (const auto& surface : cgModel->S) {
                    materialTypes[surface.cgMaterial.name]++;
                }
                
                LOG(INFO) << "Unique material names: " << materialTypes.size();
                
                // Check all geometry instances have materials
                int nullMaterialCount = 0;
                for (const auto& geomInst : model->getGeometryInstances()) {
                    if (geomInst->mat == nullptr) {
                        nullMaterialCount++;
                    }
                }
                CHECK(nullMaterialCount == 0);
                LOG(INFO) << "All " << model->getGeometryInstances().size() << " geometry instances have materials assigned";
            }
        } else {
            LOG(INFO) << "BMW bike model not found, skipping test";
        }
    }
    
    SUBCASE("Batch Process Multiple Complex Models")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        modelHandler.setMaterialHandler(&matHandler);
        
        std::vector<std::filesystem::path> testModels = {
            "E:/common_content/models/DamagedHelmet/glTF/DamagedHelmet.gltf",
            "E:/common_content/models/FlightHelmet/glTF/FlightHelmet.gltf",
            "E:/common_content/models/BarramundiFish/glTF/BarramundiFish.gltf",
            "E:/common_content/models/BoomBox/glTF/BoomBox.gltf",
            "E:/common_content/models/SciFiHelmet/glTF/SciFiHelmet.gltf",
            "E:/common_content/models/CesiumMan/glTF/CesiumMan.gltf",
            "E:/common_content/models/Duck/glTF/Duck.gltf",
            "E:/common_content/models/Avocado/glTF/Avocado.gltf"
        };
        
        LOG(INFO) << "===== Batch Material Processing Test =====";
        
        int totalSurfaces = 0;
        int totalMaterials = 1; // Start at 1 for default material
        int totalTextures = 0;
        int modelsProcessed = 0;
        
        auto batchStartTime = std::chrono::high_resolution_clock::now();
        
        for (const auto& modelPath : testModels) {
            if (!std::filesystem::exists(modelPath)) {
                continue;
            }
            
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
            
            if (cgModel) {
                std::string modelName = modelPath.filename().replace_extension().string();
                LOG(DBUG) << "Processing: " << modelName;
                
                sabi::RenderableNode node = sabi::WorldItem::create();
                node->setName(modelName);
                node->setModel(cgModel);
                
                sabi::RenderableDesc desc = node->description();
                desc.modelPath = modelPath;
                node->setDescription(desc);
                
                for (auto& s : cgModel->S) {
                    s.vertexCount = cgModel->vertexCount();
                }
                
                size_t materialsBefore = matHandler.getAllMaterials().size();
                
                ShockerModelPtr model = modelHandler.processRenderableNode(node);
                
                if (model) {
                    size_t materialsAfter = matHandler.getAllMaterials().size();
                    size_t newMaterials = materialsAfter - materialsBefore;
                    
                    totalSurfaces += cgModel->S.size();
                    totalMaterials += newMaterials;
                    totalTextures += cgModel->cgTextures.size();
                    modelsProcessed++;
                    
                    LOG(DBUG) << "  - Surfaces: " << cgModel->S.size() 
                              << ", New materials: " << newMaterials
                              << ", Textures: " << cgModel->cgTextures.size();
                }
            }
        }
        
        auto batchEndTime = std::chrono::high_resolution_clock::now();
        auto batchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(batchEndTime - batchStartTime);
        
        LOG(INFO) << "===== Batch Processing Summary =====";
        LOG(INFO) << "Models processed: " << modelsProcessed;
        LOG(INFO) << "Total surfaces: " << totalSurfaces;
        LOG(INFO) << "Total materials created: " << matHandler.getAllMaterials().size();
        LOG(INFO) << "Total textures: " << totalTextures;
        LOG(INFO) << "Total processing time: " << batchDuration.count() << "ms";
        LOG(INFO) << "Average time per model: " << (modelsProcessed > 0 ? batchDuration.count() / modelsProcessed : 0) << "ms";
        
        CHECK(modelsProcessed > 0);
        CHECK(matHandler.getAllMaterials().size() > 1); // Should have more than just default
    }
    
    SUBCASE("Material Memory and Performance Stress Test")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        modelHandler.setMaterialHandler(&matHandler);
        
        std::filesystem::path modelPath = "E:/common_content/models/DamagedHelmet/glTF/DamagedHelmet.gltf";
        
        if (std::filesystem::exists(modelPath)) {
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
            
            if (cgModel) {
                LOG(INFO) << "===== Material Stress Test =====";
                
                const int numIterations = 50;
                auto stressStartTime = std::chrono::high_resolution_clock::now();
                
                for (int i = 0; i < numIterations; ++i) {
                    sabi::RenderableNode node = sabi::WorldItem::create();
                    node->setName("StressTest_" + std::to_string(i));
                    node->setModel(cgModel);
                    
                    sabi::RenderableDesc desc = node->description();
                    desc.modelPath = modelPath;
                    node->setDescription(desc);
                    
                    for (auto& s : cgModel->S) {
                        s.vertexCount = cgModel->vertexCount();
                    }
                    
                    ShockerModelPtr model = modelHandler.processRenderableNode(node);
                    CHECK(model != nullptr);
                    
                    if (i % 10 == 0) {
                        LOG(DBUG) << "Iteration " << i << ": " << matHandler.getAllMaterials().size() << " materials";
                    }
                }
                
                auto stressEndTime = std::chrono::high_resolution_clock::now();
                auto stressDuration = std::chrono::duration_cast<std::chrono::milliseconds>(stressEndTime - stressStartTime);
                
                LOG(INFO) << "Created " << numIterations << " instances in " << stressDuration.count() << "ms";
                LOG(INFO) << "Average time per instance: " << stressDuration.count() / numIterations << "ms";
                LOG(INFO) << "Total materials: " << matHandler.getAllMaterials().size();
                LOG(INFO) << "Total models: " << modelHandler.getAllModels().size();
                
                // Materials should grow linearly since we removed caching
                CHECK(matHandler.getAllMaterials().size() > numIterations * cgModel->S.size());
                
                // Clear and verify cleanup
                matHandler.clear();
                CHECK(matHandler.getAllMaterials().empty());
            }
        } else {
            LOG(INFO) << "DamagedHelmet model not found, skipping stress test";
        }
    }
    
    SUBCASE("Material with Alpha Transparency")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        modelHandler.setMaterialHandler(&matHandler);
        
        // Try to find a model with transparency
        std::filesystem::path modelPath = "E:/common_content/models/CesiumMan/glTF/CesiumMan.gltf";
        
        if (std::filesystem::exists(modelPath)) {
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
            
            if (cgModel) {
                LOG(INFO) << "Testing transparency handling with CesiumMan";
                
                sabi::RenderableNode node = sabi::WorldItem::create();
                node->setName("CesiumMan");
                node->setModel(cgModel);
                
                sabi::RenderableDesc desc = node->description();
                desc.modelPath = modelPath;
                node->setDescription(desc);
                
                for (auto& s : cgModel->S) {
                    s.vertexCount = cgModel->vertexCount();
                }
                
                ShockerModelPtr model = modelHandler.processRenderableNode(node);
                CHECK(model != nullptr);
                
                // Check for materials with alpha
                int alphaCount = 0;
                for (const auto& surface : cgModel->S) {
                    if (surface.cgMaterial.flags.alphaMode == sabi::AlphaMode::Blend ||
                        surface.cgMaterial.flags.alphaMode == sabi::AlphaMode::Mask) {
                        alphaCount++;
                        LOG(DBUG) << "Found alpha material: " << surface.cgMaterial.name
                                  << " mode: " << static_cast<int>(surface.cgMaterial.flags.alphaMode);
                    }
                }
                
                if (alphaCount > 0) {
                    LOG(INFO) << "Found " << alphaCount << " materials with alpha";
                }
            }
        } else {
            LOG(INFO) << "CesiumMan model not found, skipping transparency test";
        }
    }
}