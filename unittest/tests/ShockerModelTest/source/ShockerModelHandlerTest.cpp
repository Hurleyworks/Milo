// ShockerModelHandlerTest.cpp
// Unit tests for ShockerModelHandler

#include <doctest/doctest.h>
#include <sabi_core/sabi_core.h>
#include <filesystem>
#include <chrono>

#include "engine_core/excludeFromBuild/engines/shocker/handlers/ShockerModelHandler.h"
#include "engine_core/excludeFromBuild/engines/shocker/models/ShockerModel.h"
#include "engine_core/excludeFromBuild/engines/shocker/models/ShockerCore.h"

static void loadGLTF (const std::filesystem::path& gltfPath)
{
  

    GLTFImporter gltf;
  //  std::vector<Animation> animations;
    auto [cgModel, animations] = gltf.importModel (gltfPath.generic_string());
    if (!cgModel)
    {
        LOG (WARNING) << "Load failed " << gltfPath.string();
        return;
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
}

TEST_CASE("ShockerModelHandler Basic Operations")
{
    SUBCASE("Initialize and Clear")
    {
        ShockerModelHandler handler;
        
        // Initialize without render context (for basic testing)
        handler.initialize(nullptr);
        
        CHECK(handler.getAllModels().empty());
        CHECK(handler.getShockerSurfaceCount() == 0);
        CHECK(handler.getShockerSurfaceGroups().empty());
        
        handler.clear();
        CHECK(handler.getAllModels().empty());
    }
    
    SUBCASE("Process RenderableNode")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Create a cube model
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        
        // Create a RenderableNode
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TestCube");
        node->setModel(cube);
        
        // Process the node
        ShockerModelPtr model = handler.processRenderableNode(node);
        
        CHECK(model != nullptr);
        CHECK(model->getGeometryType() == ShockerGeometryType::Triangle);
        CHECK(!model->getSurfaces().empty());  // Should have created surfaces
        
        // Check model is stored
        CHECK(handler.hasModel("TestCube"));
        CHECK(handler.getModel("TestCube") == model);
        CHECK(handler.getAllModels().size() == 1);
    }
    
    SUBCASE("Create Model By Type")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Test with triangle mesh
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
        ShockerModelPtr model = handler.createModelByType(cube);
        
        CHECK(model != nullptr);
        CHECK(model->getGeometryType() == ShockerGeometryType::Triangle);
        
        // Test with empty model (should create flyweight)
        sabi::CgModelPtr empty = sabi::CgModel::create();
        model = handler.createModelByType(empty);
        
        CHECK(model != nullptr);
        CHECK(model->getGeometryType() == ShockerGeometryType::Flyweight);
        
        // Test with null (should create phantom)
        model = handler.createModelByType(nullptr);
        
        CHECK(model != nullptr);
        CHECK(model->getGeometryType() == ShockerGeometryType::Phantom);
    }
    
    SUBCASE("Model Creates Its Own Surfaces")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Create and process a model
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TestCube");
        node->setModel(cube);
        
        ShockerModelPtr model = handler.processRenderableNode(node);
        CHECK(model != nullptr);
        
        // Model should have created its own surfaces
        const auto& surfaces = model->getSurfaces();
        CHECK(!surfaces.empty());
        CHECK(surfaces.size() == cube->S.size());  // One per surface
        
        // Check first surface
        const auto& firstSurface = surfaces[0];
        CHECK(firstSurface != nullptr);
        CHECK(firstSurface->geomInstSlot != SlotFinder::InvalidSlotIndex);
        CHECK(std::holds_alternative<TriangleGeometry>(firstSurface->geometry));
        
        // Model should have a surface group
        shocker::ShockerSurfaceGroup* group = model->getSurfaceGroup();
        CHECK(group != nullptr);
        CHECK(group->geomInsts.size() == surfaces.size());
    }
    
    SUBCASE("Models Have Their Own Surface Groups")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Create multiple models
        std::vector<ShockerModelPtr> models;
        
        for (int i = 0; i < 3; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("Cube_" + std::to_string(i));
            node->setModel(cube);
            
            ShockerModelPtr model = handler.processRenderableNode(node);
            models.push_back(model);
        }
        
        // Each model should have its own surface group
        for (const auto& model : models) {
            shocker::ShockerSurfaceGroup* group = model->getSurfaceGroup();
            CHECK(group != nullptr);
            CHECK(!group->geomInsts.empty());
            CHECK(group->needsRebuild == 1);
            
            // Verify AABB is calculated
            CHECK(group->aabb.minP.x != FLT_MAX);
            CHECK(group->aabb.maxP.x != -FLT_MAX);
        }
    }
    
    SUBCASE("Create Instance with Transform")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Create model
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TestCube");
        node->setModel(cube);
        
        // Set transform
        sabi::SpaceTime& st = node->getSpaceTime();
        st.localTransform.setIdentity();
        st.localTransform.translate(Eigen::Vector3f(10.0f, 5.0f, 2.0f));
        st.localTransform.scale(2.0f);
        st.worldTransform = st.localTransform;
        
        ShockerModelPtr model = handler.processRenderableNode(node);
        CHECK(model != nullptr);
        CHECK(model->getSurfaceGroup() != nullptr);  // Model must have surface group
        
        // Create node
        shocker::ShockerNode* shockerNode = handler.createShockerNode(model.get(), st);
        
        CHECK(shockerNode != nullptr);
        CHECK(shockerNode->instSlot != SlotFinder::InvalidSlotIndex);
        CHECK(shockerNode->geomGroupInst.geomGroup == model->getSurfaceGroup());
        CHECK(shockerNode->matM2W.c3.x == doctest::Approx(10.0f));
        CHECK(shockerNode->matM2W.c3.y == doctest::Approx(5.0f));
        CHECK(shockerNode->matM2W.c3.z == doctest::Approx(2.0f));
    }
    
    SUBCASE("Multiple Models Management")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Create multiple models
        for (int i = 0; i < 5; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f + i * 0.5f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            std::string name = "Model_" + std::to_string(i);
            node->setName(name);
            node->setModel(cube);
            
            handler.processRenderableNode(node);
        }
        
        CHECK(handler.getAllModels().size() == 5);
        CHECK(handler.hasModel("Model_0"));
        CHECK(handler.hasModel("Model_4"));
        CHECK(!handler.hasModel("Model_5"));
        
        // Clear and verify
        handler.clear();
        CHECK(handler.getAllModels().empty());
        CHECK(!handler.hasModel("Model_0"));
    }
}

TEST_CASE("ShockerModelHandler Real-World GLTF Models")
{
    SUBCASE("Load and Process GLTF Models")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Path to test models
        std::filesystem::path modelsPath = "E:/common_content/models";
        
        // List of GLTF models to test - these actually exist in the directory
        std::vector<std::string> testModels = {
            "DamagedHelmet/glTF/DamagedHelmet.gltf",
            "FlightHelmet/glTF/FlightHelmet.gltf",
            "BarramundiFish/glTF/BarramundiFish.gltf",
            "BoomBox/glTF/BoomBox.gltf",
            "SciFiHelmet/glTF/SciFiHelmet.gltf"
        };
        
        // Track statistics
        int modelsLoaded = 0;
        int modelsFailed = 0;
        size_t totalSurfaces = 0;
        size_t totalShockerSurfaces = 0;
        
        for (const auto& modelFile : testModels) {
            std::filesystem::path fullPath = modelsPath / modelFile;
            
            // Skip if file doesn't exist
            if (!std::filesystem::exists(fullPath)) {
                LOG(DBUG) << "Skipping non-existent model: " << fullPath.string();
                continue;
            }
            
            LOG(DBUG) << "Testing GLTF model: " << modelFile;
            
            // Load GLTF
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(fullPath.generic_string());
            
            if (!cgModel) {
                LOG(WARNING) << "Failed to load GLTF: " << fullPath.string();
                modelsFailed++;
                continue;
            }
            
            // Create RenderableNode
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName(modelFile);
            node->setModel(cgModel);
            node->getState().state |= sabi::PRenderableState::Visible;
            
            // Set model path for texture loading
            sabi::RenderableDesc desc = node->description();
            desc.modelPath = fullPath;
            node->setDescription(desc);
            
            // Update surface vertex counts
            for (auto& s : cgModel->S) {
                s.vertexCount = cgModel->vertexCount();
            }
            
            // Process through ShockerModelHandler
            ShockerModelPtr model = handler.processRenderableNode(node);
            
            if (model) {
                modelsLoaded++;
                
                // Verify model was created correctly
                CHECK(model->getGeometryType() == ShockerGeometryType::Triangle);
                CHECK(model->getSurfaceGroup() != nullptr);
                
                // Check surfaces
                const auto& surfaces = model->getSurfaces();
                CHECK(!surfaces.empty());
                CHECK(surfaces.size() == cgModel->S.size()); // One per surface
                
                // Track statistics
                totalSurfaces += cgModel->S.size();
                totalShockerSurfaces += surfaces.size();
                
                // Verify AABB is valid
                const AABB& aabb = model->getAABB();
                CHECK(std::isfinite(aabb.minP.x));
                CHECK(std::isfinite(aabb.maxP.x));
                CHECK(aabb.minP.x <= aabb.maxP.x);
                
                // Log model info
                LOG(DBUG) << "  - Vertices: " << cgModel->vertexCount();
                LOG(DBUG) << "  - Surfaces: " << cgModel->S.size();
                LOG(DBUG) << "  - ShockerSurfaces: " << surfaces.size();
                LOG(DBUG) << "  - AABB: (" << aabb.minP.x << ", " << aabb.minP.y << ", " << aabb.minP.z 
                         << ") to (" << aabb.maxP.x << ", " << aabb.maxP.y << ", " << aabb.maxP.z << ")";
            } else {
                modelsFailed++;
                LOG(WARNING) << "Failed to process model: " << modelFile;
            }
        }
        
        // Summary
        LOG(INFO) << "===== GLTF Test Summary =====";
        LOG(INFO) << "Models loaded: " << modelsLoaded;
        LOG(INFO) << "Models failed: " << modelsFailed;
        LOG(INFO) << "Total surfaces: " << totalSurfaces;
        LOG(INFO) << "Total Shocker surfaces: " << totalShockerSurfaces;
        LOG(INFO) << "Models in handler: " << handler.getAllModels().size();
        
        // Basic assertions
        CHECK(modelsLoaded > 0); // At least one model should load
        CHECK(handler.getAllModels().size() == modelsLoaded);
    }
    
    SUBCASE("Test Complex Model with Multiple Surfaces")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Try to load a complex model like bmw_bike which has many surfaces
        std::filesystem::path complexPath = "E:/common_content/models/bmw_bike/scene.gltf";
        
        if (std::filesystem::exists(complexPath)) {
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(complexPath.generic_string());
            
            if (cgModel) {
                LOG(INFO) << "Testing BMW bike model with " << cgModel->S.size() << " surfaces";
                
                sabi::RenderableNode node = sabi::WorldItem::create();
                node->setName("BMW_Bike");
                node->setModel(cgModel);
                
                for (auto& s : cgModel->S) {
                    s.vertexCount = cgModel->vertexCount();
                }
                
                ShockerModelPtr model = handler.processRenderableNode(node);
                CHECK(model != nullptr);
                
                // BMW bike has many surfaces
                CHECK(model->getSurfaces().size() == cgModel->S.size());
                CHECK(model->getSurfaces().size() > 1); // Should have multiple surfaces
                
                // Each surface should have valid data
                for (const auto& surface : model->getSurfaces()) {
                    CHECK(surface != nullptr);
                    CHECK(surface->geomInstSlot != SlotFinder::InvalidSlotIndex);
                    CHECK(std::holds_alternative<TriangleGeometry>(surface->geometry));
                    
                    // Check AABB is valid
                    CHECK(std::isfinite(surface->aabb.minP.x));
                    CHECK(std::isfinite(surface->aabb.maxP.x));
                }
            }
        } else {
            LOG(INFO) << "BMW bike model not found, skipping complex model test";
        }
    }
    
    SUBCASE("Memory and Performance Test")
    {
        ShockerModelHandler handler;
        handler.initialize(nullptr);
        
        // Load multiple instances of the same model
        std::filesystem::path modelPath = "E:/common_content/models/DamagedHelmet/glTF/DamagedHelmet.gltf";
        
        if (std::filesystem::exists(modelPath)) {
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel(modelPath.generic_string());
            
            if (cgModel) {
                const int numInstances = 10;
                auto startTime = std::chrono::high_resolution_clock::now();
                
                for (int i = 0; i < numInstances; ++i) {
                    sabi::RenderableNode node = sabi::WorldItem::create();
                    node->setName("DamagedHelmet_" + std::to_string(i));
                    node->setModel(cgModel);
                    
                    for (auto& s : cgModel->S) {
                        s.vertexCount = cgModel->vertexCount();
                    }
                    
                    ShockerModelPtr model = handler.processRenderableNode(node);
                    CHECK(model != nullptr);
                }
                
                auto endTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
                
                LOG(INFO) << "Created " << numInstances << " model instances in " << duration.count() << "ms";
                CHECK(handler.getAllModels().size() == numInstances);
                
                // Clear and check cleanup
                handler.clear();
                CHECK(handler.getAllModels().empty());
            }
        }
    }
}