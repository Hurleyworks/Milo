// ShockerSceneHandlerTest.cpp
// Unit tests for ShockerSceneHandler

#include <doctest/doctest.h>
#include <sabi_core/sabi_core.h>

#include "engine_core/excludeFromBuild/handlers/ShockerSceneHandler.h"
#include "engine_core/excludeFromBuild/handlers/ShockerModelHandler.h"
#include "engine_core/excludeFromBuild/handlers/ShockerMaterialHandler.h"
#include "engine_core/excludeFromBuild/model/ShockerModel.h"
#include "engine_core/excludeFromBuild/model/ShockerCore.h"

TEST_CASE("ShockerSceneHandler Basic Operations")
{
    SUBCASE("Initialize and Clear")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        CHECK(sceneHandler != nullptr);
        
        sceneHandler->initialize();
        CHECK(sceneHandler->getNodeCount() == 0);
        CHECK(sceneHandler->getSurfaceCount() == 0);
        CHECK(sceneHandler->getMaterialCount() == 0);
        
        sceneHandler->clear();
        CHECK(sceneHandler->getNodeCount() == 0);
    }
    
    SUBCASE("Set and Get Handlers")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        CHECK(sceneHandler->getModelHandler() == modelHandler);
        CHECK(sceneHandler->getMaterialHandler() == materialHandler);
    }
    
    SUBCASE("Process Simple Renderable Node")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        // Set up handlers
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        materialHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        // Create a simple cube
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        
        // Create a renderable node
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TestCube");
        node->setModel(cube);
        
        // Process the node
        sceneHandler->processRenderableNode(node);
        
        // Check that node was created
        CHECK(sceneHandler->getNodeCount() == 1);
        CHECK(modelHandler->getAllModels().size() == 1);
        CHECK(materialHandler->getAllMaterials().size() > 0); // At least default material
    }
}

TEST_CASE("ShockerSceneHandler Instance Creation")
{
    SUBCASE("Create Single Instance")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        materialHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        // Create a cube model
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("SingleCube");
        node->setModel(cube);
        
        // Create weak reference
        sabi::RenderableWeakRef weakNode = node;
        
        // Create Shocker node
        shocker::ShockerNode* shockerNode = sceneHandler->createShockerNode(weakNode);
        
        CHECK(shockerNode != nullptr);
        CHECK(shockerNode->instSlot != SlotFinder::InvalidSlotIndex);
        CHECK(sceneHandler->getNodeCount() == 1);
        
        // Check node mapping
        auto retrievedNode = sceneHandler->getRenderableNode(shockerNode->instSlot);
        CHECK(!retrievedNode.expired());
        CHECK(retrievedNode.lock() == node);
    }
    
    SUBCASE("Create Multiple Instances")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        materialHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        // Create multiple nodes
        const int numNodes = 5;
        std::vector<sabi::RenderableNode> nodes;
        
        for (int i = 0; i < numNodes; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f + i * 0.5f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("Cube_" + std::to_string(i));
            node->setModel(cube);
            nodes.push_back(node);
        }
        
        // Process all nodes
        for (auto& node : nodes) {
            sceneHandler->processRenderableNode(node);
        }
        
        CHECK(sceneHandler->getNodeCount() == numNodes);
        CHECK(modelHandler->getAllModels().size() == numNodes);
        
        // Check each node
        for (size_t i = 0; i < numNodes; ++i) {
            shocker::ShockerNode* node = sceneHandler->getShockerNode(i);
            CHECK(node != nullptr);
            CHECK(node->instSlot != SlotFinder::InvalidSlotIndex);
        }
    }
    
    SUBCASE("Create Instance List")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        materialHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        // Create nodes first, then create weak references
        sabi::RenderableNode node0 = sabi::WorldItem::create();
        node0->setName("ListCube_0");
        node0->setModel(sabi::MeshOps::createCube(2.0f));
        
        sabi::RenderableNode node1 = sabi::WorldItem::create();
        node1->setName("ListCube_1");
        node1->setModel(sabi::MeshOps::createCube(2.0f));
        
        sabi::RenderableNode node2 = sabi::WorldItem::create();
        node2->setName("ListCube_2");
        node2->setModel(sabi::MeshOps::createCube(2.0f));
        
        // Create weak reference list
        sabi::WeakRenderableList weakNodeList;
        weakNodeList.push_back(node0);
        weakNodeList.push_back(node1);
        weakNodeList.push_back(node2);
        
        // Verify nodes are still alive before calling createInstanceList
        for (const auto& weakNode : weakNodeList) {
            CHECK(!weakNode.expired());
        }
        
        // Create nodes from list
        sceneHandler->createNodeList(weakNodeList);
        
        CHECK(sceneHandler->getNodeCount() == 3);
        CHECK(modelHandler->getAllModels().size() == 3);
        
        // Keep nodes alive until end of test
        // node0, node1, node2 will be destroyed here at end of scope
    }
}

TEST_CASE("ShockerSceneHandler Transform and Materials")
{
    SUBCASE("Instance with Transform")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        materialHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        // Create node with transform
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TransformedCube");
        node->setModel(cube);
        
        // Set transform
        sabi::SpaceTime& st = node->getSpaceTime();
        st.localTransform.setIdentity();
        st.localTransform.translate(Eigen::Vector3f(10.0f, 5.0f, 2.0f));
        st.localTransform.scale(2.0f);
        st.worldTransform = st.localTransform;
        
        // Process node
        sceneHandler->processRenderableNode(node);
        
        CHECK(sceneHandler->getNodeCount() == 1);
        
        shocker::ShockerNode* shockerNode = sceneHandler->getShockerNode(0);
        CHECK(shockerNode != nullptr);
        
        // Check transform was applied
        CHECK(shockerNode->matM2W.c3.x == doctest::Approx(10.0f));
        CHECK(shockerNode->matM2W.c3.y == doctest::Approx(5.0f));
        CHECK(shockerNode->matM2W.c3.z == doctest::Approx(2.0f));
    }
    
    SUBCASE("Instance with Materials")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        materialHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        // Create cube with material
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        
        // Set material properties
        if (!cube->S.empty()) {
            cube->S[0].cgMaterial.name = "RedMaterial";
            cube->S[0].cgMaterial.core.baseColor = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
            cube->S[0].cgMaterial.core.roughness = 0.5f;
            cube->S[0].cgMaterial.metallic.metallic = 0.3f;
        }
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("MaterialCube");
        node->setModel(cube);
        
        // Process node
        sceneHandler->processRenderableNode(node);
        
        CHECK(sceneHandler->getNodeCount() == 1);
        CHECK(sceneHandler->getMaterialCount() > 1); // Default + surface material
        
        // Get the model and check materials were assigned
        auto models = modelHandler->getAllModels();
        CHECK(models.size() == 1);
        
        ShockerModelPtr model = models.begin()->second;
        for (const auto& surface : model->getSurfaces()) {
            CHECK(surface->mat != nullptr);
        }
    }
}

TEST_CASE("ShockerSceneHandler Statistics and Queries")
{
    SUBCASE("Get Statistics")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        materialHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        // Keep nodes alive to avoid destructor logs
        std::vector<sabi::RenderableNode> nodes;
        
        // Process multiple nodes
        for (int i = 0; i < 10; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("StatsCube_" + std::to_string(i));
            node->setModel(cube);
            nodes.push_back(node);  // Keep alive
            sceneHandler->processRenderableNode(node);
        }
        
        CHECK(sceneHandler->getNodeCount() == 10);
        
        // Add logging to debug the surface count issue
        LOG(INFO) << "Scene statistics:";
        LOG(INFO) << "  Nodes: " << sceneHandler->getNodeCount();
        LOG(INFO) << "  Surfaces: " << sceneHandler->getSurfaceCount();
        LOG(INFO) << "  Materials: " << sceneHandler->getMaterialCount();
        
        // Basic statistics check
        CHECK(modelHandler->getAllModels().size() == 10);
        CHECK(materialHandler->getAllMaterials().size() > 0);
        CHECK(sceneHandler->getSurfaceCount() > 0);  // Should have surfaces
    }
    
    SUBCASE("Get Instance by Index")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        
        // Keep nodes alive
        std::vector<sabi::RenderableNode> nodes;
        
        // Create nodes
        for (int i = 0; i < 5; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("IndexCube_" + std::to_string(i));
            node->setModel(cube);
            nodes.push_back(node);  // Keep alive
            sceneHandler->processRenderableNode(node);
        }
        
        // Test valid indices
        for (uint32_t i = 0; i < 5; ++i) {
            shocker::ShockerNode* node = sceneHandler->getShockerNode(i);
            CHECK(node != nullptr);
        }
        
        // Test invalid index
        shocker::ShockerNode* invalid = sceneHandler->getShockerNode(100);
        CHECK(invalid == nullptr);
    }
    
    SUBCASE("Node Retrieval by Instance Slot")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        
        // Create and store nodes
        std::vector<sabi::RenderableNode> originalNodes;
        
        for (int i = 0; i < 3; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("RetrievalCube_" + std::to_string(i));
            node->setModel(cube);
            originalNodes.push_back(node);
            sceneHandler->processRenderableNode(node);
        }
        
        // Retrieve nodes by node slot
        for (uint32_t i = 0; i < 3; ++i) {
            shocker::ShockerNode* node = sceneHandler->getShockerNode(i);
            CHECK(node != nullptr);
            
            auto retrievedWeakNode = sceneHandler->getRenderableNode(node->instSlot);
            CHECK(!retrievedWeakNode.expired());
            
            auto retrievedNode = retrievedWeakNode.lock();
            CHECK(retrievedNode == originalNodes[i]);
            CHECK(retrievedNode->getName() == originalNodes[i]->getName());
        }
    }
}

TEST_CASE("ShockerSceneHandler Edge Cases")
{
    SUBCASE("Process Node without Model")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        
        // Create node without model
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("EmptyNode");
        // No model set
        
        sceneHandler->processRenderableNode(node);
        
        // Should still create an instance (phantom)
        CHECK(sceneHandler->getNodeCount() == 1);
    }
    
    SUBCASE("Process without Material Handler")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        // No material handler set
        
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("NoMaterialHandlerCube");
        node->setModel(cube);
        
        sceneHandler->processRenderableNode(node);
        
        // Should still create instance without materials
        CHECK(sceneHandler->getNodeCount() == 1);
        CHECK(sceneHandler->getMaterialCount() == 0); // No material handler
    }
    
    SUBCASE("Clear and Rebuild")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        auto materialHandler = std::make_shared<ShockerMaterialHandler>();
        materialHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        sceneHandler->setMaterialHandler(materialHandler);
        
        // Create some instances
        for (int i = 0; i < 5; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("ClearTestCube_" + std::to_string(i));
            node->setModel(cube);
            sceneHandler->processRenderableNode(node);
        }
        
        CHECK(sceneHandler->getNodeCount() == 5);
        
        // Clear everything
        sceneHandler->clear();
        
        CHECK(sceneHandler->getNodeCount() == 0);
        CHECK(modelHandler->getAllModels().size() == 0);
        CHECK(materialHandler->getAllMaterials().size() == 0);
        
        // Rebuild with new instances
        for (int i = 0; i < 3; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("RebuildCube_" + std::to_string(i));
            node->setModel(cube);
            sceneHandler->processRenderableNode(node);
        }
        
        CHECK(sceneHandler->getNodeCount() == 3);
    }
}

TEST_CASE("ShockerSceneHandler Acceleration Structures")
{
    SUBCASE("Build Acceleration Structures")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        
        // Create instances
        for (int i = 0; i < 10; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("AccelCube_" + std::to_string(i));
            node->setModel(cube);
            sceneHandler->processRenderableNode(node);
        }
        
        // Build acceleration structures
        sceneHandler->buildAccelerationStructures();
        
        // Check that models have their needsRebuild flag cleared
        for (const auto& [name, model] : modelHandler->getAllModels()) {
            shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
            if (surfaceGroup) {
                CHECK(surfaceGroup->needsRebuild == 0);
            }
        }
    }
    
    SUBCASE("Update Acceleration Structures")
    {
        auto sceneHandler = ShockerSceneHandler::create(nullptr);
        sceneHandler->initialize();
        
        auto modelHandler = std::make_shared<ShockerModelHandler>();
        modelHandler->initialize(nullptr);
        
        sceneHandler->setModelHandler(modelHandler);
        
        // Create instances
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("UpdateCube");
        node->setModel(cube);
        sceneHandler->processRenderableNode(node);
        
        // Update acceleration structures (should handle refittable geometry)
        sceneHandler->updateAccelerationStructures();
        
        // This is mainly a placeholder test until OptiX integration
        CHECK(true);
    }
}