// ShockerModelHandlerTest.cpp
// Unit tests for ShockerModelHandler

#include <doctest/doctest.h>
#include "engine_core/excludeFromBuild/handlers/ShockerModelHandler.h"
#include "engine_core/excludeFromBuild/model/ShockerModel.h"

TEST_CASE("ShockerModelHandler Basic Operations")
{
    SUBCASE("Initialize and Clear")
    {
        ShockerModelHandler handler;
        
        // Initialize without render context (for basic testing)
        handler.initialize(nullptr);
        
        CHECK(handler.getAllModels().empty());
        CHECK(handler.getGeometryInstances().empty());
        CHECK(handler.getGeometryGroups().empty());
        
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
        CHECK(!model->getGeometryInstances().empty());  // Should have created geometry instances
        
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
    
    SUBCASE("Model Creates Its Own Geometry Instances")
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
        
        // Model should have created its own geometry instances
        const auto& geomInstances = model->getGeometryInstances();
        CHECK(!geomInstances.empty());
        CHECK(geomInstances.size() == cube->S.size());  // One per surface
        
        // Check first geometry instance
        const auto& firstInst = geomInstances[0];
        CHECK(firstInst != nullptr);
        CHECK(firstInst->geomInstSlot != SlotFinder::InvalidSlotIndex);
        CHECK(std::holds_alternative<TriangleGeometry>(firstInst->geometry));
        
        // Model should have a geometry group
        GeometryGroup* group = model->getGeometryGroup();
        CHECK(group != nullptr);
        CHECK(group->geomInsts.size() == geomInstances.size());
    }
    
    SUBCASE("Models Have Their Own Geometry Groups")
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
        
        // Each model should have its own geometry group
        for (const auto& model : models) {
            GeometryGroup* group = model->getGeometryGroup();
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
        CHECK(model->getGeometryGroup() != nullptr);  // Model must have geometry group
        
        // Create instance
        Instance* inst = handler.createInstance(model.get(), st);
        
        CHECK(inst != nullptr);
        CHECK(inst->instSlot != SlotFinder::InvalidSlotIndex);
        CHECK(inst->geomGroupInst.geomGroup == model->getGeometryGroup());
        CHECK(inst->matM2W.c3.x == doctest::Approx(10.0f));
        CHECK(inst->matM2W.c3.y == doctest::Approx(5.0f));
        CHECK(inst->matM2W.c3.z == doctest::Approx(2.0f));
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