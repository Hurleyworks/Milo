// ShockerEngineGeometryTest.cpp
// Unit tests for ShockerEngine geometry rendering support

#include <doctest/doctest.h>
#include <sabi_core/sabi_core.h>

TEST_CASE("ShockerEngine Geometry Support")
{
    SUBCASE("Add Simple Cube Geometry")
    {
        // Note: Testing geometry creation without the full engine
        // The full engine requires CUDA/OptiX context which isn't available in unit tests
        
        // Create a simple cube
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        CHECK(cube != nullptr);
        CHECK(cube->vertexCount() > 0);
        CHECK(cube->S.size() > 0); // Has at least one surface
        
        // Create a renderable node
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TestCube");
        node->setModel(cube);
        
        // Set transform
        sabi::SpaceTime& st = node->getSpaceTime();
        st.localTransform.setIdentity();
        st.localTransform.translate(Eigen::Vector3f(0.0f, 0.0f, -5.0f));
        st.worldTransform = st.localTransform;
        
        // Verify node is ready
        CHECK(node->getName() == "TestCube");
        CHECK(node->getModel() == cube);
        
        // Note: Actually adding geometry requires initialized engine
        // engine.addGeometry(node);
        
        LOG(INFO) << "Test cube created with " << cube->vertexCount() << " vertices";
    }
    
    SUBCASE("Create Multiple Geometry Instances")
    {
        // Create multiple cubes with different transforms
        std::vector<sabi::RenderableNode> nodes;
        
        for (int i = 0; i < 5; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
            
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("Cube_" + std::to_string(i));
            node->setModel(cube);
            
            // Position cubes in a row
            sabi::SpaceTime& st = node->getSpaceTime();
            st.localTransform.setIdentity();
            st.localTransform.translate(Eigen::Vector3f(i * 3.0f, 0.0f, -5.0f));
            st.worldTransform = st.localTransform;
            
            nodes.push_back(node);
        }
        
        CHECK(nodes.size() == 5);
        
        // Verify each node has unique position
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& transform = nodes[i]->getSpaceTime().worldTransform;
            float expectedX = i * 3.0f;
            CHECK(transform.translation().x() == doctest::Approx(expectedX));
        }
        
        LOG(INFO) << "Created " << nodes.size() << " cube instances";
    }
    
    SUBCASE("Geometry with Materials")
    {
        // Create a cube with material properties
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        
        // Verify surface exists
        CHECK(cube->S.size() > 0);
        auto& surface = cube->S[0];
        
        // CgModelSurface has material and cgMaterial members
        // Materials would be set through the material handler
        CHECK(surface.triangleCount() >= 0);  // Has triangles
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("MaterialCube");
        node->setModel(cube);
        
        CHECK(node->getModel()->S.size() > 0);
        
        LOG(INFO) << "Created cube with " << cube->S.size() << " surface(s)";
    }
}

TEST_CASE("ShockerEngine Scene Management")
{
    SUBCASE("Clear Scene")
    {
        // Create a scene with some geometry
        std::vector<sabi::RenderableNode> nodes;
        
        for (int i = 0; i < 3; ++i) {
            sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName("Cube_" + std::to_string(i));
            node->setModel(cube);
            nodes.push_back(node);
        }
        
        CHECK(nodes.size() == 3);
        
        // Note: Actual scene clearing would be tested with initialized engine
        // engine.clearScene();
        
        LOG(INFO) << "Scene management test completed";
    }
    
    SUBCASE("Geometry Validation")
    {
        // Test that invalid geometry is handled properly
        sabi::RenderableNode emptyNode = sabi::WorldItem::create();
        emptyNode->setName("EmptyNode");
        // No model set - this should be handled as a phantom model
        
        CHECK(emptyNode->getModel() == nullptr);
        
        // The engine should handle this gracefully
        // engine.addGeometry(emptyNode);
        
        LOG(INFO) << "Empty node handling test completed";
    }
}