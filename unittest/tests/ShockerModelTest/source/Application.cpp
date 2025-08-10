#include "Jahley.h"

const std::string APP_NAME = "ShockerModelTest";

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

// Include ShockerModel
#include "engine_core/excludeFromBuild/engines/shocker/models/ShockerModel.h"
#include "engine_core/excludeFromBuild/engines/shocker/models/ShockerCore.h"

// Include common_host for geometry structures  
#include "engine_core/excludeFromBuild/common/common_host.h"

TEST_CASE("ShockerModel Base Class")
{
    SUBCASE("Transform Conversion from SpaceTime")
    {
        // Create a test SpaceTime
        sabi::SpaceTime spacetime;
        spacetime.localTransform.setIdentity();
        spacetime.localTransform.translate(Eigen::Vector3f(5.0f, 3.0f, 2.0f));
        spacetime.localTransform.scale(1.5f);
        spacetime.worldTransform = spacetime.localTransform;
        
        // Create a ShockerTriangleModel to test base class methods
        auto model = ShockerTriangleModel::create();
        
        // Test convertSpaceTimeToMatrix (static method)
        Matrix4x4 mat = ShockerModel::convertSpaceTimeToMatrix(spacetime);
        
        // Test that translation is correct
        CHECK(mat.c3.x == doctest::Approx(5.0f));
        CHECK(mat.c3.y == doctest::Approx(3.0f));
        CHECK(mat.c3.z == doctest::Approx(2.0f));
        
        // Test that scale is correct (diagonal elements)
        CHECK(mat.c0.x == doctest::Approx(1.5f));
        CHECK(mat.c1.y == doctest::Approx(1.5f));
        CHECK(mat.c2.z == doctest::Approx(1.5f));
    }
    
    SUBCASE("Normal Matrix Calculation")
    {
        auto model = ShockerTriangleModel::create();
        
        // Create a transform with non-uniform scaling
        Matrix4x4 transform;
        transform.c0.x = 2.0f;  // Scale X by 2
        transform.c1.y = 3.0f;  // Scale Y by 3
        transform.c2.z = 1.0f;  // Scale Z by 1
        transform.c3.w = 1.0f;
        
        // Calculate normal matrix (static method)
        Matrix3x3 normalMat = ShockerModel::calculateNormalMatrix(transform);
        
        // For non-uniform scaling, normal matrix should be inverse transpose
        // For our diagonal scale matrix, inverse is 1/scale
        CHECK(normalMat.m00 == doctest::Approx(0.5f));  // 1/2
        CHECK(normalMat.m11 == doctest::Approx(0.333333f));  // 1/3
        CHECK(normalMat.m22 == doctest::Approx(1.0f));  // 1/1
    }
    
    SUBCASE("Static Transform Methods")
    {
        // Test that transform methods are static and work correctly
        sabi::SpaceTime st;
        st.localTransform.setIdentity();
        st.localTransform.translate(Eigen::Vector3f(1.0f, 2.0f, 3.0f));
        st.worldTransform = st.localTransform;
        
        Matrix4x4 mat = ShockerModel::convertSpaceTimeToMatrix(st);
        CHECK(mat.c3.x == doctest::Approx(1.0f));
        CHECK(mat.c3.y == doctest::Approx(2.0f));
        CHECK(mat.c3.z == doctest::Approx(3.0f));
    }
}

TEST_CASE("ShockerTriangleModel")
{
    SUBCASE("Create from Cube RenderableNode")
    {
        // Create a cube using MeshOps
        float cubeSize = 2.0f;
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(cubeSize);
        
        // Create a RenderableNode
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TestCube");
        node->setModel(cube);
        
        // Set transform
        sabi::SpaceTime& spacetime = node->getSpaceTime();
        spacetime.localTransform.setIdentity();
        spacetime.localTransform.translate(Eigen::Vector3f(10.0f, 5.0f, 0.0f));
        spacetime.worldTransform = spacetime.localTransform;
        
        // Create ShockerTriangleModel
        auto model = std::static_pointer_cast<ShockerTriangleModel>(ShockerTriangleModel::create());
        SlotFinder slotFinder;
        slotFinder.initialize(100);
        model->createFromRenderableNode(node, slotFinder, nullptr);
        
        // Verify geometry extraction
        CHECK(model->getGeometryType() == ShockerGeometryType::Triangle);
        
        // Model should have created geometry instances for each surface
        const auto& surfaces = model->getSurfaces();
        CHECK(surfaces.size() == cube->S.size());  // One geometry instance per surface
        
        // Model should have a geometry group
        shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
        CHECK(surfaceGroup != nullptr);
        CHECK(surfaceGroup->geomInsts.size() == surfaces.size());
        
        // Verify AABB calculation
        const AABB& aabb = model->getAABB();
        CHECK(aabb.minP.x == doctest::Approx(-cubeSize/2));
        CHECK(aabb.maxP.x == doctest::Approx(cubeSize/2));
    }
    
    SUBCASE("Geometry Instance Creation")
    {
        // Create a simple triangle mesh
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setModel(cube);
        
        auto model = std::static_pointer_cast<ShockerTriangleModel>(ShockerTriangleModel::create());
        SlotFinder slotFinder;
        slotFinder.initialize(100);
        model->createFromRenderableNode(node, slotFinder, nullptr);
        
        // Check geometry instances were created
        const auto& surfaces = model->getSurfaces();
        CHECK(surfaces.size() > 0);
        
        // Verify first geometry instance has valid slot
        const auto& firstSurface = surfaces[0];
        CHECK(firstSurface != nullptr);
        CHECK(firstSurface->geomInstSlot != SlotFinder::InvalidSlotIndex);
        
        // Check that geometry variant is set (should be TriangleGeometry)
        CHECK(std::holds_alternative<TriangleGeometry>(firstSurface->geometry));
    }
    
    SUBCASE("Multiple Surface Support")
    {
        // Create a cube (which has one surface)
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setModel(cube);
        
        auto model = std::static_pointer_cast<ShockerTriangleModel>(ShockerTriangleModel::create());
        SlotFinder slotFinder;
        slotFinder.initialize(100);
        model->createFromRenderableNode(node, slotFinder, nullptr);
        
        // Check one geometry instance per surface
        const auto& surfaces = model->getSurfaces();
        CHECK(surfaces.size() == cube->S.size());
        
        // Each surface will have one DisneyMaterial (to be set by MaterialHandler)
        for (const auto& surface : surfaces) {
            CHECK(surface->mat == nullptr);  // Not set yet, will be set by MaterialHandler
        }
    }
    
    SUBCASE("GeometryGroup Management")
    {
        // Create model from cube
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setModel(cube);
        
        auto model = std::static_pointer_cast<ShockerTriangleModel>(ShockerTriangleModel::create());
        SlotFinder slotFinder;
        slotFinder.initialize(100);
        model->createFromRenderableNode(node, slotFinder, nullptr);
        
        // Get the geometry group
        shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
        CHECK(surfaceGroup != nullptr);
        
        // Verify geometry group contains all geometry instances
        const auto& surfaces = model->getSurfaces();
        CHECK(surfaceGroup->geomInsts.size() == surfaces.size());
        
        // Check AABB was calculated for the group
        CHECK(surfaceGroup->aabb.minP.x == model->getAABB().minP.x);
        CHECK(surfaceGroup->aabb.maxP.x == model->getAABB().maxP.x);
        
        // Check flags are set correctly
        CHECK(surfaceGroup->needsRebuild == 1);  // Needs initial build
        CHECK(surfaceGroup->refittable == 0);     // Static geometry
    }
}

TEST_CASE("ShockerModel Instance Creation")
{
    SUBCASE("Create Instance from Model")
    {
        // Create model
        auto model = ShockerTriangleModel::create();
        SlotFinder slotFinder;
        slotFinder.initialize(100);
        
        // Create node with specific transform
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setModel(cube);
        
        sabi::SpaceTime& st = node->getSpaceTime();
        st.localTransform.setIdentity();
        st.localTransform.translate(Eigen::Vector3f(1.0f, 2.0f, 3.0f));
        st.localTransform.rotate(Eigen::AngleAxisf(M_PI/4, Eigen::Vector3f::UnitY()));
        st.worldTransform = st.localTransform;
        
        // Create model from node
        model->createFromRenderableNode(node, slotFinder, nullptr);
        
        // Model should have geometry group
        shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
        CHECK(surfaceGroup != nullptr);
        
        // Create a ShockerNode separately (as done by ShockerModelHandler)
        shocker::ShockerNode shockerNode;
        shockerNode.instSlot = 10;
        
        // Create surface group instance
        shocker::ShockerMesh::ShockerSurfaceGroupInstance groupInst;
        groupInst.geomGroup = surfaceGroup;
        groupInst.transform = ShockerModel::convertSpaceTimeToMatrix(st);
        
        shockerNode.geomGroupInst = groupInst;
        shockerNode.matM2W = groupInst.transform;
        shockerNode.nMatM2W = ShockerModel::calculateNormalMatrix(shockerNode.matM2W);
        shockerNode.prevMatM2W = shockerNode.matM2W;
        
        // Check that transform was applied
        CHECK(shockerNode.matM2W.c3.x == doctest::Approx(1.0f));
        CHECK(shockerNode.matM2W.c3.y == doctest::Approx(2.0f));
        CHECK(shockerNode.matM2W.c3.z == doctest::Approx(3.0f));
    }
    
    SUBCASE("Transform Utilities")
    {
        // Create a triangle model to test transform methods
        auto model = ShockerTriangleModel::create();
        
        // Test identity transform
        sabi::SpaceTime st;
        st.localTransform.setIdentity();
        st.worldTransform = st.localTransform;
        
        Matrix4x4 mat = model->convertSpaceTimeToMatrix(st);
        CHECK(mat.c0.x == doctest::Approx(1.0f));
        CHECK(mat.c1.y == doctest::Approx(1.0f));
        CHECK(mat.c2.z == doctest::Approx(1.0f));
        CHECK(mat.c3.w == doctest::Approx(1.0f));
        CHECK(mat.c3.x == doctest::Approx(0.0f));
        CHECK(mat.c3.y == doctest::Approx(0.0f));
        CHECK(mat.c3.z == doctest::Approx(0.0f));
        
        // Test translation
        st.localTransform.setIdentity();
        st.localTransform.translate(Eigen::Vector3f(5.0f, -3.0f, 7.0f));
        st.worldTransform = st.localTransform;
        
        mat = model->convertSpaceTimeToMatrix(st);
        CHECK(mat.c3.x == doctest::Approx(5.0f));
        CHECK(mat.c3.y == doctest::Approx(-3.0f));
        CHECK(mat.c3.z == doctest::Approx(7.0f));
        
        // Test scaling
        st.localTransform.setIdentity();
        st.localTransform.scale(2.0f);
        st.worldTransform = st.localTransform;
        
        mat = model->convertSpaceTimeToMatrix(st);
        CHECK(mat.c0.x == doctest::Approx(2.0f));
        CHECK(mat.c1.y == doctest::Approx(2.0f));
        CHECK(mat.c2.z == doctest::Approx(2.0f));
        
        // Test rotation (90 degrees around Y axis)
        st.localTransform.setIdentity();
        st.localTransform.rotate(Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitY()));
        st.worldTransform = st.localTransform;
        
        mat = model->convertSpaceTimeToMatrix(st);
        // After 90 degree Y rotation: X axis becomes Z, Z axis becomes -X
        // This is because we're rotating around Y axis positively (counter-clockwise when looking down Y)
        CHECK(mat.c0.x == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(mat.c0.z == doctest::Approx(-1.0f));
        CHECK(mat.c2.x == doctest::Approx(1.0f));
        CHECK(mat.c2.z == doctest::Approx(0.0f).epsilon(0.001f));
        
        // Test normal matrix calculation
        st.localTransform.setIdentity();
        st.localTransform.scale(2.0f);
        st.localTransform.translate(Eigen::Vector3f(1.0f, 2.0f, 3.0f));
        st.worldTransform = st.localTransform;
        
        mat = ShockerModel::convertSpaceTimeToMatrix(st);
        Matrix3x3 normalMat = ShockerModel::calculateNormalMatrix(mat);
        
        // Normal matrix should be inverse transpose of upper-left 3x3
        // For uniform scale of 2, the normal matrix should have scale of 0.5
        CHECK(normalMat.m00 == doctest::Approx(0.5f));
        CHECK(normalMat.m11 == doctest::Approx(0.5f));
        CHECK(normalMat.m22 == doctest::Approx(0.5f));
    }
}

TEST_CASE("ShockerFlyweightModel")
{
    SUBCASE("Flyweight Creation")
    {
        auto model = ShockerFlyweightModel::create();
        CHECK(model->getGeometryType() == ShockerGeometryType::Flyweight);
        
        // Create a node (flyweight doesn't need geometry)
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("FlyweightTest");
        
        sabi::SpaceTime& st = node->getSpaceTime();
        st.localTransform.setIdentity();
        st.localTransform.translate(Eigen::Vector3f(5.0f, 0.0f, 0.0f));
        st.worldTransform = st.localTransform;
        
        // Create from node
        SlotFinder slotFinder;
        slotFinder.initialize(100);
        model->createFromRenderableNode(node, slotFinder, nullptr);
        
        // Flyweight model doesn't create geometry, just references other models
        // AABB should be from source model or zero if no source
        const AABB& aabb = model->getAABB();
        CHECK(aabb.minP.x == 0.0f);
        CHECK(aabb.maxP.x == 0.0f);
    }
}

TEST_CASE("ShockerPhantomModel")
{
    SUBCASE("Phantom Creation")
    {
        auto model = ShockerPhantomModel::create();
        CHECK(model->getGeometryType() == ShockerGeometryType::Phantom);
        
        // Create a node
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("PhantomTest");
        
        // Create from node
        SlotFinder slotFinder;
        slotFinder.initialize(100);
        model->createFromRenderableNode(node, slotFinder, nullptr);
        
        // Phantom should have zero AABB (no visible geometry)
        const AABB& aabb = model->getAABB();
        CHECK(aabb.minP.x == 0.0f);
        CHECK(aabb.maxP.x == 0.0f);
        
        // Phantom should have geometry group with single empty instance
        shocker::ShockerSurfaceGroup* surfaceGroup = model->getSurfaceGroup();
        CHECK(surfaceGroup != nullptr);
        CHECK(surfaceGroup->geomInsts.size() == 1);  // Has one empty instance for the instance system
        
        // The geometry instance should exist but be empty
        const auto& surfaces = model->getSurfaces();
        CHECK(surfaces.size() == 1);
        CHECK(surfaces[0]->mat == nullptr);  // No material for phantom
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

// Include additional test files
#include "ShockerModelHandlerTest.cpp"
#include "ShockerMaterialHandlerTest.cpp"
#include "ShockerMaterialHandlerComplexTest.cpp"
#include "ShockerMaterialHandlerGPUTest.cpp"
#include "ShockerSceneHandlerTest.cpp"
#include "ShockerEngineGeometryTest.cpp"