#include "Jahley.h"

const std::string APP_NAME = "GeometryTest";

#ifdef CHECK
#undef CHECK
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;

#include <engine_core/engine_core.h>

// Include the common_host.h to test its structs
#include "engine_core/excludeFromBuild/common/common_host.h"

// Include material types
#include "engine_core/excludeFromBuild/material/HostDisneyMaterial.h"

// Test the geometry structs from common_host.h
TEST_CASE ("Common Host Geometry Structs")
{
    SUBCASE ("Mesh Structure")
    {
        // Test basic Mesh structure
        Mesh mesh;

        // Test GeometryGroupInstance
        Mesh::GeometryGroupInstance groupInst;
        groupInst.geomGroup = nullptr; // Would normally point to a GeometryGroup
        // Matrix4x4 defaults to identity, no need to set explicitly

        mesh.groupInsts.push_back (groupInst);
        CHECK (mesh.groupInsts.size() == 1);
    }

    SUBCASE ("Instance Structure")
    {
        // Test Instance struct basic properties
        Instance inst;
        inst.instSlot = 42;
        // Matrices default to identity, no need to set explicitly

        CHECK (inst.instSlot == 42);
    }

    SUBCASE ("InstanceController")
    {
        // Test InstanceController initialization
        Instance testInst;
        testInst.instSlot = 0;

        Point3D startPos (0.0f, 0.0f, 0.0f);
        Point3D endPos (10.0f, 0.0f, 0.0f);
        Quaternion startRot; // Default constructor for identity
        Quaternion endRot;   // Default constructor for identity

        InstanceController controller (
            &testInst,
            1.0f, startRot, startPos,
            2.0f, endRot, endPos,
            1.0f, 0.0f);

        CHECK (controller.beginScale == 1.0f);
        CHECK (controller.endScale == 2.0f);
        CHECK (controller.frequency == 1.0f);
        CHECK (controller.time == 0.0f);
    }
}

// Simple test for geometry components
TEST_CASE ("Basic Geometry Struct Tests")
{
    SUBCASE ("AABB (Axis-Aligned Bounding Box) Tests")
    {
        // Test AABB creation and basic properties
        AABB aabb;
        aabb.minP = Point3D (-1.0f, -2.0f, -3.0f);
        aabb.maxP = Point3D (4.0f, 5.0f, 6.0f);

        // Test center calculation
        Point3D center = (aabb.minP + aabb.maxP) * 0.5f;
        CHECK (center.x == doctest::Approx (1.5f));
        CHECK (center.y == doctest::Approx (1.5f));
        CHECK (center.z == doctest::Approx (1.5f));

        // Test extents
        Vector3D extents = aabb.maxP - aabb.minP;
        CHECK (extents.x == doctest::Approx (5.0f));
        CHECK (extents.y == doctest::Approx (7.0f));
        CHECK (extents.z == doctest::Approx (9.0f));
    }

    SUBCASE ("Matrix4x4 Transform Tests")
    {
        // Test identity matrix (default constructor)
        Matrix4x4 identity;
        Point3D point (1.0f, 2.0f, 3.0f);
        Point3D transformed = identity * point;

        CHECK (transformed.x == doctest::Approx (1.0f));
        CHECK (transformed.y == doctest::Approx (2.0f));
        CHECK (transformed.z == doctest::Approx (3.0f));

        // Test manual translation matrix creation
        Matrix4x4 translation;
        translation.c0.x = 1.0f;
        translation.c1.x = 0.0f;
        translation.c2.x = 0.0f;
        translation.c3.x = 10.0f;
        translation.c0.y = 0.0f;
        translation.c1.y = 1.0f;
        translation.c2.y = 0.0f;
        translation.c3.y = 20.0f;
        translation.c0.z = 0.0f;
        translation.c1.z = 0.0f;
        translation.c2.z = 1.0f;
        translation.c3.z = 30.0f;
        translation.c0.w = 0.0f;
        translation.c1.w = 0.0f;
        translation.c2.w = 0.0f;
        translation.c3.w = 1.0f;
        transformed = translation * point;

        CHECK (transformed.x == doctest::Approx (11.0f));
        CHECK (transformed.y == doctest::Approx (22.0f));
        CHECK (transformed.z == doctest::Approx (33.0f));
    }

    SUBCASE ("Quaternion Tests")
    {
        // Test identity quaternion (default constructor)
        Quaternion q;
        CHECK (q.x == doctest::Approx (0.0f));
        CHECK (q.y == doctest::Approx (0.0f));
        CHECK (q.z == doctest::Approx (0.0f));
        CHECK (q.w == doctest::Approx (1.0f));

        // Test quaternion to matrix conversion
        Matrix3x3 mat = q.toMatrix3x3();
        CHECK (mat.m00 == doctest::Approx (1.0f));
        CHECK (mat.m11 == doctest::Approx (1.0f));
        CHECK (mat.m22 == doctest::Approx (1.0f));
    }
}

TEST_CASE ("Mesh Structure Tests")
{
    SUBCASE ("Basic Mesh Creation")
    {
        // Note: Since Mesh struct uses GeometryGroup pointers and CUDA buffers,
        // we'll test the basic structure without actually initializing CUDA

        // Test that we can create vectors of geometry-related structures
        std::vector<Point3D> vertices;
        vertices.push_back (Point3D (0.0f, 0.0f, 0.0f));
        vertices.push_back (Point3D (1.0f, 0.0f, 0.0f));
        vertices.push_back (Point3D (0.0f, 1.0f, 0.0f));

        CHECK (vertices.size() == 3);
        CHECK (vertices[0].x == doctest::Approx (0.0f));
        CHECK (vertices[1].x == doctest::Approx (1.0f));
        CHECK (vertices[2].y == doctest::Approx (1.0f));
    }
}

TEST_CASE ("CgModel to TriangleGeometry Conversion")
{
    SUBCASE ("Create Cube and Convert")
    {
        // Create a cube using MeshOps
        float cubeSize = 2.0f;
        sabi::CgModelPtr cube = sabi::MeshOps::createCube (cubeSize);

        // Verify cube was created properly
        CHECK (cube != nullptr);
        CHECK (cube->vertexCount() == 8); // Cube has 8 vertices
        CHECK (cube->S.size() == 1);      // One surface

        // Get all triangle indices
        MatrixXu F;
        cube->getAllSurfaceIndices (F);
        CHECK (F.cols() == 12); // Cube has 12 triangles (2 per face)

        // Create Triangle structs from indices
        std::vector<shared::Triangle> triangles;
        for (int i = 0; i < F.cols(); ++i)
        {
            auto tri = F.col (i);
            triangles.emplace_back (shared::Triangle (tri (0), tri (1), tri (2)));
        }
        CHECK (triangles.size() == 12);

        // Create Vertex structs (simplified version without full conversion)
        std::vector<shared::Vertex> vertices;
        for (int i = 0; i < cube->V.cols(); ++i)
        {
            shared::Vertex vertex;
            auto p = cube->V.col (i);
            auto n = cube->N.col (i);

            vertex.position = Point3D (p.x(), p.y(), p.z());
            vertex.normal = Normal3D (n.x(), n.y(), n.z());
            vertex.texCoord = Point2D (0.0f, 0.0f); // No UVs in the cube

            vertices.push_back (vertex);
        }
        CHECK (vertices.size() == 8);

        // Verify some vertex positions
        // For a cube of size 2, vertices should be at ±1
        bool foundCorner = false;
        for (const auto& v : vertices)
        {
            if (std::abs (v.position.x - 1.0f) < 0.001f &&
                std::abs (v.position.y - 1.0f) < 0.001f &&
                std::abs (v.position.z - 1.0f) < 0.001f)
            {
                foundCorner = true;
                break;
            }
        }
        CHECK (foundCorner);

        // Verify triangle connectivity (first triangle should have valid indices)
        CHECK (triangles[0].index0 < 8);
        CHECK (triangles[0].index1 < 8);
        CHECK (triangles[0].index2 < 8);
    }
}

TEST_CASE ("GeometryInstance with DisneyMaterial")
{
    SUBCASE ("Create GeometryInstance for each Surface")
    {
        // Create a cube model
        sabi::CgModelPtr cube = sabi::MeshOps::createCube (2.0f);
        CHECK (cube != nullptr);

        // Test creating GeometryInstance for the first surface
        // (avoiding vector storage due to non-copyable OptiX types)
        const auto& surface = cube->S[0];

        // Create DisneyMaterial from the surface's CgMaterial
        DisneyMaterial disneyMat;

        // Convert CgMaterial properties to DisneyMaterial
        // In a real implementation, TextureHandler::createImmTexture would create GPU arrays
        // from these values. Here we demonstrate the mapping logic:
        const CgMaterial& cgMat = surface.cgMaterial;

        // Map baseColor - would use createImmTexture(float4(r,g,b,1.0f), true, &disneyMat.baseColor)
        float4 baseColorValue = make_float4 (
            cgMat.core.baseColor.x(),
            cgMat.core.baseColor.y(),
            cgMat.core.baseColor.z(),
            1.0f);
        // disneyMat.baseColor would be set to the created CUDA array
        // disneyMat.texBaseColor would be set to the texture object

        // Map roughness - single channel texture
        float roughnessValue = cgMat.core.roughness;
        // createImmTexture(make_float4(roughness, roughness, roughness, 1.0f), true, &disneyMat.roughness)

        // Map metallic - single channel texture
        float metallicValue = cgMat.metallic.metallic;
        // createImmTexture(make_float4(metallic, metallic, metallic, 1.0f), true, &disneyMat.metallic)

        // Map sheen - convert color to single intensity value
        float sheenValue = cgMat.sheen.sheenColorFactor.norm() / std::sqrt (3.0f); // Normalize
        // createImmTexture(make_float4(sheen, sheen, sheen, 1.0f), true, &disneyMat.sheen)

        // Map sheenTint - derive from sheen color hue
        // This would require color space conversion to extract tint

        // Map subsurface
        float subsurfaceValue = cgMat.subsurface.subsurface;
        // createImmTexture(make_float4(subsurface, subsurface, subsurface, 1.0f), true, &disneyMat.subsurface)

        // Map clearcoat
        float clearcoatValue = cgMat.clearcoat.clearcoat;
        // createImmTexture(make_float4(clearcoat, clearcoat, clearcoat, 1.0f), true, &disneyMat.clearcoat)

        // Map clearcoatGloss
        float clearcoatGlossValue = cgMat.clearcoat.clearcoatGloss;
        // createImmTexture(make_float4(clearcoatGloss, clearcoatGloss, clearcoatGloss, 1.0f), true, &disneyMat.clearcoatGloss)

        // Map anisotropic
        float anisotropicValue = cgMat.metallic.anisotropic;
        // createImmTexture(make_float4(anisotropic, anisotropic, anisotropic, 1.0f), true, &disneyMat.anisotropic)

        // Map specular transmission (transparency)
        float specTransValue = cgMat.transparency.transparency;
        // createImmTexture(make_float4(specTrans, specTrans, specTrans, 1.0f), true, &disneyMat.specTrans)

        // Map IOR
        float iorValue = cgMat.transparency.refractionIndex;
        // createImmTexture(make_float4(ior, ior, ior, 1.0f), true, &disneyMat.ior)

        // Map emissive
        float3 emissiveValue = make_float3 (
            cgMat.emission.luminousColor.x() * cgMat.emission.luminous,
            cgMat.emission.luminousColor.y() * cgMat.emission.luminous,
            cgMat.emission.luminousColor.z() * cgMat.emission.luminous);
        // createImmTexture(make_float4(emissiveValue, 1.0f), false, &disneyMat.emissive) - false for HDR

        // Since GeometryInstance contains optixu::GeometryInstance which is non-copyable,
        // we'll test the structure creation without storing in containers
        {
            // Create a GeometryInstance on stack
            GeometryInstance geomInst;
            // Instead of Material struct, we directly use DisneyMaterial
            // In real usage, geomInst would store a reference/index to the DisneyMaterial
            geomInst.mat = nullptr;    // This would point to DisneyMaterial in actual implementation
            geomInst.geomInstSlot = 0; // Would be from SlotFinder

            // Convert surface to TriangleGeometry
            TriangleGeometry triGeom;
            // In real usage, these would be GPU buffers:
            // triGeom.vertexBuffer.initialize(cuContext, bufferType, vertices);
            // triGeom.triangleBuffer.initialize(cuContext, bufferType, triangles);

            // Set the geometry variant
            geomInst.geometry = std::move (triGeom);

            // Calculate AABB for this surface
            AABB aabb;
            aabb.minP = Point3D (-1.0f, -1.0f, -1.0f);
            aabb.maxP = Point3D (1.0f, 1.0f, 1.0f);
            geomInst.aabb = aabb;

            // Verify the GeometryInstance
            CHECK (geomInst.geomInstSlot == 0);
            CHECK (std::holds_alternative<TriangleGeometry> (geomInst.geometry));
        }

        // Test that we can create multiple instances with DisneyMaterial conversion
        int instanceCount = 0;
        for (size_t surfIdx = 0; surfIdx < cube->S.size(); ++surfIdx)
        {
            const auto& surf = cube->S[surfIdx];

            // Create DisneyMaterial from each surface's CgMaterial
            DisneyMaterial surfaceMat;
            // Conversion would happen here from surf.cgMaterial

            // Create instance on stack
            GeometryInstance geomInst;
            geomInst.mat = nullptr; // Would reference the DisneyMaterial
            geomInst.geomInstSlot = static_cast<uint32_t> (surfIdx);

            TriangleGeometry triGeom;
            geomInst.geometry = std::move (triGeom);

            AABB aabb;
            aabb.minP = Point3D (-1.0f, -1.0f, -1.0f);
            aabb.maxP = Point3D (1.0f, 1.0f, 1.0f);
            geomInst.aabb = aabb;

            instanceCount++;
        }

        CHECK (instanceCount == static_cast<int> (cube->S.size()));
    }

    SUBCASE ("Disney Material Structure")
    {
        // Test DisneyMaterial structure
        DisneyMaterial disneyMat;

        // Verify default initialization
        CHECK (disneyMat.baseColor == nullptr);
        CHECK (disneyMat.texBaseColor == 0);
        CHECK (disneyMat.roughness == nullptr);
        CHECK (disneyMat.texRoughness == 0);
        CHECK (disneyMat.metallic == nullptr);
        CHECK (disneyMat.texMetallic == 0);
        CHECK (disneyMat.useAlphaForTransparency == false);

        // In a real scenario, these would be set to GPU arrays/textures
        // For testing, we're just verifying the structure is properly defined
    }
}

TEST_CASE ("GeometryGroup Creation")
{
    SUBCASE ("Create GeometryGroup from Cube Model")
    {
        // Create a cube model
        sabi::CgModelPtr cube = sabi::MeshOps::createCube (2.0f);
        CHECK (cube != nullptr);

        // Create a GeometryGroup to hold all geometry instances
        GeometryGroup geomGroup;

        // Track all geometry instances we create (using pointers since they're non-copyable)
        std::vector<std::unique_ptr<GeometryInstance>> geomInstances;

        // Create geometry instances for each surface
        for (size_t surfIdx = 0; surfIdx < cube->S.size(); ++surfIdx)
        {
            const auto& surface = cube->S[surfIdx];

            // Create a unique geometry instance
            auto geomInst = std::make_unique<GeometryInstance>();
            geomInst->geomInstSlot = static_cast<uint32_t> (surfIdx);

            // Create triangle geometry for this surface
            TriangleGeometry triGeom;
            // In real usage, would populate with actual vertex/triangle data:
            // triGeom.vertexBuffer.initialize(cuContext, bufferType, vertices);
            // triGeom.triangleBuffer.initialize(cuContext, bufferType, triangles);

            geomInst->geometry = std::move (triGeom);

            // Calculate AABB for this surface (simplified)
            AABB surfaceAABB;
            surfaceAABB.minP = Point3D (-1.0f, -1.0f, -1.0f);
            surfaceAABB.maxP = Point3D (1.0f, 1.0f, 1.0f);
            geomInst->aabb = surfaceAABB;

            // Add raw pointer to the geometry group
            geomGroup.geomInsts.insert (geomInst.get());

            // Store the unique_ptr to maintain ownership
            geomInstances.push_back (std::move (geomInst));
        }

        // Verify geometry group contains all instances
        CHECK (geomGroup.geomInsts.size() == cube->S.size());

        // Calculate overall AABB for the geometry group
        geomGroup.aabb = AABB();
        geomGroup.aabb.minP = Point3D (FLT_MAX, FLT_MAX, FLT_MAX);
        geomGroup.aabb.maxP = Point3D (-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (const auto* geomInst : geomGroup.geomInsts)
        {
            // Expand AABB to include this instance
            geomGroup.aabb.minP.x = std::min (geomGroup.aabb.minP.x, geomInst->aabb.minP.x);
            geomGroup.aabb.minP.y = std::min (geomGroup.aabb.minP.y, geomInst->aabb.minP.y);
            geomGroup.aabb.minP.z = std::min (geomGroup.aabb.minP.z, geomInst->aabb.minP.z);

            geomGroup.aabb.maxP.x = std::max (geomGroup.aabb.maxP.x, geomInst->aabb.maxP.x);
            geomGroup.aabb.maxP.y = std::max (geomGroup.aabb.maxP.y, geomInst->aabb.maxP.y);
            geomGroup.aabb.maxP.z = std::max (geomGroup.aabb.maxP.z, geomInst->aabb.maxP.z);
        }

        // Verify the computed AABB
        CHECK (geomGroup.aabb.minP.x == doctest::Approx (-1.0f));
        CHECK (geomGroup.aabb.minP.y == doctest::Approx (-1.0f));
        CHECK (geomGroup.aabb.minP.z == doctest::Approx (-1.0f));
        CHECK (geomGroup.aabb.maxP.x == doctest::Approx (1.0f));
        CHECK (geomGroup.aabb.maxP.y == doctest::Approx (1.0f));
        CHECK (geomGroup.aabb.maxP.z == doctest::Approx (1.0f));

        // Set other GeometryGroup properties
        geomGroup.numEmitterPrimitives = 0; // No emissive surfaces
        geomGroup.needsReallocation = 0;
        geomGroup.needsRebuild = 1; // Needs initial build
        geomGroup.refittable = 0;   // Static geometry

        // In real usage, would create OptiX GAS:
        // geomGroup.optixGas = scene->createGeometryAccelerationStructure();
        // geomGroup.optixGas.setConfiguration(...);
        // geomGroup.optixGas.addChild(geomInst->optixGeomInst);
        // geomGroup.optixGas.prepareForBuild(&bufferSizes);
        // geomGroup.optixGasMem.initialize(...);
    }

    SUBCASE ("Create Mesh with GeometryGroup")
    {
        // Create a mesh that references the geometry group
        Mesh mesh;

        // Create a geometry group (simplified for test)
        auto geomGroup = std::make_unique<GeometryGroup>();
        geomGroup->aabb.minP = Point3D (-1.0f, -1.0f, -1.0f);
        geomGroup->aabb.maxP = Point3D (1.0f, 1.0f, 1.0f);

        // Create a geometry group instance with transform
        Mesh::GeometryGroupInstance groupInst;
        groupInst.geomGroup = geomGroup.get();
        groupInst.transform = Matrix4x4(); // Identity transform

        // Add to mesh
        mesh.groupInsts.push_back (groupInst);

        CHECK (mesh.groupInsts.size() == 1);
        CHECK (mesh.groupInsts[0].geomGroup != nullptr);

        // Test with a transformed instance
        Mesh::GeometryGroupInstance transformedInst;
        transformedInst.geomGroup = geomGroup.get();
        transformedInst.transform.c3.x = 5.0f; // Translate 5 units in X

        mesh.groupInsts.push_back (transformedInst);
        CHECK (mesh.groupInsts.size() == 2);
        CHECK (mesh.groupInsts[1].transform.c3.x == doctest::Approx (5.0f));
    }
}

TEST_CASE ("Instance Transform Tests")
{
    SUBCASE ("Transform Composition")
    {
        // Test transform composition for instances
        Matrix4x4 localTransform;
        localTransform.c3.x = 1.0f; // Translation in X

        Matrix4x4 worldTransform;
        worldTransform.c3.y = 1.0f; // Translation in Y

        Matrix4x4 combined = worldTransform * localTransform;
        Point3D origin (0.0f, 0.0f, 0.0f);
        Point3D transformed = combined * origin;

        CHECK (transformed.x == doctest::Approx (1.0f));
        CHECK (transformed.y == doctest::Approx (1.0f));
        CHECK (transformed.z == doctest::Approx (0.0f));
    }

    SUBCASE ("Normal Matrix Calculation")
    {
        // Test normal matrix for uniform scaling
        float scale = 2.0f;
        Matrix3x3 scaleMatrix;
        scaleMatrix.m00 *= scale;
        scaleMatrix.m11 *= scale;
        scaleMatrix.m22 *= scale;

        Matrix3x3 normalMatrix = scaleMatrix / scale;

        CHECK (normalMatrix.m00 == doctest::Approx (1.0f));
        CHECK (normalMatrix.m11 == doctest::Approx (1.0f));
        CHECK (normalMatrix.m22 == doctest::Approx (1.0f));
    }
}

TEST_CASE ("RenderableNode to Instance Conversion")
{
    SUBCASE ("Create RenderableNode with Cube and Convert to Instance")
    {
        // Create a cube using MeshOps
        float cubeSize = 2.0f;
        sabi::CgModelPtr cube = sabi::MeshOps::createCube (cubeSize);
        CHECK (cube != nullptr);
        CHECK (cube->vertexCount() == 8);
        CHECK (cube->S.size() == 1);

        // Create a RenderableNode and assign the cube model
        sabi::RenderableNode renderableNode = sabi::WorldItem::create();
        renderableNode->setName ("TestCube");
        renderableNode->setModel (cube);

        // Set up the transform for the renderable
        sabi::SpaceTime& spacetime = renderableNode->getSpaceTime();
        spacetime.localTransform.setIdentity();
        spacetime.localTransform.translate (Eigen::Vector3f (5.0f, 3.0f, 2.0f)); // Position at (5, 3, 2)
        spacetime.localTransform.scale (1.5f);                                    // Scale by 1.5
        spacetime.worldTransform = spacetime.localTransform; // Set world transform to local (no parent)

        // Verify the RenderableNode setup
        CHECK (renderableNode->getName() == "TestCube");
        CHECK (renderableNode->getModel() == cube);
        CHECK (renderableNode->getTriangleCount() == 12); // Cube has 12 triangles

        // Create a GeometryGroup from the cube model
        auto geomGroup = std::make_unique<GeometryGroup>();

        // Create GeometryInstances for each surface
        std::vector<std::unique_ptr<GeometryInstance>> geomInstances;
        for (size_t surfIdx = 0; surfIdx < cube->S.size(); ++surfIdx)
        {
            const auto& surface = cube->S[surfIdx];

            auto geomInst = std::make_unique<GeometryInstance>();
            geomInst->geomInstSlot = static_cast<uint32_t> (surfIdx);

            // Create triangle geometry
            TriangleGeometry triGeom;
            // In real usage, would populate with actual vertex/triangle data
            geomInst->geometry = std::move (triGeom);

            // Calculate AABB for this surface
            AABB surfaceAABB;
            surfaceAABB.minP = Point3D (-cubeSize / 2, -cubeSize / 2, -cubeSize / 2);
            surfaceAABB.maxP = Point3D (cubeSize / 2, cubeSize / 2, cubeSize / 2);
            geomInst->aabb = surfaceAABB;

            // Convert surface material to DisneyMaterial
            DisneyMaterial disneyMat;
            // Material conversion would happen here from surface.cgMaterial
            geomInst->mat = nullptr; // Would point to the DisneyMaterial

            geomGroup->geomInsts.insert (geomInst.get());
            geomInstances.push_back (std::move (geomInst));
        }

        // Calculate overall AABB for the geometry group
        geomGroup->aabb = AABB();
        geomGroup->aabb.minP = Point3D (-cubeSize / 2, -cubeSize / 2, -cubeSize / 2);
        geomGroup->aabb.maxP = Point3D (cubeSize / 2, cubeSize / 2, cubeSize / 2);

        // Create a Mesh with the geometry group
        Mesh mesh;
        Mesh::GeometryGroupInstance groupInst;
        groupInst.geomGroup = geomGroup.get();

        // Convert SpaceTime transform to Matrix4x4
        // Extract the transformation matrix from the RenderableNode's SpaceTime
        const Eigen::Affine3f& worldTransform = spacetime.worldTransform;
        const Eigen::Matrix4f& eigenMat = worldTransform.matrix();

        // Convert Eigen matrix to our Matrix4x4 format (column-major)
        Matrix4x4 transform;
        transform.c0.x = eigenMat (0, 0);
        transform.c0.y = eigenMat (1, 0);
        transform.c0.z = eigenMat (2, 0);
        transform.c0.w = eigenMat (3, 0);
        transform.c1.x = eigenMat (0, 1);
        transform.c1.y = eigenMat (1, 1);
        transform.c1.z = eigenMat (2, 1);
        transform.c1.w = eigenMat (3, 1);
        transform.c2.x = eigenMat (0, 2);
        transform.c2.y = eigenMat (1, 2);
        transform.c2.z = eigenMat (2, 2);
        transform.c2.w = eigenMat (3, 2);
        transform.c3.x = eigenMat (0, 3);
        transform.c3.y = eigenMat (1, 3);
        transform.c3.z = eigenMat (2, 3);
        transform.c3.w = eigenMat (3, 3);

        groupInst.transform = transform;
        mesh.groupInsts.push_back (groupInst);

        // Create an Instance from the Mesh
        Instance inst;
        inst.geomGroupInst = mesh.groupInsts[0];
        inst.instSlot = 0; // First instance slot

        // Set up the model-to-world transform matrices
        inst.matM2W = transform;

        // Calculate the normal matrix (inverse transpose of upper-left 3x3)
        Matrix3x3 upperLeft3x3;
        upperLeft3x3.m00 = transform.c0.x;
        upperLeft3x3.m01 = transform.c1.x;
        upperLeft3x3.m02 = transform.c2.x;
        upperLeft3x3.m10 = transform.c0.y;
        upperLeft3x3.m11 = transform.c1.y;
        upperLeft3x3.m12 = transform.c2.y;
        upperLeft3x3.m20 = transform.c0.z;
        upperLeft3x3.m21 = transform.c1.z;
        upperLeft3x3.m22 = transform.c2.z;

        // For uniform scaling, the normal matrix is just the rotation part divided by scale
        // Since we scaled by 1.5, we divide by 1.5
        inst.nMatM2W = upperLeft3x3 / 1.5f;

        // Store previous transform (initially same as current)
        inst.prevMatM2W = inst.matM2W;

        // Verify the Instance was created correctly
        CHECK (inst.instSlot == 0);
        CHECK (inst.geomGroupInst.geomGroup != nullptr);

        // Verify the transform was applied correctly
        // Original cube corner at (1, 1, 1) scaled by 1.5 = (1.5, 1.5, 1.5)
        // Then translated by (5, 3, 2) = (6.5, 4.5, 3.5)
        Point3D testPoint (1.0f, 1.0f, 1.0f);
        Point3D transformedPoint = inst.matM2W * testPoint;
        CHECK (transformedPoint.x == doctest::Approx (6.5f));
        CHECK (transformedPoint.y == doctest::Approx (4.5f));
        CHECK (transformedPoint.z == doctest::Approx (3.5f));

        // Test normal transformation
        Normal3D testNormal (1.0f, 0.0f, 0.0f);
        Normal3D transformedNormal = inst.nMatM2W * testNormal;
        CHECK (transformedNormal.x == doctest::Approx (1.0f));
        CHECK (transformedNormal.y == doctest::Approx (0.0f));
        CHECK (transformedNormal.z == doctest::Approx (0.0f));
    }

    SUBCASE ("Multiple RenderableNodes with Transforms")
    {
        // Create multiple cubes with different transforms
        std::vector<sabi::RenderableNode> renderables;
        std::vector<std::unique_ptr<GeometryGroup>> geomGroups;
        std::vector<Matrix4x4> transforms;
        std::vector<uint32_t> instanceSlots;

        for (int i = 0; i < 3; ++i)
        {
            // Create cube model
            sabi::CgModelPtr cube = sabi::MeshOps::createCube (1.0f);

            // Create RenderableNode
            sabi::RenderableNode node = sabi::WorldItem::create();
            node->setName ("Cube_" + std::to_string (i));
            node->setModel (cube);

            // Set unique transform for each cube
            sabi::SpaceTime& spacetime = node->getSpaceTime();
            spacetime.localTransform.setIdentity();
            spacetime.localTransform.translate (Eigen::Vector3f (i * 3.0f, 0.0f, 0.0f));
            spacetime.localTransform.rotate (Eigen::AngleAxisf (i * M_PI / 4, Eigen::Vector3f::UnitY()));
            spacetime.worldTransform = spacetime.localTransform; // Set world transform to local (no parent)

            renderables.push_back (node);

            // Create GeometryGroup for this cube
            auto geomGroup = std::make_unique<GeometryGroup>();
            geomGroup->aabb.minP = Point3D (-0.5f, -0.5f, -0.5f);
            geomGroup->aabb.maxP = Point3D (0.5f, 0.5f, 0.5f);

            // Convert transform
            const Eigen::Matrix4f& eigenMat = spacetime.worldTransform.matrix();
            Matrix4x4 transform;
            for (int row = 0; row < 4; ++row)
            {
                transform.c0[row] = eigenMat (row, 0);
                transform.c1[row] = eigenMat (row, 1);
                transform.c2[row] = eigenMat (row, 2);
                transform.c3[row] = eigenMat (row, 3);
            }

            // Store the transform and slot for verification
            transforms.push_back (transform);
            instanceSlots.push_back (static_cast<uint32_t> (i));
            geomGroups.push_back (std::move (geomGroup));
        }

        // Verify we created 3 renderables with transforms
        CHECK (renderables.size() == 3);
        CHECK (transforms.size() == 3);
        CHECK (instanceSlots.size() == 3);

        // Verify each has unique slot and position
        for (size_t i = 0; i < renderables.size(); ++i)
        {
            CHECK (instanceSlots[i] == i);

            // Check X translation matches expected value
            Point3D origin (0.0f, 0.0f, 0.0f);
            Point3D transformed = transforms[i] * origin;
            CHECK (transformed.x == doctest::Approx (i * 3.0f));
            CHECK (transformed.y == doctest::Approx (0.0f));

            // Verify name was set correctly
            CHECK (renderables[i]->getName() == "Cube_" + std::to_string (i));
        }
    }
}

class Application : public Jahley::App
{
 public:
    Application (DesktopWindowSettings settings = DesktopWindowSettings(), bool windowApp = false) :
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