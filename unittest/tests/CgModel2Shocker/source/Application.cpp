#include "Jahley.h"

const std::string APP_NAME = "CgModel2Shocker";

#ifdef CHECK
#undef CHECK
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;

#include <sabi_core/sabi_core.h>
#include <claude_core/excludeFromBuild/common/common_host.h>
#include <claude_core/excludeFromBuild/GPUContext.h>

// We need to define these types since they're in the shared namespace
using shared::Vertex;
using shared::Triangle;

namespace
{
    std::shared_ptr<sabi::CgModel> loadGLTF (const std::filesystem::path& gltfPath)
    {
        try
        {
            GLTFImporter gltf;
            auto [cgModel, animations] = gltf.importModel (gltfPath.generic_string());
            if (!cgModel)
            {
                LOG (WARNING) << "Load failed " << gltfPath.string();
                return nullptr;
            }
            return cgModel;
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Exception loading " << gltfPath.string() << ": " << e.what();
            return nullptr;
        }
    }
} // namespace

TEST_CASE("CgModel GLTF Loading - Box Model")
{
    SUBCASE("Load valid Box.gltf file")
    {
        std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
        auto model = loadGLTF(gltfPath);
        
        REQUIRE(model != nullptr);
    }
    
    SUBCASE("Verify Box model geometry")
    {
        std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
        auto model = loadGLTF(gltfPath);
        
        REQUIRE(model != nullptr);
        REQUIRE(model->S.size() > 0);
        
        // Check first surface
        const auto& surface = model->S[0];
        
        // Box should have 24 vertices (6 faces * 4 vertices per face)
        CHECK(model->vertexCount() == 24);
        
        // Box should have 12 triangles (6 faces * 2 triangles)
        CHECK(surface.triangleCount() == 12);
        
        // Should have normals
        CHECK(model->N.cols() == 24);
        
        // Verify bounding box is reasonable for a unit box
        auto bbox = model->computeBoundingBox();
        CHECK(bbox.min().x() >= -0.6f);
        CHECK(bbox.min().y() >= -0.6f);
        CHECK(bbox.min().z() >= -0.6f);
        CHECK(bbox.max().x() <= 0.6f);
        CHECK(bbox.max().y() <= 0.6f);
        CHECK(bbox.max().z() <= 0.6f);
    }
    
    SUBCASE("Verify Box model material")
    {
        std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
        auto model = loadGLTF(gltfPath);
        
        REQUIRE(model != nullptr);
        REQUIRE(model->S.size() > 0);
        
        const auto& surface = model->S[0];
        const auto& material = surface.cgMaterial;
        
        // Check material name
        CHECK(material.name == "Red");
        
        // Check base color (should be red: 0.8, 0.0, 0.0)
        CHECK(material.core.baseColor.x() == doctest::Approx(0.8f));
        CHECK(material.core.baseColor.y() == doctest::Approx(0.0f));
        CHECK(material.core.baseColor.z() == doctest::Approx(0.0f));
        
        // Check metallic factor
        CHECK(material.metallic.metallic == doctest::Approx(0.0f));
    }
    
    SUBCASE("Load non-existent file returns nullptr")
    {
        std::filesystem::path invalidPath = "E:/non_existent_file.gltf";
        auto model = loadGLTF(invalidPath);
        
        CHECK(model == nullptr);
    }
}

TEST_CASE("CgModelSurface to GeometryInstance Conversion")
{
    SUBCASE("Convert single surface to GeometryInstance")
    {
        // Initialize GPU context
        GPUContext gpuContext;
        bool initSuccess = gpuContext.initialize();
        REQUIRE(initSuccess);
        
        CUcontext cudaContext = gpuContext.getCudaContext();
        REQUIRE(cudaContext != nullptr);
        
        {  // Scope for GeometryInstance to ensure it's destroyed before GPU cleanup
            std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
            auto model = loadGLTF(gltfPath);
            
            REQUIRE(model != nullptr);
            REQUIRE(model->S.size() > 0);
            
            // Get the first surface
            const auto& surface = model->S[0];
            const auto& V = model->V;  // Vertex positions
            const auto& N = model->N;  // Normals
            const auto& UV0 = model->UV0;  // Texture coordinates
            
            // Create GeometryInstance
            GeometryInstance geomInst;
            
            // Create TriangleGeometry
            TriangleGeometry triGeom;
            
            // Allocate vertex buffer
            std::vector<shared::Vertex> vertices;
            vertices.reserve(model->vertexCount());
            
            // Convert vertices
            for (size_t i = 0; i < model->vertexCount(); ++i)
            {
            shared::Vertex vertex;
            
            // Position
            vertex.position = Point3D(V(0, i), V(1, i), V(2, i));
            
            // Normal
            if (N.cols() > i)
            {
                vertex.normal = Normal3D(N(0, i), N(1, i), N(2, i));
            }
            else
            {
                vertex.normal = Normal3D(0, 1, 0);  // Default up normal
            }
            
            // Texture coordinates
            if (UV0.cols() > i)
            {
                vertex.texCoord = Point2D(UV0(0, i), UV0(1, i));
            }
            else
            {
                vertex.texCoord = Point2D(0, 0);
            }
            
            // Texture coordinate direction (tangent) - simplified for now
            vertex.texCoord0Dir = Vector3D(1, 0, 0);
            
            vertices.push_back(vertex);
            }
            
            // Convert indices to triangles
            std::vector<shared::Triangle> triangles;
            const auto& F = surface.indices();
            triangles.reserve(F.cols());
            
            for (int i = 0; i < F.cols(); ++i)
            {
            const auto& tri = F.col(i);
            shared::Triangle triangle;
            triangle.index0 = tri[0];
            triangle.index1 = tri[1];
            triangle.index2 = tri[2];
            triangles.push_back(triangle);
            }
            
            // Verify conversion
            CHECK(vertices.size() == model->vertexCount());
            CHECK(triangles.size() == surface.triangleCount());
            
            // Debug output to understand the data
            LOG(INFO) << "Vertex count: " << vertices.size();
            LOG(INFO) << "Triangle count: " << triangles.size();
            LOG(INFO) << "First vertex position: (" << V(0, 0) << ", " << V(1, 0) << ", " << V(2, 0) << ")";
            LOG(INFO) << "First triangle indices: [" << F(0, 0) << ", " << F(1, 0) << ", " << F(2, 0) << "]";
            
            // Verify first vertex
            CHECK(vertices[0].position.x == doctest::Approx(V(0, 0)));
            CHECK(vertices[0].position.y == doctest::Approx(V(1, 0)));
            CHECK(vertices[0].position.z == doctest::Approx(V(2, 0)));
            
            // Verify first triangle indices
            // F.col(0) gives us the first triangle's indices
            const auto& firstTri = F.col(0);
            CHECK(triangles[0].index0 == firstTri[0]);
            CHECK(triangles[0].index1 == firstTri[1]);
            CHECK(triangles[0].index2 == firstTri[2]);
            
            // Now upload to GPU buffers
            triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
            
            triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
            
            // Set the geometry variant
            geomInst.geometry = std::move(triGeom);
            
            // Verify we can access it back
            CHECK(std::holds_alternative<TriangleGeometry>(geomInst.geometry));
        
            // Verify buffer sizes
            const auto& storedGeom = std::get<TriangleGeometry>(geomInst.geometry);
            CHECK(storedGeom.vertexBuffer.numElements() == vertices.size());
            CHECK(storedGeom.triangleBuffer.numElements() == triangles.size());
        }  // GeometryInstance destroyed here
        
        // Clean up GPU context
        gpuContext.cleanup();
    }
    
    SUBCASE("Verify vertex normal conversion")
    {
        std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
        auto model = loadGLTF(gltfPath);
        
        REQUIRE(model != nullptr);
        
        std::vector<shared::Vertex> vertices;
        const auto& N = model->N;
        
        // Convert just the normals
        for (int i = 0; i < N.cols(); ++i)
        {
            shared::Vertex vertex;
            vertex.normal = Normal3D(N(0, i), N(1, i), N(2, i));
            vertices.push_back(vertex);
            
            // Verify normal is still normalized after conversion
            float length = std::sqrt(
                vertex.normal.x * vertex.normal.x +
                vertex.normal.y * vertex.normal.y +
                vertex.normal.z * vertex.normal.z
            );
            CHECK(length == doctest::Approx(1.0f).epsilon(0.01f));
        }
    }
}

TEST_CASE("GeometryGroup Creation")
{
    SUBCASE("Create GeometryGroup with single GeometryInstance")
    {
        // Initialize GPU context
        GPUContext gpuContext;
        bool initSuccess = gpuContext.initialize();
        REQUIRE(initSuccess);
        
        CUcontext cudaContext = gpuContext.getCudaContext();
        optixu::Context optixContext = gpuContext.getOptiXContext();
        REQUIRE(cudaContext != nullptr);
        
        {  // Scope for GPU resources
            std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
            auto model = loadGLTF(gltfPath);
            
            REQUIRE(model != nullptr);
            REQUIRE(model->S.size() > 0);
            
            // Create a GeometryInstance (reusing code from previous test)
            const auto& surface = model->S[0];
            const auto& V = model->V;
            const auto& N = model->N;
            const auto& UV0 = model->UV0;
            
            auto geomInst = std::make_unique<GeometryInstance>();
            
            // Create and populate TriangleGeometry
            TriangleGeometry triGeom;
            std::vector<shared::Vertex> vertices;
            vertices.reserve(model->vertexCount());
            
            for (size_t i = 0; i < model->vertexCount(); ++i)
            {
                shared::Vertex vertex;
                vertex.position = Point3D(V(0, i), V(1, i), V(2, i));
                vertex.normal = (N.cols() > i) ? Normal3D(N(0, i), N(1, i), N(2, i)) : Normal3D(0, 1, 0);
                vertex.texCoord = (UV0.cols() > i) ? Point2D(UV0(0, i), UV0(1, i)) : Point2D(0, 0);
                vertex.texCoord0Dir = Vector3D(1, 0, 0);
                vertices.push_back(vertex);
            }
            
            std::vector<shared::Triangle> triangles;
            const auto& F = surface.indices();
            triangles.reserve(F.cols());
            
            for (int i = 0; i < F.cols(); ++i)
            {
                const auto& tri = F.col(i);
                shared::Triangle triangle;
                triangle.index0 = tri[0];
                triangle.index1 = tri[1];
                triangle.index2 = tri[2];
                triangles.push_back(triangle);
            }
            
            // Upload to GPU
            triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
            triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
            geomInst->geometry = std::move(triGeom);
            
            // Calculate AABB for the geometry
            AABB geomAABB;
            geomAABB.minP = Point3D(V.row(0).minCoeff(), V.row(1).minCoeff(), V.row(2).minCoeff());
            geomAABB.maxP = Point3D(V.row(0).maxCoeff(), V.row(1).maxCoeff(), V.row(2).maxCoeff());
            geomInst->aabb = geomAABB;
            
            // Create GeometryGroup
            GeometryGroup geomGroup;
            geomGroup.geomInsts.insert(geomInst.get());
            geomGroup.numEmitterPrimitives = 0;  // No emissive geometry
            geomGroup.aabb = geomAABB;
            geomGroup.needsReallocation = 1;
            geomGroup.needsRebuild = 1;
            geomGroup.refittable = 0;
            
            // Verify GeometryGroup creation
            CHECK(geomGroup.geomInsts.size() == 1);
            CHECK(geomGroup.geomInsts.count(geomInst.get()) == 1);
            CHECK(geomGroup.aabb.minP.x == doctest::Approx(-0.5f));
            CHECK(geomGroup.aabb.maxP.x == doctest::Approx(0.5f));
        }
        
        // Clean up GPU context
        gpuContext.cleanup();
    }
    
    SUBCASE("Create GeometryGroup with multiple GeometryInstances")
    {
        // Initialize GPU context
        GPUContext gpuContext;
        bool initSuccess = gpuContext.initialize();
        REQUIRE(initSuccess);
        
        CUcontext cudaContext = gpuContext.getCudaContext();
        REQUIRE(cudaContext != nullptr);
        
        {  // Scope for GPU resources
            // Create multiple GeometryInstances
            std::vector<std::unique_ptr<GeometryInstance>> geomInstances;
            
            // Create 3 dummy geometry instances
            for (int idx = 0; idx < 3; ++idx)
            {
                auto geomInst = std::make_unique<GeometryInstance>();
                
                // Create simple triangle geometry
                TriangleGeometry triGeom;
                
                // Create a simple triangle with offset
                std::vector<shared::Vertex> vertices(3);
                float offset = idx * 2.0f;
                vertices[0].position = Point3D(offset, 0, 0);
                vertices[1].position = Point3D(offset + 1, 0, 0);
                vertices[2].position = Point3D(offset + 0.5f, 1, 0);
                
                for (auto& v : vertices)
                {
                    v.normal = Normal3D(0, 0, 1);
                    v.texCoord = Point2D(0, 0);
                    v.texCoord0Dir = Vector3D(1, 0, 0);
                }
                
                std::vector<shared::Triangle> triangles(1);
                triangles[0].index0 = 0;
                triangles[0].index1 = 1;
                triangles[0].index2 = 2;
                
                // Upload to GPU
                triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
                triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
                geomInst->geometry = std::move(triGeom);
                
                // Set AABB
                AABB aabb;
                aabb.minP = Point3D(offset, 0, 0);
                aabb.maxP = Point3D(offset + 1, 1, 0);
                geomInst->aabb = aabb;
                
                geomInstances.push_back(std::move(geomInst));
            }
            
            // Create GeometryGroup with all instances
            GeometryGroup geomGroup;
            AABB combinedAABB;
            combinedAABB.minP = Point3D(0, 0, 0);
            combinedAABB.maxP = Point3D(5, 1, 0);
            
            for (const auto& inst : geomInstances)
            {
                geomGroup.geomInsts.insert(inst.get());
            }
            
            geomGroup.numEmitterPrimitives = 0;
            geomGroup.aabb = combinedAABB;
            geomGroup.needsReallocation = 1;
            geomGroup.needsRebuild = 1;
            geomGroup.refittable = 0;
            
            // Verify GeometryGroup with multiple instances
            CHECK(geomGroup.geomInsts.size() == 3);
            CHECK(geomGroup.aabb.minP.x == doctest::Approx(0.0f));
            CHECK(geomGroup.aabb.maxP.x == doctest::Approx(5.0f));
            
            // Verify all instances are in the group
            for (const auto& inst : geomInstances)
            {
                CHECK(geomGroup.geomInsts.count(inst.get()) == 1);
            }
        }
        
        // Clean up GPU context
        gpuContext.cleanup();
    }
}

TEST_CASE("GeometryInstanceData Creation") 
{
    SUBCASE("Create GeometryInstanceData from GeometryInstance")
    {
        // Initialize GPU context
        GPUContext gpuContext;
        bool initSuccess = gpuContext.initialize();
        REQUIRE(initSuccess);
        
        CUcontext cudaContext = gpuContext.getCudaContext();
        REQUIRE(cudaContext != nullptr);
        
        {  // Scope for GPU resources
            // Load model
            std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
            auto model = loadGLTF(gltfPath);
            
            REQUIRE(model != nullptr);
            REQUIRE(model->S.size() > 0);
            
            // Create geometry data
            const auto& surface = model->S[0];
            const auto& V = model->V;
            const auto& N = model->N;
            const auto& UV0 = model->UV0;
            
            // Create vertex and triangle buffers
            std::vector<shared::Vertex> vertices;
            vertices.reserve(model->vertexCount());
            
            for (size_t i = 0; i < model->vertexCount(); ++i)
            {
                shared::Vertex vertex;
                vertex.position = Point3D(V(0, i), V(1, i), V(2, i));
                vertex.normal = (N.cols() > i) ? Normal3D(N(0, i), N(1, i), N(2, i)) : Normal3D(0, 1, 0);
                vertex.texCoord = (UV0.cols() > i) ? Point2D(UV0(0, i), UV0(1, i)) : Point2D(0, 0);
                vertex.texCoord0Dir = Vector3D(1, 0, 0);
                vertices.push_back(vertex);
            }
            
            std::vector<shared::Triangle> triangles;
            const auto& F = surface.indices();
            triangles.reserve(F.cols());
            
            for (int i = 0; i < F.cols(); ++i)
            {
                const auto& tri = F.col(i);
                shared::Triangle triangle;
                triangle.index0 = tri[0];
                triangle.index1 = tri[1];
                triangle.index2 = tri[2];
                triangles.push_back(triangle);
            }
            
            // Create GeometryInstance with TriangleGeometry
            auto geomInst = std::make_unique<GeometryInstance>();
            TriangleGeometry triGeom;
            triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
            triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
            geomInst->geometry = std::move(triGeom);
            
            // Assign a slot number (simulating scene slot management)
            geomInst->geomInstSlot = 0;
            
            // Create GeometryInstanceData (similar to sample code)
            shared::GeometryInstanceData geomInstData = {};
            
            // Get the TriangleGeometry from the variant
            const auto& geom = std::get<TriangleGeometry>(geomInst->geometry);
            
            // Set the buffer views for GPU access
            geomInstData.vertexBuffer = geom.vertexBuffer.getROBuffer<shared::enableBufferOobCheck>();
            geomInstData.triangleBuffer = geom.triangleBuffer.getROBuffer<shared::enableBufferOobCheck>();
            
            // Set material and geometry instance slots
            geomInstData.materialSlot = 0;  // Default material
            geomInstData.geomInstSlot = geomInst->geomInstSlot;
            
            // Verify GeometryInstanceData fields
            CHECK(geomInstData.geomInstSlot == 0);
            CHECK(geomInstData.materialSlot == 0);
            
            // Successfully created GeometryInstanceData with ROBuffer references
            // The buffers are valid since they were created from initialized TypedBuffers
            
            LOG(INFO) << "GeometryInstanceData created successfully";
            LOG(INFO) << "  Vertex buffer: " << vertices.size() << " vertices";
            LOG(INFO) << "  Triangle buffer: " << triangles.size() << " triangles";
            LOG(INFO) << "  Material slot: " << geomInstData.materialSlot;
            LOG(INFO) << "  Geometry instance slot: " << geomInstData.geomInstSlot;
        }
        
        // Clean up GPU context
        gpuContext.cleanup();
    }
    
    SUBCASE("Create GeometryInstanceData buffer for multiple instances")
    {
        // Initialize GPU context
        GPUContext gpuContext;
        bool initSuccess = gpuContext.initialize();
        REQUIRE(initSuccess);
        
        CUcontext cudaContext = gpuContext.getCudaContext();
        REQUIRE(cudaContext != nullptr);
        
        {  // Scope for GPU resources
            // For simplicity, we'll just verify the basic structure creation
            // In a real scenario, Scene would manage the buffer with proper mapping
            
            // Create a host-side array to simulate what would be in the mapped buffer
            std::vector<shared::GeometryInstanceData> geomInstDataArray;
            std::vector<std::unique_ptr<GeometryInstance>> geomInstances;
            
            for (uint32_t idx = 0; idx < 3; ++idx)
            {
                // Create simple triangle
                std::vector<shared::Vertex> vertices(3);
                float offset = idx * 2.0f;
                vertices[0].position = Point3D(offset, 0, 0);
                vertices[1].position = Point3D(offset + 1, 0, 0);
                vertices[2].position = Point3D(offset + 0.5f, 1, 0);
                
                for (auto& v : vertices)
                {
                    v.normal = Normal3D(0, 0, 1);
                    v.texCoord = Point2D(0, 0);
                    v.texCoord0Dir = Vector3D(1, 0, 0);
                }
                
                std::vector<shared::Triangle> triangles(1);
                triangles[0].index0 = 0;
                triangles[0].index1 = 1;
                triangles[0].index2 = 2;
                
                // Create GeometryInstance
                auto geomInst = std::make_unique<GeometryInstance>();
                TriangleGeometry triGeom;
                triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
                triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
                geomInst->geometry = std::move(triGeom);
                geomInst->geomInstSlot = idx;
                
                // Create GeometryInstanceData
                const auto& geom = std::get<TriangleGeometry>(geomInst->geometry);
                shared::GeometryInstanceData geomInstData = {};
                geomInstData.vertexBuffer = geom.vertexBuffer.getROBuffer<shared::enableBufferOobCheck>();
                geomInstData.triangleBuffer = geom.triangleBuffer.getROBuffer<shared::enableBufferOobCheck>();
                geomInstData.materialSlot = idx;  // Different material per instance
                geomInstData.geomInstSlot = geomInst->geomInstSlot;
                
                // Store in our host-side array
                geomInstDataArray.push_back(geomInstData);
                
                geomInstances.push_back(std::move(geomInst));
            }
            
            // Verify the data contents
            for (uint32_t idx = 0; idx < 3; ++idx)
            {
                const auto& data = geomInstDataArray[idx];
                CHECK(data.geomInstSlot == idx);
                CHECK(data.materialSlot == idx);
                // ROBuffer provides getNumElements() method for verification
                CHECK(data.vertexBuffer.getNumElements() == 3);
                CHECK(data.triangleBuffer.getNumElements() == 1);
            }
            
            LOG(INFO) << "Created " << geomInstDataArray.size() << " GeometryInstanceData structures";
        }
        
        // Clean up GPU context
        gpuContext.cleanup();
    }
}

TEST_CASE("CgModel Data Integrity")
{
    SUBCASE("Verify triangle indices are valid")
    {
        std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
        auto model = loadGLTF(gltfPath);
        
        REQUIRE(model != nullptr);
        REQUIRE(model->S.size() > 0);
        
        const auto& surface = model->S[0];
        const auto& F = surface.indices();
        size_t vertexCount = model->vertexCount();
        
        // Check that all indices are within valid range
        for (int i = 0; i < F.cols(); ++i)
        {
            const auto& tri = F.col(i);
            CHECK(tri[0] < vertexCount);
            CHECK(tri[1] < vertexCount);
            CHECK(tri[2] < vertexCount);
        }
        
        // Check that we have the expected number of triangles
        CHECK(F.cols() == 12);
    }
    
    SUBCASE("Verify normal vectors are normalized")
    {
        std::filesystem::path gltfPath = "E:/common_content/models/Box/glTF/Box.gltf";
        auto model = loadGLTF(gltfPath);
        
        REQUIRE(model != nullptr);
        
        // Check that normals are approximately unit length
        for (int i = 0; i < model->N.cols(); ++i)
        {
            float length = model->N.col(i).norm();
            CHECK(length == doctest::Approx(1.0f).epsilon(0.01f));
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
