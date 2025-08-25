#include "Jahley.h"

const std::string APP_NAME = "AreaLightTest";

#ifdef CHECK
#undef CHECK
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;

#include <sabi_core/sabi_core.h>
#include <claude_core/excludeFromBuild/common/common_host.h>
#include <claude_core/excludeFromBuild/RenderContext.h>
#include <claude_core/excludeFromBuild/material/HostDisneyMaterial.h>

// We need to define these types since they're in the shared namespace
using shared::Vertex;
using shared::Triangle;

// Custom GeometryInstance for testing with OptiX Material
struct TestGeometryInstance
{
    optixu::Material optixMat;  // OptiX material from DisneyMaterialHandler
    uint32_t materialSlot;      // Material slot in the material buffer
    uint32_t geomInstSlot;
    optixu::GeometryInstance optixGeomInst;
    std::variant<TriangleGeometry, CurveGeometry, TFDMGeometry, NRTDSMGeometry> geometry;
    AABB aabb;

    void finalize()
    {
        optixGeomInst.destroy();
    }
};

TEST_CASE("Luminous CgModel to GeometryInstance with LightDistribution")
{
    SUBCASE("Create GeometryInstance from luminous rectangle with DisneyMaterialHandler")
    {
        // Create RenderContext which initializes all handlers including DisneyMaterialHandler
        auto renderContext = std::make_shared<RenderContext>();
        REQUIRE(renderContext != nullptr);
        
        bool initSuccess = renderContext->initialize();
        REQUIRE(initSuccess);
        
        // Get the DisneyMaterialHandler from the Handlers
        auto& handlers = renderContext->getHandlers();
        auto materialHandler = handlers.disneyMaterialHandler;
        REQUIRE(materialHandler != nullptr);
        
        CUcontext cudaContext = renderContext->getCudaContext();
        REQUIRE(cudaContext != nullptr);
        
        {
            // Create a luminous rectangle
            auto rectangle = sabi::MeshOps::createLuminousRectangle(4.0f, 3.0f, 
                                                                   Eigen::Vector3f(1.0f, 0.8f, 0.3f), 
                                                                   10.0f);
            REQUIRE(rectangle != nullptr);
            
            const auto& surface = rectangle->S[0];
            const auto& V = rectangle->V;
            const auto& cgMaterial = surface.cgMaterial;
            
            // Verify it's emissive
            CHECK(cgMaterial.emission.luminous > 0.0f);
            
            // Convert CgMaterial to Disney material using the handler
            optixu::Material optixMat = materialHandler->createDisneyMaterial(
                cgMaterial, 
                std::filesystem::path(), // Empty path for textures
                rectangle);
            CHECK(optixMat);
            
            // Material slot would be managed by the handler internally
            uint32_t materialSlot = 0; // In real usage, this would be tracked by SceneHandler
            
            // Create vertex buffer
            std::vector<shared::Vertex> vertices;
            vertices.reserve(V.cols());
            
            for (int i = 0; i < V.cols(); ++i)
            {
                shared::Vertex vertex;
                vertex.position = Point3D(V(0, i), V(1, i), V(2, i));
                vertex.normal = Normal3D(0, 0, 1); // Luminous rectangles face +Z
                vertex.texCoord = Point2D(0, 0);
                vertex.texCoord0Dir = Vector3D(1, 0, 0);
                vertices.push_back(vertex);
            }
            
            // Create triangle buffer
            std::vector<shared::Triangle> triangles;
            const auto& F = surface.F;
            triangles.reserve(F.cols());
            
            for (int i = 0; i < F.cols(); ++i)
            {
                shared::Triangle triangle;
                triangle.index0 = F(0, i);
                triangle.index1 = F(1, i);
                triangle.index2 = F(2, i);
                triangles.push_back(triangle);
            }
            
            // Create TestGeometryInstance
            auto geomInst = std::make_unique<TestGeometryInstance>();
            TriangleGeometry triGeom;
            
            triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
            triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
            
            geomInst->geometry = std::move(triGeom);
            geomInst->geomInstSlot = 0;
            
            // Set the OptiX material
            geomInst->optixMat = optixMat;
            geomInst->materialSlot = materialSlot;
            
            // Calculate AABB
            AABB geomAABB;
            geomAABB.minP = Point3D(V.row(0).minCoeff(), V.row(1).minCoeff(), V.row(2).minCoeff());
            geomAABB.maxP = Point3D(V.row(0).maxCoeff(), V.row(1).maxCoeff(), V.row(2).maxCoeff());
            geomInst->aabb = geomAABB;
            
            // Verify geometry instance
            CHECK(std::holds_alternative<TriangleGeometry>(geomInst->geometry));
            const auto& storedGeom = std::get<TriangleGeometry>(geomInst->geometry);
            CHECK(storedGeom.vertexBuffer.numElements() == 4);
            CHECK(storedGeom.triangleBuffer.numElements() == 2);
            
            // Verify material is created
            CHECK(geomInst->optixMat);
            // Disney material handles emission through the texture system
            
            // Create GeometryGroup
            // Note: GeometryGroup expects GeometryInstance pointers,
            // but we're using TestGeometryInstance for testing purposes
            // In real usage, this would be handled by SceneHandler
            std::set<const TestGeometryInstance*> testGeomInsts;
            testGeomInsts.insert(geomInst.get());
            // Verify test geometry instance collection
            CHECK(testGeomInsts.size() == 1);
            CHECK(triangles.size() == 2); // 2 emitter primitives
            
            // Create GeometryInstanceData with LightDistribution
            shared::GeometryInstanceData geomInstData = {};
            
            geomInstData.vertexBuffer = storedGeom.vertexBuffer.getROBuffer<shared::enableBufferOobCheck>();
            geomInstData.triangleBuffer = storedGeom.triangleBuffer.getROBuffer<shared::enableBufferOobCheck>();
            geomInstData.materialSlot = materialSlot;
            geomInstData.geomInstSlot = geomInst->geomInstSlot;
            
            // Create LightDistribution for emitter primitives
            // This would normally be filled by the scene handler
            // For now we'll create buffers that would be used to construct it
            std::vector<float> weights(triangles.size());
            std::vector<float> cdf(triangles.size());
            float integral = 0.0f;
            
            // Calculate area-based distribution weights for each triangle
            for (size_t i = 0; i < triangles.size(); ++i)
            {
                // Get triangle vertices
                const auto& tri = triangles[i];
                Point3D v0 = vertices[tri.index0].position;
                Point3D v1 = vertices[tri.index1].position;
                Point3D v2 = vertices[tri.index2].position;
                
                // Calculate triangle area
                Vector3D e1 = v1 - v0;
                Vector3D e2 = v2 - v0;
                Vector3D cross = Vector3D(
                    e1.y * e2.z - e1.z * e2.y,
                    e1.z * e2.x - e1.x * e2.z,
                    e1.x * e2.y - e1.y * e2.x
                );
                float area = 0.5f * std::sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
                
                // Weight by area and emission intensity
                weights[i] = area * cgMaterial.emission.luminous;
                integral += weights[i];
                cdf[i] = integral;
            }
            
            // Allocate GPU buffers for the distribution
            cudau::TypedBuffer<float> weightsBuffer;
            cudau::TypedBuffer<float> cdfBuffer;
            weightsBuffer.initialize(cudaContext, cudau::BufferType::Device, weights);
            cdfBuffer.initialize(cudaContext, cudau::BufferType::Device, cdf);
            
            // Create the DiscreteDistribution1D
            // Note: In actual usage, this would be constructed by the scene handler
            shared::LightDistribution emitterPrimDist(
                weightsBuffer.getDevicePointer(),
                cdfBuffer.getDevicePointer(),
                integral,
                static_cast<uint32_t>(triangles.size())
            );
            
            // Verify light distribution
            CHECK(emitterPrimDist.numValues() == 2);
            CHECK(emitterPrimDist.integral() > 0.0f);
            
            // Store the emitter primitive distribution
            geomInstData.emitterPrimDist = emitterPrimDist;
            
            LOG(INFO) << "Created GeometryInstance with " << triangles.size() << " emitter primitives";
            LOG(INFO) << "Light distribution with integral: " << emitterPrimDist.integral();
            LOG(INFO) << "Material slot: " << materialSlot;
        }
        
        renderContext->cleanup();
    }
    
    SUBCASE("Create GeometryGroup with multiple luminous geometries using DisneyMaterialHandler")
    {
        // Create RenderContext
        auto renderContext = std::make_shared<RenderContext>();
        REQUIRE(renderContext != nullptr);
        
        bool initSuccess = renderContext->initialize();
        REQUIRE(initSuccess);
        
        auto& handlers = renderContext->getHandlers();
        auto materialHandler = handlers.disneyMaterialHandler;
        REQUIRE(materialHandler != nullptr);
        
        CUcontext cudaContext = renderContext->getCudaContext();
        REQUIRE(cudaContext != nullptr);
        
        {
            std::vector<std::unique_ptr<TestGeometryInstance>> geomInstances;
            std::vector<cudau::TypedBuffer<float>> distBuffers; // Keep buffers alive
            
            // Create multiple luminous rectangles with different materials
            for (int idx = 0; idx < 3; ++idx)
            {
                auto rectangle = sabi::MeshOps::createLuminousRectangle(
                    2.0f + idx, 1.5f + idx * 0.5f,
                    Eigen::Vector3f(1.0f - idx * 0.2f, 0.5f + idx * 0.1f, idx * 0.3f),
                    5.0f + idx * 2.0f
                );
                
                REQUIRE(rectangle != nullptr);
                
                const auto& surface = rectangle->S[0];
                const auto& V = rectangle->V;
                const auto& cgMaterial = surface.cgMaterial;
                
                // Create Disney material
                optixu::Material optixMat = materialHandler->createDisneyMaterial(
                    cgMaterial,
                    std::filesystem::path(),
                    rectangle);
                CHECK(optixMat);
                
                uint32_t materialSlot = idx; // In real usage, tracked by SceneHandler
                
                // Create vertices
                std::vector<shared::Vertex> vertices;
                vertices.reserve(V.cols());
                
                for (int i = 0; i < V.cols(); ++i)
                {
                    shared::Vertex vertex;
                    vertex.position = Point3D(V(0, i) + idx * 3.0f, V(1, i), V(2, i));
                    vertex.normal = Normal3D(0, 0, 1);
                    vertex.texCoord = Point2D(0, 0);
                    vertex.texCoord0Dir = Vector3D(1, 0, 0);
                    vertices.push_back(vertex);
                }
                
                // Create triangles
                std::vector<shared::Triangle> triangles;
                const auto& F = surface.F;
                triangles.reserve(F.cols());
                
                for (int i = 0; i < F.cols(); ++i)
                {
                    shared::Triangle triangle;
                    triangle.index0 = F(0, i);
                    triangle.index1 = F(1, i);
                    triangle.index2 = F(2, i);
                    triangles.push_back(triangle);
                }
                
                // Create TestGeometryInstance
                auto geomInst = std::make_unique<TestGeometryInstance>();
                TriangleGeometry triGeom;
                
                triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
                triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
                
                geomInst->geometry = std::move(triGeom);
                geomInst->geomInstSlot = idx;
                geomInst->optixMat = optixMat;
                geomInst->materialSlot = materialSlot;
                
                // Calculate AABB
                AABB aabb;
                aabb.minP = Point3D(V.row(0).minCoeff() + idx * 3.0f, V.row(1).minCoeff(), V.row(2).minCoeff());
                aabb.maxP = Point3D(V.row(0).maxCoeff() + idx * 3.0f, V.row(1).maxCoeff(), V.row(2).maxCoeff());
                geomInst->aabb = aabb;
                
                // Create light distribution for this geometry
                std::vector<float> distribution(triangles.size());
                for (size_t i = 0; i < triangles.size(); ++i)
                {
                    // Simple uniform distribution for this test
                    distribution[i] = cgMaterial.emission.luminous;
                }
                
                distBuffers.emplace_back();
                distBuffers.back().initialize(cudaContext, cudau::BufferType::Device, distribution);
                
                geomInstances.push_back(std::move(geomInst));
            }
            
            // Create test collection with all instances
            std::set<const TestGeometryInstance*> testGeomInsts;
            
            uint32_t totalEmitterPrims = 0;
            AABB combinedAABB;
            combinedAABB.minP = Point3D(1e10f, 1e10f, 1e10f);
            combinedAABB.maxP = Point3D(-1e10f, -1e10f, -1e10f);
            
            for (const auto& inst : geomInstances)
            {
                testGeomInsts.insert(inst.get());
                
                // Count emitter primitives
                const auto& geom = std::get<TriangleGeometry>(inst->geometry);
                totalEmitterPrims += geom.triangleBuffer.numElements();
                
                // Update combined AABB
                combinedAABB.minP.x = std::min(combinedAABB.minP.x, inst->aabb.minP.x);
                combinedAABB.minP.y = std::min(combinedAABB.minP.y, inst->aabb.minP.y);
                combinedAABB.minP.z = std::min(combinedAABB.minP.z, inst->aabb.minP.z);
                combinedAABB.maxP.x = std::max(combinedAABB.maxP.x, inst->aabb.maxP.x);
                combinedAABB.maxP.y = std::max(combinedAABB.maxP.y, inst->aabb.maxP.y);
                combinedAABB.maxP.z = std::max(combinedAABB.maxP.z, inst->aabb.maxP.z);
            }
            
            // Verify test geometry instances
            CHECK(testGeomInsts.size() == 3);
            CHECK(totalEmitterPrims == 6); // 3 rectangles * 2 triangles each
            
            // Verify we have 3 different materials
            CHECK(geomInstances.size() == 3);
            
            LOG(INFO) << "Created test collection with " << testGeomInsts.size() << " instances";
            LOG(INFO) << "Total emitter primitives: " << totalEmitterPrims;
        }
        
        renderContext->cleanup();
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