#include "Jahley.h"

const std::string APP_NAME = "GASHandlerTest";

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
#include "GASHandler.h"

// Use shared namespace types
using shared::Vertex;
using shared::Triangle;

// Helper function to create a simple triangle geometry
void createTriangleGeometry (
    CUcontext cudaContext,
    TriangleGeometry& triGeom,
    float xOffset = 0.0f)
{
    // Create a simple triangle
    std::vector<Vertex> vertices (3);
    vertices[0].position = Point3D (xOffset + 0.0f, 0.0f, 0.0f);
    vertices[1].position = Point3D (xOffset + 1.0f, 0.0f, 0.0f);
    vertices[2].position = Point3D (xOffset + 0.5f, 1.0f, 0.0f);

    for (auto& v : vertices)
    {
        v.normal = Normal3D (0, 0, 1);
        v.texCoord = Point2D (0, 0);
        v.texCoord0Dir = Vector3D (1, 0, 0);
    }

    std::vector<Triangle> triangles (1);
    triangles[0].index0 = 0;
    triangles[0].index1 = 1;
    triangles[0].index2 = 2;

    // Initialize buffers
    triGeom.vertexBuffer.initialize (cudaContext, cudau::BufferType::Device, vertices);
    triGeom.triangleBuffer.initialize (cudaContext, cudau::BufferType::Device, triangles);
}

// Helper function to create a cube geometry
void createCubeGeometry (
    CUcontext cudaContext,
    TriangleGeometry& triGeom)
{
    // Cube vertices
    std::vector<Vertex> vertices (24); // 6 faces * 4 vertices per face

    // Front face (z = 0.5)
    vertices[0].position = Point3D (-0.5f, -0.5f, 0.5f);
    vertices[1].position = Point3D (0.5f, -0.5f, 0.5f);
    vertices[2].position = Point3D (0.5f, 0.5f, 0.5f);
    vertices[3].position = Point3D (-0.5f, 0.5f, 0.5f);

    // Back face (z = -0.5)
    vertices[4].position = Point3D (-0.5f, -0.5f, -0.5f);
    vertices[5].position = Point3D (0.5f, -0.5f, -0.5f);
    vertices[6].position = Point3D (0.5f, 0.5f, -0.5f);
    vertices[7].position = Point3D (-0.5f, 0.5f, -0.5f);

    // Left face (x = -0.5)
    vertices[8].position = Point3D (-0.5f, -0.5f, -0.5f);
    vertices[9].position = Point3D (-0.5f, -0.5f, 0.5f);
    vertices[10].position = Point3D (-0.5f, 0.5f, 0.5f);
    vertices[11].position = Point3D (-0.5f, 0.5f, -0.5f);

    // Right face (x = 0.5)
    vertices[12].position = Point3D (0.5f, -0.5f, -0.5f);
    vertices[13].position = Point3D (0.5f, -0.5f, 0.5f);
    vertices[14].position = Point3D (0.5f, 0.5f, 0.5f);
    vertices[15].position = Point3D (0.5f, 0.5f, -0.5f);

    // Top face (y = 0.5)
    vertices[16].position = Point3D (-0.5f, 0.5f, -0.5f);
    vertices[17].position = Point3D (0.5f, 0.5f, -0.5f);
    vertices[18].position = Point3D (0.5f, 0.5f, 0.5f);
    vertices[19].position = Point3D (-0.5f, 0.5f, 0.5f);

    // Bottom face (y = -0.5)
    vertices[20].position = Point3D (-0.5f, -0.5f, -0.5f);
    vertices[21].position = Point3D (0.5f, -0.5f, -0.5f);
    vertices[22].position = Point3D (0.5f, -0.5f, 0.5f);
    vertices[23].position = Point3D (-0.5f, -0.5f, 0.5f);

    // Set normals and texture coordinates
    for (int i = 0; i < 4; i++)
    {
        vertices[i].normal = Normal3D (0, 0, 1);       // Front
        vertices[i + 4].normal = Normal3D (0, 0, -1);  // Back
        vertices[i + 8].normal = Normal3D (-1, 0, 0);  // Left
        vertices[i + 12].normal = Normal3D (1, 0, 0);  // Right
        vertices[i + 16].normal = Normal3D (0, 1, 0);  // Top
        vertices[i + 20].normal = Normal3D (0, -1, 0); // Bottom
    }

    for (auto& v : vertices)
    {
        v.texCoord = Point2D (0, 0);
        v.texCoord0Dir = Vector3D (1, 0, 0);
    }

    // Cube indices (12 triangles, 2 per face)
    std::vector<Triangle> triangles (12);

    // Front face
    triangles[0] = {0, 1, 2};
    triangles[1] = {2, 3, 0};

    // Back face
    triangles[2] = {5, 4, 7};
    triangles[3] = {7, 6, 5};

    // Left face
    triangles[4] = {8, 9, 10};
    triangles[5] = {10, 11, 8};

    // Right face
    triangles[6] = {13, 12, 15};
    triangles[7] = {15, 14, 13};

    // Top face
    triangles[8] = {16, 17, 18};
    triangles[9] = {18, 19, 16};

    // Bottom face
    triangles[10] = {21, 20, 23};
    triangles[11] = {23, 22, 21};

    // Initialize buffers
    triGeom.vertexBuffer.initialize (cudaContext, cudau::BufferType::Device, vertices);
    triGeom.triangleBuffer.initialize (cudaContext, cudau::BufferType::Device, triangles);
}


TEST_CASE("GASHandler Basic Operations")
{
    // Initialize GPU context
    GPUContext gpuContext;
    bool initSuccess = gpuContext.initialize();
    REQUIRE(initSuccess);
    
    CUcontext cudaContext = gpuContext.getCudaContext();
    optixu::Context optixContext = gpuContext.getOptiXContext();
    REQUIRE(cudaContext != nullptr);
    
    // Create a scene
    optixu::Scene scene = optixContext.createScene();
    REQUIRE(scene);
    
    // Create stream
    CUstream stream;
    cuStreamCreate(&stream, 0);
    
    SUBCASE("Create and Initialize GASHandler")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        CHECK(gasHandler != nullptr);
        
        gasHandler->initialize();
        
        // Check initial state
        CHECK(gasHandler->getGASCount() == 0);
        CHECK(gasHandler->getTotalMemoryUsage() == 0);
        
        auto ids = gasHandler->getAllGASIds();
        CHECK(ids.empty());
        
        gasHandler->finalize();
    }
    
    SUBCASE("Create Single GAS")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create geometry instance
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        TriangleGeometry triGeom;
        createTriangleGeometry(cudaContext, triGeom);
        
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        // Create GAS
        uint32_t gasId = gasHandler->createGAS(geomInst);
        CHECK(gasId > 0);
        CHECK(gasHandler->hasGAS(gasId));
        CHECK(gasHandler->getGASCount() == 1);
        
        // GAS should be dirty (not built yet)
        CHECK(gasHandler->isDirty(gasId));
        
        // Build the GAS
        gasHandler->buildGAS(gasId, stream);
        CHECK(!gasHandler->isDirty(gasId));
        
        // Get traversable handle
        OptixTraversableHandle handle = gasHandler->getTraversableHandle(gasId);
        CHECK(handle != 0);
        
        // Check memory usage
        CHECK(gasHandler->getTotalMemoryUsage() > 0);
        CHECK(gasHandler->getGASMemoryUsage(gasId) > 0);
        
        // Clean up
        geomInst.destroy();
        gasHandler->finalize();
    }
    
    SUBCASE("Create Multiple GAS from Multiple GeometryInstances")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create multiple geometry instances for a single GAS
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms(3);
        
        for (int i = 0; i < 3; ++i)
        {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            createTriangleGeometry(cudaContext, triGeoms[i], i * 2.0f);
            
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms[i].vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms[i].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            geomInsts.push_back(geomInst);
        }
        
        // Create GAS with multiple geometry instances
        uint32_t gasId = gasHandler->createGAS(geomInsts);
        CHECK(gasId > 0);
        CHECK(gasHandler->getGASCount() == 1);
        
        // Build and check
        gasHandler->buildGAS(gasId, stream);
        CHECK(gasHandler->getTraversableHandle(gasId) != 0);
        
        // Clean up
        for (auto& inst : geomInsts)
        {
            inst.destroy();
        }
        gasHandler->finalize();
    }
    
    SUBCASE("Remove GAS")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create and build a GAS
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        TriangleGeometry triGeom;
        createTriangleGeometry(cudaContext, triGeom);
        
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        uint32_t gasId = gasHandler->createGAS(geomInst);
        gasHandler->buildGAS(gasId, stream);
        
        size_t memBefore = gasHandler->getTotalMemoryUsage();
        CHECK(memBefore > 0);
        
        // Remove the GAS
        gasHandler->removeGAS(gasId);
        CHECK(!gasHandler->hasGAS(gasId));
        CHECK(gasHandler->getGASCount() == 0);
        CHECK(gasHandler->getTotalMemoryUsage() == 0);
        
        // Clean up
        geomInst.destroy();
        gasHandler->finalize();
    }
    
    // Clean up CUDA resources
    cuStreamDestroy(stream);
    scene.destroy();
    gpuContext.cleanup();
}

TEST_CASE("GASHandler Batch Operations")
{
    GPUContext gpuContext;
    bool initSuccess = gpuContext.initialize();
    REQUIRE(initSuccess);
    
    CUcontext cudaContext = gpuContext.getCudaContext();
    optixu::Context optixContext = gpuContext.getOptiXContext();
    optixu::Scene scene = optixContext.createScene();
    CUstream stream;
    cuStreamCreate(&stream, 0);
    
    SUBCASE("Build All Dirty")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create multiple GAS objects
        std::vector<uint32_t> gasIds;
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms(5);
        
        for (int i = 0; i < 5; ++i)
        {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            createTriangleGeometry(cudaContext, triGeoms[i], i * 2.0f);
            
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms[i].vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms[i].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            uint32_t gasId = gasHandler->createGAS(geomInst);
            gasIds.push_back(gasId);
            geomInsts.push_back(geomInst);
        }
        
        // All should be dirty
        for (uint32_t id : gasIds)
        {
            CHECK(gasHandler->isDirty(id));
        }
        
        // Build all dirty at once
        gasHandler->buildAllDirty(stream);
        
        // None should be dirty now
        for (uint32_t id : gasIds)
        {
            CHECK(!gasHandler->isDirty(id));
            CHECK(gasHandler->getTraversableHandle(id) != 0);
        }
        
        // Memory should be allocated
        CHECK(gasHandler->getTotalMemoryUsage() > 0);
        
        // Clean up
        for (auto& inst : geomInsts)
        {
            inst.destroy();
        }
        gasHandler->finalize();
    }
    
    SUBCASE("Rebuild All")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create and build some GAS objects
        std::vector<uint32_t> gasIds;
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms(3);
        
        for (int i = 0; i < 3; ++i)
        {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            createTriangleGeometry(cudaContext, triGeoms[i]);
            
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms[i].vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms[i].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            uint32_t gasId = gasHandler->createGAS(geomInst);
            gasIds.push_back(gasId);
            geomInsts.push_back(geomInst);
        }
        
        // Build all initially
        gasHandler->buildAllDirty(stream);
        
        // Verify none are dirty
        for (uint32_t id : gasIds)
        {
            CHECK(!gasHandler->isDirty(id));
        }
        
        // Rebuild all (even though not dirty)
        gasHandler->rebuildAll(stream);
        
        // Should still not be dirty after rebuild
        for (uint32_t id : gasIds)
        {
            CHECK(!gasHandler->isDirty(id));
            CHECK(gasHandler->getTraversableHandle(id) != 0);
        }
        
        // Clean up
        for (auto& inst : geomInsts)
        {
            inst.destroy();
        }
        gasHandler->finalize();
    }
    
    SUBCASE("Compact All")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create GAS with compaction enabled
        std::vector<uint32_t> gasIds;
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms(3);
        
        for (int i = 0; i < 3; ++i)
        {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            createCubeGeometry(cudaContext, triGeoms[i]); // Use cube for more complex geometry
            
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms[i].vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms[i].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            // Create with compaction enabled
            uint32_t gasId = gasHandler->createGAS(
                geomInst,
                optixu::ASTradeoff::PreferFastTrace,
                false,  // allowUpdate
                true);  // allowCompaction
            gasIds.push_back(gasId);
            geomInsts.push_back(geomInst);
        }
        
        // Build all
        gasHandler->buildAllDirty(stream);
        
        size_t memoryBefore = gasHandler->getTotalMemoryUsage();
        
        // Compact all
        gasHandler->compactAll(stream);
        
        size_t memoryAfter = gasHandler->getTotalMemoryUsage();
        
        // Memory usage should be the same or less after compaction
        CHECK(memoryAfter <= memoryBefore);
        
        // Traversable handles should still be valid
        for (uint32_t id : gasIds)
        {
            CHECK(gasHandler->getTraversableHandle(id) != 0);
        }
        
        // Clean up
        for (auto& inst : geomInsts)
        {
            inst.destroy();
        }
        gasHandler->finalize();
    }
    
    // Clean up
    cuStreamDestroy(stream);
    scene.destroy();
    gpuContext.cleanup();
}

TEST_CASE("GASHandler Configuration")
{
    GPUContext gpuContext;
    bool initSuccess = gpuContext.initialize();
    REQUIRE(initSuccess);
    
    CUcontext cudaContext = gpuContext.getCudaContext();
    optixu::Context optixContext = gpuContext.getOptiXContext();
    optixu::Scene scene = optixContext.createScene();
    CUstream stream;
    cuStreamCreate(&stream, 0);
    
    SUBCASE("Material Configuration")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create a GAS
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        TriangleGeometry triGeom;
        createTriangleGeometry(cudaContext, triGeom);
        
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        uint32_t gasId = gasHandler->createGAS(geomInst);
        
        // Configure materials
        std::vector<uint32_t> rayTypeCounts = {2, 1, 3}; // 3 material sets with different ray type counts
        gasHandler->setMaterialConfiguration(gasId, 3, rayTypeCounts);
        
        // Build (configuration changes require rebuild)
        gasHandler->buildGAS(gasId, stream);
        CHECK(gasHandler->getTraversableHandle(gasId) != 0);
        
        // Clean up
        geomInst.destroy();
        gasHandler->finalize();
    }
    
    SUBCASE("Motion Blur Configuration")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create geometry instance with motion
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        geomInst.setMotionStepCount(2);
        
        // Create vertex buffers for each motion step
        std::vector<TriangleGeometry> motionGeoms(2);
        for (uint32_t step = 0; step < 2; ++step)
        {
            createTriangleGeometry(cudaContext, motionGeoms[step], step * 1.0f);
            geomInst.setVertexBuffer(motionGeoms[step].vertexBuffer, step);
            
            if (step == 0)
            {
                geomInst.setTriangleBuffer(motionGeoms[step].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            }
        }
        
        uint32_t gasId = gasHandler->createGAS(geomInst);
        
        // Configure motion blur
        gasHandler->setMotionOptions(gasId, 2, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        
        // Build
        gasHandler->buildGAS(gasId, stream);
        CHECK(gasHandler->getTraversableHandle(gasId) != 0);
        
        // Clean up
        geomInst.destroy();
        gasHandler->finalize();
    }
    
    SUBCASE("Different Build Configurations")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms(3);
        std::vector<uint32_t> gasIds;
        
        // Create GAS with different configurations
        for (int i = 0; i < 3; ++i)
        {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            createTriangleGeometry(cudaContext, triGeoms[i]);
            
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms[i].vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms[i].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            // Different configurations for each
            optixu::ASTradeoff tradeoff = (i == 0) ? optixu::ASTradeoff::PreferFastTrace : optixu::ASTradeoff::PreferFastBuild;
            bool allowUpdate = (i == 1);
            bool allowCompaction = (i == 2);
            
            uint32_t gasId = gasHandler->createGAS(geomInst, tradeoff, allowUpdate, allowCompaction);
            gasIds.push_back(gasId);
            geomInsts.push_back(geomInst);
        }
        
        // Build all
        gasHandler->buildAllDirty(stream);
        
        // All should have valid handles
        for (uint32_t id : gasIds)
        {
            CHECK(gasHandler->getTraversableHandle(id) != 0);
        }
        
        // Clean up
        for (auto& inst : geomInsts)
        {
            inst.destroy();
        }
        gasHandler->finalize();
    }
    
    // Clean up
    cuStreamDestroy(stream);
    scene.destroy();
    gpuContext.cleanup();
}

TEST_CASE("GASHandler Memory Management")
{
    GPUContext gpuContext;
    bool initSuccess = gpuContext.initialize();
    REQUIRE(initSuccess);
    
    CUcontext cudaContext = gpuContext.getCudaContext();
    optixu::Context optixContext = gpuContext.getOptiXContext();
    optixu::Scene scene = optixContext.createScene();
    CUstream stream;
    cuStreamCreate(&stream, 0);
    
    SUBCASE("Memory Tracking")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        CHECK(gasHandler->getTotalMemoryUsage() == 0);
        
        // Create and build multiple GAS objects
        std::vector<uint32_t> gasIds;
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms;
        
        size_t expectedTotalMemory = 0;
        
        for (int i = 0; i < 3; ++i)
        {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            triGeoms.emplace_back();
            
            // Use different geometry sizes
            if (i == 0)
                createTriangleGeometry(cudaContext, triGeoms.back());
            else
                createCubeGeometry(cudaContext, triGeoms.back());
            
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms.back().vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms.back().triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            uint32_t gasId = gasHandler->createGAS(geomInst);
            gasHandler->buildGAS(gasId, stream);
            
            size_t gasMemory = gasHandler->getGASMemoryUsage(gasId);
            CHECK(gasMemory > 0);
            expectedTotalMemory += gasMemory;
            
            gasIds.push_back(gasId);
            geomInsts.push_back(geomInst);
        }
        
        // Total memory should match sum of individual GAS memories
        CHECK(gasHandler->getTotalMemoryUsage() == expectedTotalMemory);
        
        // Remove one GAS and check memory is updated
        size_t removedMemory = gasHandler->getGASMemoryUsage(gasIds[0]);
        gasHandler->removeGAS(gasIds[0]);
        CHECK(gasHandler->getTotalMemoryUsage() == expectedTotalMemory - removedMemory);
        
        // Clean up
        for (auto& inst : geomInsts)
        {
            inst.destroy();
        }
        gasHandler->finalize();
    }
    
    SUBCASE("Scratch Buffer Reuse")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Get initial scratch buffer
        cudau::Buffer& scratch = gasHandler->getScratchBuffer();
        size_t initialSize = scratch.sizeInBytes();
        CHECK(initialSize > 0); // Should have minimum size
        
        // Create a large GAS that might need bigger scratch
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        TriangleGeometry triGeom;
        createCubeGeometry(cudaContext, triGeom);
        
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        uint32_t gasId = gasHandler->createGAS(geomInst);
        gasHandler->buildGAS(gasId, stream);
        
        // Scratch buffer might have grown
        size_t afterBuildSize = scratch.sizeInBytes();
        CHECK(afterBuildSize >= initialSize);
        
        // Build another GAS - should reuse the same scratch buffer
        optixu::GeometryInstance geomInst2 = scene.createGeometryInstance();
        TriangleGeometry triGeom2;
        createTriangleGeometry(cudaContext, triGeom2);
        
        geomInst2.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst2.setVertexBuffer(triGeom2.vertexBuffer);
        geomInst2.setTriangleBuffer(triGeom2.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        uint32_t gasId2 = gasHandler->createGAS(geomInst2);
        gasHandler->buildGAS(gasId2, stream);
        
        // Scratch buffer should not shrink
        CHECK(scratch.sizeInBytes() >= afterBuildSize);
        
        // Clean up
        geomInst.destroy();
        geomInst2.destroy();
        gasHandler->finalize();
    }
    
    // Clean up
    cuStreamDestroy(stream);
    scene.destroy();
    gpuContext.cleanup();
}

TEST_CASE("GASHandler Edge Cases")
{
    GPUContext gpuContext;
    bool initSuccess = gpuContext.initialize();
    REQUIRE(initSuccess);
    
    CUcontext cudaContext = gpuContext.getCudaContext();
    optixu::Context optixContext = gpuContext.getOptiXContext();
    optixu::Scene scene = optixContext.createScene();
    CUstream stream;
    cuStreamCreate(&stream, 0);
    
    SUBCASE("Empty GAS Creation")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Try to create GAS with empty geometry list
        std::vector<optixu::GeometryInstance> emptyList;
        uint32_t gasId = gasHandler->createGAS(emptyList);
        CHECK(gasId == 0); // Should return invalid ID
        CHECK(gasHandler->getGASCount() == 0);
    }
    
    SUBCASE("Invalid GAS Operations")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        uint32_t invalidId = 999;
        
        // Operations on non-existent GAS should not crash
        CHECK(!gasHandler->hasGAS(invalidId));
        CHECK(gasHandler->getGASMemoryUsage(invalidId) == 0);
        CHECK(gasHandler->getTraversableHandle(invalidId) == 0);
        CHECK(!gasHandler->isDirty(invalidId));
        
        gasHandler->buildGAS(invalidId, stream); // Should not crash
        gasHandler->compactGAS(invalidId, stream); // Should not crash
        gasHandler->markDirty(invalidId); // Should not crash
        gasHandler->removeGAS(invalidId); // Should not crash
        
        gasHandler->finalize();
    }
    
    SUBCASE("Find GAS Containing GeometryInstance")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create multiple GAS with different geometry
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms(3);
        std::vector<uint32_t> gasIds;
        
        for (int i = 0; i < 3; ++i)
        {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            createTriangleGeometry(cudaContext, triGeoms[i], i * 2.0f);
            
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms[i].vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms[i].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            uint32_t gasId = gasHandler->createGAS(geomInst);
            gasIds.push_back(gasId);
            geomInsts.push_back(geomInst);
        }
        
        // Find each geometry instance
        for (size_t i = 0; i < geomInsts.size(); ++i)
        {
            uint32_t foundId = gasHandler->findGASContaining(geomInsts[i]);
            CHECK(foundId == gasIds[i]);
        }
        
        // Try to find non-existent geometry instance
        optixu::GeometryInstance nonExistent = scene.createGeometryInstance();
        uint32_t notFoundId = gasHandler->findGASContaining(nonExistent);
        CHECK(notFoundId == 0);
        
        // Clean up
        for (auto& inst : geomInsts)
        {
            inst.destroy();
        }
        nonExistent.destroy();
        gasHandler->finalize();
    }
    
    SUBCASE("Mark Dirty and Rebuild")
    {
        GASHandlerPtr gasHandler = GASHandler::create(scene, cudaContext);
        gasHandler->initialize();
        
        // Create and build a GAS
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        TriangleGeometry triGeom;
        createTriangleGeometry(cudaContext, triGeom);
        
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        uint32_t gasId = gasHandler->createGAS(geomInst);
        gasHandler->buildGAS(gasId, stream);
        CHECK(!gasHandler->isDirty(gasId));
        
        // Mark as dirty
        gasHandler->markDirty(gasId);
        CHECK(gasHandler->isDirty(gasId));
        
        // Building again should clear dirty flag
        gasHandler->buildGAS(gasId, stream);
        CHECK(!gasHandler->isDirty(gasId));
        
        // Clean up
        geomInst.destroy();
        gasHandler->finalize();
    }
    
    // Clean up
    cuStreamDestroy(stream);
    scene.destroy();
    gpuContext.cleanup();
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