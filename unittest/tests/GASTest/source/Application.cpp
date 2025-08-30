#include "Jahley.h"

const std::string APP_NAME = "GASTest";

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

// Use shared namespace types
using shared::Vertex;
using shared::Triangle;

// Helper function to create a simple triangle geometry
void createTriangleGeometry(
    CUcontext cudaContext,
    TriangleGeometry& triGeom,
    float xOffset = 0.0f)
{
    // Create a simple triangle
    std::vector<Vertex> vertices(3);
    vertices[0].position = Point3D(xOffset + 0.0f, 0.0f, 0.0f);
    vertices[1].position = Point3D(xOffset + 1.0f, 0.0f, 0.0f);
    vertices[2].position = Point3D(xOffset + 0.5f, 1.0f, 0.0f);
    
    for (auto& v : vertices) {
        v.normal = Normal3D(0, 0, 1);
        v.texCoord = Point2D(0, 0);
        v.texCoord0Dir = Vector3D(1, 0, 0);
    }
    
    std::vector<Triangle> triangles(1);
    triangles[0].index0 = 0;
    triangles[0].index1 = 1;
    triangles[0].index2 = 2;
    
    // Initialize buffers
    triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
    triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
}

// Helper function to create a cube geometry
void createCubeGeometry(
    CUcontext cudaContext,
    TriangleGeometry& triGeom)
{
    // Cube vertices
    std::vector<Vertex> vertices(24);  // 6 faces * 4 vertices per face
    
    // Front face (z = 0.5)
    vertices[0].position = Point3D(-0.5f, -0.5f,  0.5f);
    vertices[1].position = Point3D( 0.5f, -0.5f,  0.5f);
    vertices[2].position = Point3D( 0.5f,  0.5f,  0.5f);
    vertices[3].position = Point3D(-0.5f,  0.5f,  0.5f);
    
    // Back face (z = -0.5)
    vertices[4].position = Point3D(-0.5f, -0.5f, -0.5f);
    vertices[5].position = Point3D( 0.5f, -0.5f, -0.5f);
    vertices[6].position = Point3D( 0.5f,  0.5f, -0.5f);
    vertices[7].position = Point3D(-0.5f,  0.5f, -0.5f);
    
    // Left face (x = -0.5)
    vertices[8].position = Point3D(-0.5f, -0.5f, -0.5f);
    vertices[9].position = Point3D(-0.5f, -0.5f,  0.5f);
    vertices[10].position = Point3D(-0.5f,  0.5f,  0.5f);
    vertices[11].position = Point3D(-0.5f,  0.5f, -0.5f);
    
    // Right face (x = 0.5)
    vertices[12].position = Point3D(0.5f, -0.5f, -0.5f);
    vertices[13].position = Point3D(0.5f, -0.5f,  0.5f);
    vertices[14].position = Point3D(0.5f,  0.5f,  0.5f);
    vertices[15].position = Point3D(0.5f,  0.5f, -0.5f);
    
    // Top face (y = 0.5)
    vertices[16].position = Point3D(-0.5f, 0.5f, -0.5f);
    vertices[17].position = Point3D( 0.5f, 0.5f, -0.5f);
    vertices[18].position = Point3D( 0.5f, 0.5f,  0.5f);
    vertices[19].position = Point3D(-0.5f, 0.5f,  0.5f);
    
    // Bottom face (y = -0.5)
    vertices[20].position = Point3D(-0.5f, -0.5f, -0.5f);
    vertices[21].position = Point3D( 0.5f, -0.5f, -0.5f);
    vertices[22].position = Point3D( 0.5f, -0.5f,  0.5f);
    vertices[23].position = Point3D(-0.5f, -0.5f,  0.5f);
    
    // Set normals and texture coordinates
    for (int i = 0; i < 4; i++) {
        vertices[i].normal = Normal3D(0, 0, 1);      // Front
        vertices[i+4].normal = Normal3D(0, 0, -1);   // Back
        vertices[i+8].normal = Normal3D(-1, 0, 0);   // Left
        vertices[i+12].normal = Normal3D(1, 0, 0);   // Right
        vertices[i+16].normal = Normal3D(0, 1, 0);   // Top
        vertices[i+20].normal = Normal3D(0, -1, 0);  // Bottom
    }
    
    for (auto& v : vertices) {
        v.texCoord = Point2D(0, 0);
        v.texCoord0Dir = Vector3D(1, 0, 0);
    }
    
    // Cube indices (12 triangles, 2 per face)
    std::vector<Triangle> triangles(12);
    
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
    triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices);
    triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles);
}

TEST_CASE("GeometryAccelerationStructure Basic Operations")
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
    CHECK(scene);
    
    // Create stream
    CUstream stream;
    cuStreamCreate(&stream, 0);
    
    SUBCASE("Create and Configure GAS")
    {
        // Create a GAS
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        CHECK(gas);
        
        // Configure the GAS
        gas.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes,
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        
        // Set material set count
        gas.setMaterialSetCount(1);
        
        // Set ray type count for material set 0
        gas.setRayTypeCount(0, 1);
        
        // Verify configuration
        CHECK(gas.getMaterialSetCount() == 1);
        CHECK(gas.getRayTypeCount(0) == 1);
        
        // Clean up
        gas.destroy();
    }
    
    SUBCASE("Add GeometryInstance to GAS")
    {
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        
        // Create a geometry instance for triangles
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        CHECK(geomInst);
        
        // Create triangle geometry
        TriangleGeometry triGeom;
        createTriangleGeometry(cudaContext, triGeom);
        
        // Set geometry data using OptiX utility wrapper
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        // Configure GAS
        gas.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        
        // Add child to GAS
        gas.addChild(geomInst);
        
        // Mark dirty for rebuild
        gas.markDirty();
        
        // Prepare for build
        OptixAccelBufferSizes memReq;
        gas.prepareForBuild(&memReq);
        
        CHECK(memReq.outputSizeInBytes > 0);
        CHECK(memReq.tempSizeInBytes > 0);
        
        // Clean up
        geomInst.destroy();
        gas.destroy();
    }
    
    SUBCASE("Build and Rebuild GAS")
    {
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        
        // Create triangle geometry
        TriangleGeometry triGeom;
        createTriangleGeometry(cudaContext, triGeom);
        
        // Set geometry data
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        // Configure and add to GAS
        gas.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        gas.addChild(geomInst);
        gas.markDirty();
        
        // Prepare for build
        OptixAccelBufferSizes memReq;
        gas.prepareForBuild(&memReq);
        
        // Allocate buffers using cudau::Buffer
        cudau::Buffer accelBuffer;
        cudau::Buffer scratchBuffer;
        accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
        scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.tempSizeInBytes, 1);
        
        // Build the GAS
        OptixTraversableHandle traversable = gas.rebuild(stream, accelBuffer, scratchBuffer);
        CHECK(traversable != 0);
        
        // Wait for build to complete
        cuStreamSynchronize(stream);
        
        // Clean up
        accelBuffer.finalize();
        scratchBuffer.finalize();
        geomInst.destroy();
        gas.destroy();
    }
    
    SUBCASE("GAS with Multiple Children")
    {
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        
        // Create multiple geometry instances
        const int numInstances = 3;
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms(numInstances);
        
        for (int i = 0; i < numInstances; ++i) {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            
            // Create triangle geometry with offset
            createTriangleGeometry(cudaContext, triGeoms[i], i * 2.0f);
            
            // Set geometry data
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms[i].vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms[i].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            // Add to GAS
            gas.addChild(geomInst);
            geomInsts.push_back(geomInst);
        }
        
        // Configure GAS
        gas.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        gas.markDirty();
        
        // Get child count
        CHECK(gas.getChildCount() == numInstances);
        
        // Remove one child
        gas.removeChildAt(1);
        CHECK(gas.getChildCount() == numInstances - 1);
        
        // Clear all children
        gas.clearChildren();
        CHECK(gas.getChildCount() == 0);
        
        // Clean up
        for (auto& inst : geomInsts) {
            inst.destroy();
        }
        gas.destroy();
    }
    
    SUBCASE("GAS Compaction")
    {
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        
        // Create cube geometry (more complex than triangle)
        TriangleGeometry triGeom;
        createCubeGeometry(cudaContext, triGeom);
        
        // Set geometry data
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        // Configure for compaction
        gas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes,  // Enable compaction
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        gas.addChild(geomInst);
        gas.markDirty();
        
        // Prepare for build
        OptixAccelBufferSizes memReq;
        gas.prepareForBuild(&memReq);
        
        // Allocate buffers
        cudau::Buffer accelBuffer;
        cudau::Buffer scratchBuffer;
        accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
        scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.tempSizeInBytes, 1);
        
        // Build the GAS
        OptixTraversableHandle traversable = gas.rebuild(stream, accelBuffer, scratchBuffer);
        CHECK(traversable != 0);
        
        // Wait for build to complete
        cuStreamSynchronize(stream);
        
        // Prepare for compaction
        size_t compactedSize;
        gas.prepareForCompact(&compactedSize);
        CHECK(compactedSize > 0);
        CHECK(compactedSize <= memReq.outputSizeInBytes);
        
        // Allocate compacted buffer
        cudau::Buffer compactedBuffer;
        compactedBuffer.initialize(cudaContext, cudau::BufferType::Device, compactedSize, 1);
        
        // Perform compaction
        OptixTraversableHandle compactedTraversable = gas.compact(stream, compactedBuffer);
        CHECK(compactedTraversable != 0);
        
        // Wait for compaction to complete
        cuStreamSynchronize(stream);
        
        LOG(INFO) << "GAS compaction: " << memReq.outputSizeInBytes << " bytes -> " << compactedSize << " bytes";
        
        // Clean up
        accelBuffer.finalize();
        scratchBuffer.finalize();
        compactedBuffer.finalize();
        geomInst.destroy();
        gas.destroy();
    }
    
    SUBCASE("Motion Blur Configuration")
    {
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        
        // Configure motion options
        uint32_t numKeys = 2;  // Two motion steps
        float timeBegin = 0.0f;
        float timeEnd = 1.0f;
        OptixMotionFlags motionFlags = OPTIX_MOTION_FLAG_NONE;
        
        gas.setMotionOptions(numKeys, timeBegin, timeEnd, motionFlags);
        
        // Create geometry instance with motion
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        geomInst.setMotionStepCount(numKeys);
        
        // Create vertex buffers for each motion step
        std::vector<TriangleGeometry> motionGeoms(numKeys);
        for (uint32_t step = 0; step < numKeys; ++step) {
            // Create triangle with different positions for each step
            createTriangleGeometry(cudaContext, motionGeoms[step], step * 1.0f);
            
            // Set vertex buffer for this motion step
            geomInst.setVertexBuffer(motionGeoms[step].vertexBuffer, step);
            
            // Triangle buffer only needs to be set once
            if (step == 0) {
                geomInst.setTriangleBuffer(motionGeoms[step].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            }
        }
        
        // Add to GAS and configure
        gas.addChild(geomInst);
        gas.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        gas.markDirty();
        
        // Prepare for build
        OptixAccelBufferSizes memReq;
        gas.prepareForBuild(&memReq);
        
        CHECK(memReq.outputSizeInBytes > 0);
        CHECK(memReq.tempSizeInBytes > 0);
        
        // Clean up
        geomInst.destroy();
        gas.destroy();
    }
    
    SUBCASE("Material Set and Ray Type Configuration")
    {
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        
        // Configure multiple material sets
        uint32_t numMaterialSets = 3;
        gas.setMaterialSetCount(numMaterialSets);
        
        // Configure ray types for each material set
        gas.setRayTypeCount(0, 2);  // Material set 0: 2 ray types
        gas.setRayTypeCount(1, 1);  // Material set 1: 1 ray type
        gas.setRayTypeCount(2, 3);  // Material set 2: 3 ray types
        
        // Verify configuration took effect
        CHECK(gas.getMaterialSetCount() == numMaterialSets);
        CHECK(gas.getRayTypeCount(0) == 2);
        CHECK(gas.getRayTypeCount(1) == 1);
        CHECK(gas.getRayTypeCount(2) == 3);
        
        // Clean up
        gas.destroy();
    }
    
    SUBCASE("Per-Child User Data")
    {
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        
        // User data structure
        struct UserData {
            uint32_t id;
            float value;
        };
        
        // Create geometry instances with user data
        const int numInstances = 2;
        std::vector<optixu::GeometryInstance> geomInsts;
        std::vector<TriangleGeometry> triGeoms(numInstances);
        
        for (int i = 0; i < numInstances; ++i) {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            
            createTriangleGeometry(cudaContext, triGeoms[i], i * 2.0f);
            
            geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            geomInst.setVertexBuffer(triGeoms[i].vertexBuffer);
            geomInst.setTriangleBuffer(triGeoms[i].triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
            
            // Add child with user data
            UserData userData = {static_cast<uint32_t>(i), static_cast<float>(i) * 10.0f};
            gas.addChild(geomInst, 0, userData);
            
            geomInsts.push_back(geomInst);
        }
        
        gas.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        gas.markDirty();
        
        CHECK(gas.getChildCount() == numInstances);
        
        // Clean up
        for (auto& inst : geomInsts) {
            inst.destroy();
        }
        gas.destroy();
    }
    
    // Clean up CUDA resources
    cuStreamDestroy(stream);
    scene.destroy();
    gpuContext.cleanup();
}

TEST_CASE("GeometryAccelerationStructure Update Operations")
{
    // Initialize GPU context
    GPUContext gpuContext;
    bool initSuccess = gpuContext.initialize();
    REQUIRE(initSuccess);
    
    CUcontext cudaContext = gpuContext.getCudaContext();
    optixu::Context optixContext = gpuContext.getOptiXContext();
    REQUIRE(cudaContext != nullptr);
    
    optixu::Scene scene = optixContext.createScene();
    CUstream stream;
    cuStreamCreate(&stream, 0);
    
    SUBCASE("GAS Update After Vertex Modification")
    {
        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
        optixu::GeometryInstance geomInst = scene.createGeometryInstance();
        
        // Create initial triangle geometry
        TriangleGeometry triGeom;
        createTriangleGeometry(cudaContext, triGeom);
        
        // Configure for update
        gas.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::Yes,  // Enable updates
            optixu::AllowCompaction::No,
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        
        geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        geomInst.setTriangleBuffer(triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
        
        gas.addChild(geomInst);
        gas.markDirty();
        
        // Initial build
        OptixAccelBufferSizes memReq;
        gas.prepareForBuild(&memReq);
        
        cudau::Buffer accelBuffer;
        cudau::Buffer scratchBuffer;
        accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
        scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.tempUpdateSizeInBytes, 1); // Use update size
        
        OptixTraversableHandle traversable = gas.rebuild(stream, accelBuffer, scratchBuffer);
        CHECK(traversable != 0);
        cuStreamSynchronize(stream);
        
        // Modify vertices
        std::vector<Vertex> newVertices(3);
        newVertices[0].position = Point3D(0.0f, 1.0f, 0.0f);  // Move up
        newVertices[1].position = Point3D(1.0f, 1.0f, 0.0f);
        newVertices[2].position = Point3D(0.5f, 2.0f, 0.0f);
        
        for (auto& v : newVertices) {
            v.normal = Normal3D(0, 0, 1);
            v.texCoord = Point2D(0, 0);
            v.texCoord0Dir = Vector3D(1, 0, 0);
        }
        
        // Update vertex buffer - reinitialize with new data
        triGeom.vertexBuffer.finalize();
        triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, newVertices);
        geomInst.setVertexBuffer(triGeom.vertexBuffer);
        
        // Update GAS (not rebuild, just update)
        gas.update(stream, scratchBuffer);
        cuStreamSynchronize(stream);
        
        // Clean up
        accelBuffer.finalize();
        scratchBuffer.finalize();
        geomInst.destroy();
        gas.destroy();
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