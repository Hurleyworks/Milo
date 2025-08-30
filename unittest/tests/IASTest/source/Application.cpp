#include "Jahley.h"

const std::string APP_NAME = "IASTest";

#ifdef CHECK
#undef CHECK
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;

#include <algorithm>  // for std::max

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

// Helper structures to keep GAS and its buffers alive
struct GASWithBuffers {
    optixu::GeometryAccelerationStructure gas;
    optixu::GeometryInstance geomInst;
    cudau::Buffer accelBuffer;
    cudau::Buffer scratchBuffer;
    TriangleGeometry triGeom;
    
    // Default constructor
    GASWithBuffers() = default;
    
    // Move constructor
    GASWithBuffers(GASWithBuffers&& other) = default;
    
    // Move assignment
    GASWithBuffers& operator=(GASWithBuffers&& other) = default;
    
    // Delete copy operations
    GASWithBuffers(const GASWithBuffers&) = delete;
    GASWithBuffers& operator=(const GASWithBuffers&) = delete;
};

// Helper function to create a simple GAS
GASWithBuffers createSimpleGAS(
    optixu::Scene& scene,
    optixu::Context& optixContext,
    CUcontext cudaContext,
    CUstream stream,
    float xOffset = 0.0f)
{
    GASWithBuffers result;
    result.gas = scene.createGeometryAccelerationStructure();
    result.geomInst = scene.createGeometryInstance();
    
    // Create triangle geometry
    createTriangleGeometry(cudaContext, result.triGeom, xOffset);
    
    // Set geometry data
    result.geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
    result.geomInst.setVertexBuffer(result.triGeom.vertexBuffer);
    result.geomInst.setTriangleBuffer(result.triGeom.triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
    
    // Create and set a simple material
    optixu::Material mat = optixContext.createMaterial();
    result.geomInst.setMaterialCount(1, optixu::BufferView(), optixu::IndexSize::k4Bytes);
    result.geomInst.setMaterial(0, 0, mat);
    result.geomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
    
    // Configure and add to GAS
    result.gas.setConfiguration(
        optixu::ASTradeoff::PreferFastBuild,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::No,
        optixu::AllowRandomVertexAccess::No,
        optixu::AllowOpacityMicroMapUpdate::No,
        optixu::AllowDisableOpacityMicroMaps::No);
    result.gas.setMaterialSetCount(1);
    result.gas.setRayTypeCount(0, 1);  // Set ray type count for material set 0
    result.gas.addChild(result.geomInst);
    result.gas.markDirty();
    
    // Build the GAS
    OptixAccelBufferSizes memReq;
    result.gas.prepareForBuild(&memReq);
    
    result.accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
    result.scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.tempSizeInBytes, 1);
    
    result.gas.rebuild(stream, result.accelBuffer, result.scratchBuffer);
    cuStreamSynchronize(stream);
    
    return result;
}

TEST_CASE("InstanceAccelerationStructure Basic Operations")
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
    
    SUBCASE("Create and Configure IAS")
    {
        // Create an IAS
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        CHECK(ias);
        
        // Configure the IAS
        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes,
            optixu::AllowRandomInstanceAccess::No);
        
        // Verify configuration
        optixu::ASTradeoff tradeoff;
        optixu::AllowUpdate allowUpdate = optixu::AllowUpdate::No;
        optixu::AllowCompaction allowCompaction = optixu::AllowCompaction::No;
        optixu::AllowRandomInstanceAccess allowRandomInstanceAccess = optixu::AllowRandomInstanceAccess::No;
        
        ias.getConfiguration(&tradeoff, &allowUpdate, &allowCompaction, &allowRandomInstanceAccess);
        CHECK(tradeoff == optixu::ASTradeoff::PreferFastBuild);
        CHECK(allowUpdate == optixu::AllowUpdate::No);
        CHECK(allowCompaction == optixu::AllowCompaction::Yes);
        CHECK(allowRandomInstanceAccess == optixu::AllowRandomInstanceAccess::No);
        
        // Clean up
        ias.destroy();
    }
    
    SUBCASE("Add Instance to IAS")
    {
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        
        // Create a GAS to be instanced
        GASWithBuffers gasWithBuffers = createSimpleGAS(scene, optixContext, cudaContext, stream);
        
        // Create an instance
        optixu::Instance instance = scene.createInstance();
        CHECK(instance);
        
        // Set the GAS as child of the instance
        instance.setChild(gasWithBuffers.gas);
        
        // Configure IAS
        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);
        
        // Add instance to IAS
        ias.addChild(instance);
        
        // Verify child was added
        CHECK(ias.getChildCount() == 1);
        
        // Mark dirty for rebuild
        ias.markDirty();
        
        // Generate shader binding table layout for the scene
        size_t hitGroupSbtSize;
        scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
        
        // Prepare for build
        OptixAccelBufferSizes memReq;
        ias.prepareForBuild(&memReq);
        
        CHECK(memReq.outputSizeInBytes > 0);
        CHECK(memReq.tempSizeInBytes > 0);
        
        // Clean up
        instance.destroy();
        gasWithBuffers.gas.destroy();
        gasWithBuffers.geomInst.destroy();
        ias.destroy();
    }
    
    SUBCASE("Build and Rebuild IAS")
    {
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        GASWithBuffers gasWithBuffers = createSimpleGAS(scene, optixContext, cudaContext, stream);
        optixu::Instance instance = scene.createInstance();
        
        // Set up instance
        instance.setChild(gasWithBuffers.gas);
        
        // Configure and add to IAS
        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);
        ias.addChild(instance);
        ias.markDirty();
        
        // Generate shader binding table layout for the scene
        // This is required before building the IAS
        size_t hitGroupSbtSize;
        scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
        
        // Prepare for build
        OptixAccelBufferSizes memReq;
        ias.prepareForBuild(&memReq);
        
        // Allocate buffers
        cudau::Buffer instanceBuffer;
        cudau::Buffer accelBuffer;
        cudau::Buffer scratchBuffer;
        
        // Instance buffer needs to hold instance records
        instanceBuffer.initialize(cudaContext, cudau::BufferType::Device, ias.getChildCount() * sizeof(OptixInstance), 1);
        accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
        scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.tempSizeInBytes, 1);
        
        // Build the IAS
        OptixTraversableHandle traversable = ias.rebuild(stream, instanceBuffer, accelBuffer, scratchBuffer);
        CHECK(traversable != 0);
        
        // Wait for build to complete
        cuStreamSynchronize(stream);
        
        // Verify IAS is ready
        CHECK(ias.isReady());
        CHECK(ias.getHandle() == traversable);
        
        // Clean up
        instanceBuffer.finalize();
        accelBuffer.finalize();
        scratchBuffer.finalize();
        instance.destroy();
        gasWithBuffers.gas.destroy();
        gasWithBuffers.geomInst.destroy();
        ias.destroy();
    }
    
    SUBCASE("IAS with Multiple Instances")
    {
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        
        // Create multiple GAS objects with different offsets
        const int numInstances = 3;
        std::vector<GASWithBuffers> gasObjects;
        std::vector<optixu::Instance> instances;
        
        for (int i = 0; i < numInstances; ++i) {
            // Create GAS with offset
            gasObjects.push_back(createSimpleGAS(scene, optixContext, cudaContext, stream, i * 2.0f));
            
            // Create instance
            optixu::Instance instance = scene.createInstance();
            instance.setChild(gasObjects[i].gas);
            
            // Set instance ID
            instance.setID(i);
            
            // Add to IAS
            ias.addChild(instance);
            instances.push_back(instance);
        }
        
        // Configure IAS
        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);
        ias.markDirty();
        
        // Verify child count
        CHECK(ias.getChildCount() == numInstances);
        
        // Find child index
        uint32_t foundIndex = ias.findChildIndex(instances[1]);
        CHECK(foundIndex == 1);
        
        // Get child by index
        optixu::Instance retrievedInstance = ias.getChild(1);
        CHECK(retrievedInstance);
        
        // Remove one child
        ias.removeChildAt(1);
        CHECK(ias.getChildCount() == numInstances - 1);
        
        // Clear all children
        ias.clearChildren();
        CHECK(ias.getChildCount() == 0);
        
        // Clean up
        for (auto& inst : instances) {
            inst.destroy();
        }
        for (auto& gasWithBuffers : gasObjects) {
            gasWithBuffers.gas.destroy();
            gasWithBuffers.geomInst.destroy();
        }
        ias.destroy();
    }
    
    SUBCASE("IAS with Transform")
    {
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        GASWithBuffers gasWithBuffers = createSimpleGAS(scene, optixContext, cudaContext, stream);
        
        // Create transform
        optixu::Transform transform = scene.createTransform();
        
        // Configure transform as matrix motion with 1 key (static effect)
        size_t transformSize;
        transform.setConfiguration(optixu::TransformType::MatrixMotion, 1, &transformSize);
        transform.setMotionOptions(0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        
        // Set transform matrix (translate by 5 units in X)
        float matrix[12] = {
            1.0f, 0.0f, 0.0f, 5.0f,  // Row 0: X axis + translation
            0.0f, 1.0f, 0.0f, 0.0f,  // Row 1: Y axis
            0.0f, 0.0f, 1.0f, 0.0f   // Row 2: Z axis
        };
        transform.setMatrixMotionKey(0, matrix);
        transform.setChild(gasWithBuffers.gas);
        transform.markDirty();
        
        // Rebuild transform
        cudau::Buffer transformBuffer;
        transformBuffer.initialize(cudaContext, cudau::BufferType::Device, transformSize, 1);
        OptixTraversableHandle transformHandle = transform.rebuild(stream, transformBuffer);
        
        // Create instance with transform
        optixu::Instance instance = scene.createInstance();
        instance.setChild(transform);
        
        // Add to IAS
        ias.addChild(instance);
        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);
        
        // Generate shader binding table layout
        size_t sbtSize;
        scene.generateShaderBindingTableLayout(&sbtSize);
        
        // Build IAS
        OptixAccelBufferSizes memReq;
        ias.prepareForBuild(&memReq);
        
        cudau::Buffer instanceBuffer;
        cudau::Buffer accelBuffer;
        cudau::Buffer scratchBuffer;
        
        size_t instanceBufferSize = sizeof(OptixInstance) * ias.getChildCount();
        instanceBuffer.initialize(cudaContext, cudau::BufferType::Device, instanceBufferSize, 1);
        accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
        scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.tempSizeInBytes, 1);
        
        OptixTraversableHandle traversable = ias.rebuild(stream, instanceBuffer, accelBuffer, scratchBuffer);
        CHECK(traversable != 0);
        
        cuStreamSynchronize(stream);
        
        // Clean up
        instanceBuffer.finalize();
        accelBuffer.finalize();
        scratchBuffer.finalize();
        transformBuffer.finalize();
        instance.destroy();
        transform.destroy();
        gasWithBuffers.gas.destroy();
        gasWithBuffers.geomInst.destroy();
        ias.destroy();
    }
    
    SUBCASE("IAS Compaction")
    {
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        
        // Create multiple instances for more complex structure
        const int numInstances = 5;
        std::vector<GASWithBuffers> gasObjects;
        std::vector<optixu::Instance> instances;
        
        for (int i = 0; i < numInstances; ++i) {
            gasObjects.push_back(createSimpleGAS(scene, optixContext, cudaContext, stream, i * 3.0f));
            
            optixu::Instance instance = scene.createInstance();
            instance.setChild(gasObjects[i].gas);
            instance.setID(i);
            
            ias.addChild(instance);
            instances.push_back(instance);
        }
        
        // Configure for compaction
        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes,  // Enable compaction
            optixu::AllowRandomInstanceAccess::No);
        ias.markDirty();
        
        // Generate shader binding table layout
        size_t sbtSize;
        scene.generateShaderBindingTableLayout(&sbtSize);
        
        // Prepare for build
        OptixAccelBufferSizes memReq;
        ias.prepareForBuild(&memReq);
        
        // Allocate buffers
        cudau::Buffer instanceBuffer;
        cudau::Buffer accelBuffer;
        cudau::Buffer scratchBuffer;
        
        size_t instanceBufferSize = sizeof(OptixInstance) * ias.getChildCount();
        instanceBuffer.initialize(cudaContext, cudau::BufferType::Device, instanceBufferSize, 1);
        accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
        scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.tempSizeInBytes, 1);
        
        // Build the IAS
        OptixTraversableHandle traversable = ias.rebuild(stream, instanceBuffer, accelBuffer, scratchBuffer);
        CHECK(traversable != 0);
        
        // Wait for build to complete
        cuStreamSynchronize(stream);
        
        // Prepare for compaction
        size_t compactedSize;
        ias.prepareForCompact(&compactedSize);
        CHECK(compactedSize > 0);
        CHECK(compactedSize <= memReq.outputSizeInBytes);
        
        // Allocate compacted buffer
        cudau::Buffer compactedBuffer;
        compactedBuffer.initialize(cudaContext, cudau::BufferType::Device, compactedSize, 1);
        
        // Perform compaction
        OptixTraversableHandle compactedTraversable = ias.compact(stream, compactedBuffer);
        CHECK(compactedTraversable != 0);
        
        // Wait for compaction to complete
        cuStreamSynchronize(stream);
        
        // Remove uncompacted data
        ias.removeUncompacted();
        
        LOG(INFO) << "IAS compaction: " << memReq.outputSizeInBytes << " bytes -> " << compactedSize << " bytes";
        
        // Clean up
        instanceBuffer.finalize();
        accelBuffer.finalize();
        scratchBuffer.finalize();
        compactedBuffer.finalize();
        for (auto& inst : instances) {
            inst.destroy();
        }
        for (auto& gasWithBuffers : gasObjects) {
            gasWithBuffers.gas.destroy();
            gasWithBuffers.geomInst.destroy();
        }
        ias.destroy();
    }
    
    SUBCASE("Motion Blur Configuration")
    {
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        
        // Configure motion options
        uint32_t numKeys = 2;  // Two motion steps
        float timeBegin = 0.0f;
        float timeEnd = 1.0f;
        OptixMotionFlags motionFlags = OPTIX_MOTION_FLAG_NONE;
        
        ias.setMotionOptions(numKeys, timeBegin, timeEnd, motionFlags);
        
        // Verify motion options
        uint32_t retrievedNumKeys;
        float retrievedTimeBegin, retrievedTimeEnd;
        OptixMotionFlags retrievedFlags;
        
        ias.getMotionOptions(&retrievedNumKeys, &retrievedTimeBegin, &retrievedTimeEnd, &retrievedFlags);
        CHECK(retrievedNumKeys == numKeys);
        CHECK(retrievedTimeBegin == timeBegin);
        CHECK(retrievedTimeEnd == timeEnd);
        CHECK(retrievedFlags == motionFlags);
        
        // Clean up
        ias.destroy();
    }
    
    SUBCASE("Nested IAS (Multi-level Instancing)")
    {
        // Create bottom-level IAS
        optixu::InstanceAccelerationStructure bottomIAS = scene.createInstanceAccelerationStructure();
        
        // Add multiple GAS to bottom IAS
        std::vector<GASWithBuffers> gasObjects;
        std::vector<optixu::Instance> instances;
        for (int i = 0; i < 2; ++i) {
            gasObjects.push_back(createSimpleGAS(scene, optixContext, cudaContext, stream, i * 1.5f));
            optixu::Instance instance = scene.createInstance();
            instance.setChild(gasObjects[i].gas);
            bottomIAS.addChild(instance);
            instances.push_back(instance);
        }
        
        bottomIAS.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);
        bottomIAS.markDirty();
        
        // Generate shader binding table layout
        size_t sbtSize;
        scene.generateShaderBindingTableLayout(&sbtSize);
        
        // Build bottom IAS
        OptixAccelBufferSizes bottomMemReq;
        bottomIAS.prepareForBuild(&bottomMemReq);
        
        cudau::Buffer bottomInstanceBuffer;
        cudau::Buffer bottomAccelBuffer;
        cudau::Buffer bottomScratchBuffer;
        
        size_t bottomInstanceBufferSize = sizeof(OptixInstance) * bottomIAS.getChildCount();
        bottomInstanceBuffer.initialize(cudaContext, cudau::BufferType::Device, bottomInstanceBufferSize, 1);
        bottomAccelBuffer.initialize(cudaContext, cudau::BufferType::Device, bottomMemReq.outputSizeInBytes, 1);
        bottomScratchBuffer.initialize(cudaContext, cudau::BufferType::Device, bottomMemReq.tempSizeInBytes, 1);
        
        bottomIAS.rebuild(stream, bottomInstanceBuffer, bottomAccelBuffer, bottomScratchBuffer);
        cuStreamSynchronize(stream);
        
        // Create top-level IAS
        optixu::InstanceAccelerationStructure topIAS = scene.createInstanceAccelerationStructure();
        
        // Create instance of bottom IAS
        optixu::Instance topInstance = scene.createInstance();
        topInstance.setChild(bottomIAS);
        topIAS.addChild(topInstance);
        
        topIAS.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);
        topIAS.markDirty();
        
        // Build top IAS
        OptixAccelBufferSizes topMemReq;
        topIAS.prepareForBuild(&topMemReq);
        
        cudau::Buffer topInstanceBuffer;
        cudau::Buffer topAccelBuffer;
        cudau::Buffer topScratchBuffer;
        
        size_t topInstanceBufferSize = sizeof(OptixInstance) * topIAS.getChildCount();
        topInstanceBuffer.initialize(cudaContext, cudau::BufferType::Device, topInstanceBufferSize, 1);
        topAccelBuffer.initialize(cudaContext, cudau::BufferType::Device, topMemReq.outputSizeInBytes, 1);
        topScratchBuffer.initialize(cudaContext, cudau::BufferType::Device, topMemReq.tempSizeInBytes, 1);
        
        OptixTraversableHandle topTraversable = topIAS.rebuild(stream, topInstanceBuffer, topAccelBuffer, topScratchBuffer);
        CHECK(topTraversable != 0);
        
        cuStreamSynchronize(stream);
        
        // Verify both IAS are ready
        CHECK(bottomIAS.isReady());
        CHECK(topIAS.isReady());
        
        // Clean up (note: order matters when destroying nested structures)
        topInstanceBuffer.finalize();
        topAccelBuffer.finalize();
        topScratchBuffer.finalize();
        bottomInstanceBuffer.finalize();
        bottomAccelBuffer.finalize();
        bottomScratchBuffer.finalize();
        topInstance.destroy();
        topIAS.destroy();
        for (auto& inst : instances) {
            inst.destroy();
        }
        for (auto& gasWithBuffers : gasObjects) {
            gasWithBuffers.gas.destroy();
            gasWithBuffers.geomInst.destroy();
        }
        bottomIAS.destroy();
    }
    
    // Clean up CUDA resources
    cuStreamDestroy(stream);
    scene.destroy();
    gpuContext.cleanup();
}

TEST_CASE("InstanceAccelerationStructure Update Operations")
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
    
    SUBCASE("IAS Update After Instance Modification")
    {
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        GASWithBuffers gasWithBuffers = createSimpleGAS(scene, optixContext, cudaContext, stream);
        optixu::Instance instance = scene.createInstance();
        
        // Set up instance
        instance.setChild(gasWithBuffers.gas);
        
        // Configure for update
        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::Yes,  // Enable updates
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);
        ias.addChild(instance);
        ias.markDirty();
        
        // Generate shader binding table layout for the scene
        size_t hitGroupSbtSize;
        scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
        
        // Initial build
        OptixAccelBufferSizes memReq;
        ias.prepareForBuild(&memReq);
        
        cudau::Buffer instanceBuffer;
        cudau::Buffer accelBuffer;
        cudau::Buffer scratchBuffer;
        
        instanceBuffer.initialize(cudaContext, cudau::BufferType::Device, ias.getChildCount() * sizeof(OptixInstance), 1);
        accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
        // For updates, allocate the larger of temp or tempUpdate size
        size_t scratchSize = std::max(memReq.tempSizeInBytes, memReq.tempUpdateSizeInBytes);
        scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, scratchSize, 1);
        
        OptixTraversableHandle traversable = ias.rebuild(stream, instanceBuffer, accelBuffer, scratchBuffer);
        CHECK(traversable != 0);
        cuStreamSynchronize(stream);
        
        // Modify instance properties (e.g., change instance ID)
        instance.setID(42);
        
        // Update IAS (not rebuild, just update)
        ias.update(stream, scratchBuffer);
        cuStreamSynchronize(stream);
        
        // Verify IAS is still ready
        CHECK(ias.isReady());
        
        // Clean up
        instanceBuffer.finalize();
        accelBuffer.finalize();
        scratchBuffer.finalize();
        instance.destroy();
        gasWithBuffers.gas.destroy();
        gasWithBuffers.geomInst.destroy();
        ias.destroy();
    }
    
    SUBCASE("IAS with Instance Flags")
    {
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        GASWithBuffers gasWithBuffers = createSimpleGAS(scene, optixContext, cudaContext, stream);
        optixu::Instance instance = scene.createInstance();
        
        // Set up instance with flags
        instance.setChild(gasWithBuffers.gas);
        
        // Set instance flags
        instance.setFlags(OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT);
        
        // Set visibility mask
        instance.setVisibilityMask(0xFF);  // Visible to all ray types
        
        // Add to IAS
        ias.addChild(instance);
        ias.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::No);
        ias.markDirty();
        
        // Generate shader binding table layout
        size_t hitGroupSbtSize;
        scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
        
        // Build IAS
        OptixAccelBufferSizes memReq;
        ias.prepareForBuild(&memReq);
        
        cudau::Buffer instanceBuffer;
        cudau::Buffer accelBuffer;
        cudau::Buffer scratchBuffer;
        
        size_t instanceBufferSize = sizeof(OptixInstance) * ias.getChildCount();
        instanceBuffer.initialize(cudaContext, cudau::BufferType::Device, instanceBufferSize, 1);
        accelBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.outputSizeInBytes, 1);
        scratchBuffer.initialize(cudaContext, cudau::BufferType::Device, memReq.tempSizeInBytes, 1);
        
        OptixTraversableHandle traversable = ias.rebuild(stream, instanceBuffer, accelBuffer, scratchBuffer);
        CHECK(traversable != 0);
        
        cuStreamSynchronize(stream);
        
        // Clean up
        instanceBuffer.finalize();
        accelBuffer.finalize();
        scratchBuffer.finalize();
        instance.destroy();
        gasWithBuffers.gas.destroy();
        gasWithBuffers.geomInst.destroy();
        ias.destroy();
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