#include "TriangleMeshHandler.h"

TriangleMeshHandler::TriangleMeshHandler(RenderContextPtr ctx)
    : renderContext_(ctx)
{
    cudaContext_ = ctx->getCudaContext();
}

TriangleMeshHandler::~TriangleMeshHandler()
{
    finalize();
}

bool TriangleMeshHandler::initialize()
{
    // Initialize geometry instance data buffer (mapped for CPU/GPU access)
    geomInstDataBuffer_.initialize(
        cudaContext_, 
        cudau::BufferType::ZeroCopy,  // ZeroCopy allows CPU/GPU access
        maxNumGeometryInstances);
    
    geomInstSlotFinder_.initialize(maxNumGeometryInstances);
    
    return true;
}

GeometryInstance* TriangleMeshHandler::createGeometryInstance(
    const sabi::CgModel& model,
    uint32_t surfaceIndex,
    uint32_t materialSlot)
{
    const auto& surface = model.S[surfaceIndex];
    const auto& V = model.V;
    const auto& N = model.N;
    const auto& UV0 = model.UV0;
    
    // Convert vertices
    std::vector<shared::Vertex> vertices;
    vertices.reserve(model.vertexCount());
    
    for (size_t i = 0; i < model.vertexCount(); ++i)
    {
        shared::Vertex vertex;
        vertex.position = Point3D(V(0, i), V(1, i), V(2, i));
        vertex.normal = N.cols() > i ? 
            Normal3D(N(0, i), N(1, i), N(2, i)) : 
            Normal3D(0, 1, 0);
        vertex.texCoord = UV0.cols() > i ? 
            Point2D(UV0(0, i), UV0(1, i)) : 
            Point2D(0, 0);
        vertex.texCoord0Dir = Vector3D(1, 0, 0); // Simple tangent
        vertices.push_back(vertex);
    }
    
    // Convert triangles
    std::vector<shared::Triangle> triangles;
    const auto& F = surface.indices();
    triangles.reserve(F.cols());
    
    for (int i = 0; i < F.cols(); ++i)
    {
        shared::Triangle triangle;
        triangle.index0 = F(0, i);
        triangle.index1 = F(1, i);
        triangle.index2 = F(2, i);
        triangles.push_back(triangle);
    }
    
    // Create GeometryInstance
    GeometryInstance* geomInst = new GeometryInstance();
    geomInst->geometry = TriangleGeometry();
    auto& geom = std::get<TriangleGeometry>(geomInst->geometry);
    
    // Calculate AABB
    geomInst->aabb = AABB();
    for (const auto& vertex : vertices)
    {
        geomInst->aabb.unify(vertex.position);
    }
    
    // Initialize GPU buffers
    geom.vertexBuffer.initialize(cudaContext_, cudau::BufferType::Device, vertices);
    geom.triangleBuffer.initialize(cudaContext_, cudau::BufferType::Device, triangles);
    
    // Handle emissive geometry
    const auto& material = surface.cgMaterial;
    if (material.emission.luminous > 0.0f)
    {
        // Initialize emitter distribution for light sampling
        geom.emitterPrimDist.initialize(
            cudaContext_, 
            cudau::BufferType::Device,
            nullptr,
            static_cast<uint32_t>(triangles.size()));
    }
    
    // Get slot and create OptiX geometry instance
    geomInst->geomInstSlot = geomInstSlotFinder_.getFirstAvailableSlot();
    geomInstSlotFinder_.setInUse(geomInst->geomInstSlot);
    
    // Create OptiX geometry instance through RenderContext's scene
    optixu::Scene scene = renderContext_->getScene();
    geomInst->optixGeomInst = scene.createGeometryInstance();
    
    // Setup geometry data in mapped buffer
    shared::GeometryInstanceData* geomInstDataOnHost = 
        geomInstDataBuffer_.getMappedPointer();
    
    shared::GeometryInstanceData& geomInstData = 
        geomInstDataOnHost[geomInst->geomInstSlot];
    
    geomInstData.vertexBuffer = 
        geom.vertexBuffer.getROBuffer<shared::enableBufferOobCheck>();
    geomInstData.triangleBuffer = 
        geom.triangleBuffer.getROBuffer<shared::enableBufferOobCheck>();
    geomInstData.materialSlot = materialSlot;
    geomInstData.geomInstSlot = geomInst->geomInstSlot;
    
    if (geom.emitterPrimDist.isInitialized())
    {
        geom.emitterPrimDist.getDeviceType(&geomInstData.emitterPrimDist);
    }
    
    // Set up OptiX geometry instance
    geomInst->optixGeomInst.setVertexBuffer(geom.vertexBuffer);
    geomInst->optixGeomInst.setTriangleBuffer(geom.triangleBuffer);
    // Note: setNumMaterials and setMaterial should be called when we have an actual OptiX material
    // For now, we'll skip material setup - it should be done by the caller who has the material
    geomInst->optixGeomInst.setUserData(geomInst->geomInstSlot);
    
    geometryInstances_.push_back(geomInst);
    return geomInst;
}

GeometryGroup* TriangleMeshHandler::createGeometryGroup(
    const std::set<GeometryInstance*>& instances)
{
    GeometryGroup* group = new GeometryGroup();
    
    // Copy instance pointers
    for (auto* inst : instances)
    {
        group->geomInsts.insert(inst);
        group->aabb.unify(inst->aabb);
        
        // Count emitter primitives
        if (std::holds_alternative<TriangleGeometry>(inst->geometry))
        {
            const auto& geom = std::get<TriangleGeometry>(inst->geometry);
            if (geom.emitterPrimDist.isInitialized())
            {
                group->numEmitterPrimitives += 
                    static_cast<uint32_t>(geom.triangleBuffer.numElements());
            }
        }
    }
    
    // Create OptiX GAS through RenderContext's scene
    optixu::Scene scene = renderContext_->getScene();
    group->optixGas = scene.createGeometryAccelerationStructure(optixu::GeometryType::Triangles);
    
    // Add children
    for (auto* inst : instances)
    {
        group->optixGas.addChild(inst->optixGeomInst);
    }
    
    // Configure for build
    group->optixGas.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::No,
        optixu::AllowRandomVertexAccess::No);
    
    // Note: setNumMaterialSets and setNumRayTypes should be called by the engine
    // that knows how many material sets and ray types it uses
    
    group->needsReallocation = true;
    group->needsRebuild = true;
    group->refittable = false;
    
    geometryGroups_.push_back(group);
    return group;
}

void TriangleMeshHandler::buildGAS(GeometryGroup* group, CUstream stream)
{
    OptixAccelBufferSizes sizes;  // Correct type
    
    // Prepare for build
    if (group->needsReallocation)
    {
        group->optixGas.prepareForBuild(&sizes);
        
        // Allocate or resize GAS memory
        if (group->optixGasMem.isInitialized())
            group->optixGasMem.resize(sizes.outputSizeInBytes, 1);
        else
            group->optixGasMem.initialize(
                cudaContext_, 
                cudau::BufferType::Device,
                sizes.outputSizeInBytes, 1);
        
        // Get shared scratch buffer from RenderContext and resize if needed
        cudau::Buffer& asScratchMem = renderContext_->getASScratchBuffer();
        if (!asScratchMem.isInitialized())
            asScratchMem.initialize(
                cudaContext_,
                cudau::BufferType::Device,
                sizes.tempSizeInBytes, 1);
        else if (sizes.tempSizeInBytes > asScratchMem.sizeInBytes())
            asScratchMem.resize(sizes.tempSizeInBytes, 1);
        
        group->needsReallocation = false;
    }
    
    // Build or rebuild GAS
    if (group->needsRebuild)
    {
        cudau::Buffer& asScratchMem = renderContext_->getASScratchBuffer();
        group->optixGas.rebuild(
            stream,
            group->optixGasMem,
            asScratchMem);
        
        group->needsRebuild = false;
    }
}

OptixTraversableHandle TriangleMeshHandler::getTraversableHandle(
    GeometryGroup* group) const
{
    return group->optixGas.getHandle();
}

shared::GeometryInstanceData* TriangleMeshHandler::getGeometryInstanceData(uint32_t slot)
{
    if (slot >= maxNumGeometryInstances)
        return nullptr;
    
    shared::GeometryInstanceData* geomInstDataOnHost = 
        geomInstDataBuffer_.getMappedPointer();
    
    return &geomInstDataOnHost[slot];
}

void TriangleMeshHandler::destroyGeometryInstance(GeometryInstance* instance)
{
    if (!instance)
        return;
    
    // Free the slot
    geomInstSlotFinder_.setNotInUse(instance->geomInstSlot);
    
    // Clean up the instance
    instance->finalize();
    
    // Remove from tracking
    auto it = std::find(geometryInstances_.begin(), geometryInstances_.end(), instance);
    if (it != geometryInstances_.end())
        geometryInstances_.erase(it);
    
    delete instance;
}

void TriangleMeshHandler::destroyGeometryGroup(GeometryGroup* group)
{
    if (!group)
        return;
    
    // Clean up OptiX resources
    group->optixGas.destroy();
    group->optixGasMem.finalize();
    
    // Remove from tracking
    auto it = std::find(geometryGroups_.begin(), geometryGroups_.end(), group);
    if (it != geometryGroups_.end())
        geometryGroups_.erase(it);
    
    delete group;
}

void TriangleMeshHandler::finalize()
{
    // Clean up geometry instances
    for (auto* inst : geometryInstances_)
    {
        inst->finalize();
        delete inst;
    }
    geometryInstances_.clear();
    
    // Clean up geometry groups
    for (auto* group : geometryGroups_)
    {
        group->optixGas.destroy();
        group->optixGasMem.finalize();
        delete group;
    }
    geometryGroups_.clear();
    
    // Clean up buffers
    geomInstDataBuffer_.finalize();
}