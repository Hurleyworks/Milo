#include "TriangleMeshHandler.h"
#include "DisneyMaterialHandler.h"
#include "Handlers.h"

TriangleMeshHandler::TriangleMeshHandler (RenderContextPtr ctx) :
    renderContext_ (ctx)
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
    geomInstDataBuffer_.initialize (
        cudaContext_,
        cudau::BufferType::ZeroCopy, // ZeroCopy allows CPU/GPU access
        maxNumGeometryInstances);

    geomInstSlotFinder_.initialize (maxNumGeometryInstances);

    return true;
}

OptiXTriMeshSurface* TriangleMeshHandler::createTriMeshSurface (
    const sabi::CgModel& model,
    uint32_t surfaceIndex,
    DisneyMaterial* disneyMaterial)
{
    const auto& surface = model.S[surfaceIndex];
    const auto& V = model.V;
    const auto& N = model.N;
    const auto& UV0 = model.UV0;

    // Convert vertices
    std::vector<shared::Vertex> vertices;
    vertices.reserve (model.vertexCount());

    for (size_t i = 0; i < model.vertexCount(); ++i)
    {
        shared::Vertex vertex;
        vertex.position = Point3D (V (0, i), V (1, i), V (2, i));
        vertex.normal = N.cols() > i ? Normal3D (N (0, i), N (1, i), N (2, i)) : Normal3D (0, 1, 0);
        vertex.texCoord = UV0.cols() > i ? Point2D (UV0 (0, i), UV0 (1, i)) : Point2D (0, 0);
        vertex.texCoord0Dir = Vector3D (1, 0, 0); // Simple tangent
        vertices.push_back (vertex);
    }

    // Convert triangles
    std::vector<shared::Triangle> triangles;
    const auto& F = surface.indices();
    triangles.reserve (F.cols());

    for (int i = 0; i < F.cols(); ++i)
    {
        shared::Triangle triangle;
        triangle.index0 = F (0, i);
        triangle.index1 = F (1, i);
        triangle.index2 = F (2, i);
        triangles.push_back (triangle);
    }

    // Create OptiXTriMeshSurface
    OptiXTriMeshSurface* triMeshSurface = new OptiXTriMeshSurface();
    auto& geom = triMeshSurface->triGeometry;

    // Calculate AABB
    triMeshSurface->aabb = AABB();
    for (const auto& vertex : vertices)
    {
        triMeshSurface->aabb.unify (vertex.position);
    }

    // Initialize GPU buffers
    geom.vertexBuffer.initialize (cudaContext_, cudau::BufferType::Device, vertices);
    geom.triangleBuffer.initialize (cudaContext_, cudau::BufferType::Device, triangles);

    // Handle emissive geometry
    const auto& material = surface.cgMaterial;
    if (material.emission.luminous > 0.0f)
    {
        // Initialize emitter distribution for light sampling
        geom.emitterPrimDist.initialize (
            cudaContext_,
            cudau::BufferType::Device,
            nullptr,
            static_cast<uint32_t> (triangles.size()));
    }

    // Get slot and create OptiX geometry instance
    triMeshSurface->geomInstSlot = geomInstSlotFinder_.getFirstAvailableSlot();
    geomInstSlotFinder_.setInUse (triMeshSurface->geomInstSlot);

    // Create OptiX geometry instance through RenderContext's scene
    optixu::Scene scene = renderContext_->getScene();
    triMeshSurface->optixGeomInst = scene.createGeometryInstance();

    // Setup geometry data in mapped buffer
    shared::GeometryInstanceData* geomInstDataOnHost =
        geomInstDataBuffer_.getMappedPointer();

    shared::GeometryInstanceData& geomInstData =
        geomInstDataOnHost[triMeshSurface->geomInstSlot];

    geomInstData.vertexBuffer =
        geom.vertexBuffer.getROBuffer<shared::enableBufferOobCheck>();
    geomInstData.triangleBuffer =
        geom.triangleBuffer.getROBuffer<shared::enableBufferOobCheck>();
    // Note: We still need to set materialSlot for the shader binding table
    // but we also store the DisneyMaterial pointer directly
    geomInstData.materialSlot = 0;  // Could be used for SBT indexing if needed
    geomInstData.geomInstSlot = triMeshSurface->geomInstSlot;

    if (geom.emitterPrimDist.isInitialized())
    {
        geom.emitterPrimDist.getDeviceType (&geomInstData.emitterPrimDist);
    }

    // Set up OptiX geometry instance
    triMeshSurface->optixGeomInst.setVertexBuffer (geom.vertexBuffer);
    triMeshSurface->optixGeomInst.setTriangleBuffer (geom.triangleBuffer);
    // Note: setNumMaterials and setMaterial should be called when we have an actual OptiX material
    // For now, we'll skip material setup - it should be done by the caller who has the material
    triMeshSurface->optixGeomInst.setUserData (triMeshSurface->geomInstSlot);
    triMeshSurface->disneyMaterial = disneyMaterial;

    triMeshSurfaces_.push_back (triMeshSurface);
    return triMeshSurface;
}

OptiXTriMesh* TriangleMeshHandler::createTriMeshFromModel (
    sabi::CgModelPtr model,
    const std::vector<DisneyMaterial*>& disneyMaterials)
{
    if (!model)
    {
        LOG (WARNING) << "TriangleMeshHandler::createGeometryGroupFromModel: null model";
        return nullptr;
    }

    // Collect all geometry surfaces from the model
    std::set<OptiXTriMeshSurface*> surfaces;

    // Iterate through all surfaces in the model
    for (uint32_t surfaceIndex = 0; surfaceIndex < model->S.size(); ++surfaceIndex)
    {
        const auto& surface = model->S[surfaceIndex];

        // Skip empty surfaces
        if (surface.indices().cols() == 0)
            continue;

        // Determine Disney material for this surface
        DisneyMaterial* disneyMaterial = nullptr;
        if (!disneyMaterials.empty())
        {
            if (surfaceIndex < disneyMaterials.size())
                disneyMaterial = disneyMaterials[surfaceIndex];
            else
                disneyMaterial = disneyMaterials.back(); // Use last material for remaining surfaces
        }

        // Create geometry surface for this surface
        OptiXTriMeshSurface* triMeshSurface = createTriMeshSurface (*model, surfaceIndex, disneyMaterial);
        if (triMeshSurface)
        {
            surfaces.insert (triMeshSurface);
        }
    }

    // Check if we have any valid surfaces
    if (surfaces.empty())
    {
        LOG (WARNING) << "TriangleMeshHandler::createGeometryGroupFromModel: no valid surfaces found";
        return nullptr;
    }

    // Create the OptiX trimesh with all surfaces
    OptiXTriMesh* mesh = createTriMesh (surfaces);

    return mesh;
}

OptiXTriMesh* TriangleMeshHandler::createTriMesh (
    const std::set<OptiXTriMeshSurface*>& surfaces)
{
    OptiXTriMesh* mesh = new OptiXTriMesh();

    // Copy surface pointers
    for (auto* surface : surfaces)
    {
        mesh->surfaces.insert (surface);
        mesh->aabb.unify (surface->aabb);

        // Count emitter primitives
        const auto& geom = surface->triGeometry;
        if (geom.emitterPrimDist.isInitialized())
        {
            mesh->numEmitterPrimitives +=
                static_cast<uint32_t> (geom.triangleBuffer.numElements());
        }
    }

    // Create OptiX GAS through RenderContext's scene
    optixu::Scene scene = renderContext_->getScene();
    mesh->optixGas = scene.createGeometryAccelerationStructure (optixu::GeometryType::Triangles);

    // Add children
    for (auto* surface : surfaces)
    {
        mesh->optixGas.addChild (surface->optixGeomInst);
    }

    // Configure for build
    mesh->optixGas.setConfiguration (
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::No,
        optixu::AllowRandomVertexAccess::No);

    // Note: setNumMaterialSets and setNumRayTypes should be called by the engine
    // that knows how many material sets and ray types it uses

    mesh->needsReallocation = true;
    mesh->needsRebuild = true;
    mesh->refittable = false;

    triMeshes_.push_back (mesh);
    return mesh;
}

void TriangleMeshHandler::buildGAS (OptiXTriMesh* mesh, CUstream stream)
{
    OptixAccelBufferSizes sizes; // Correct type

    // Prepare for build
    if (mesh->needsReallocation)
    {
        mesh->optixGas.prepareForBuild (&sizes);

        // Allocate or resize GAS memory
        if (mesh->optixGasMem.isInitialized())
            mesh->optixGasMem.resize (sizes.outputSizeInBytes, 1);
        else
            mesh->optixGasMem.initialize (
                cudaContext_,
                cudau::BufferType::Device,
                sizes.outputSizeInBytes, 1);

        // Get shared scratch buffer from RenderContext and resize if needed
        cudau::Buffer& asScratchMem = renderContext_->getASScratchBuffer();
        if (!asScratchMem.isInitialized())
            asScratchMem.initialize (
                cudaContext_,
                cudau::BufferType::Device,
                sizes.tempSizeInBytes, 1);
        else if (sizes.tempSizeInBytes > asScratchMem.sizeInBytes())
            asScratchMem.resize (sizes.tempSizeInBytes, 1);

        mesh->needsReallocation = false;
    }

    // Build or rebuild GAS
    if (mesh->needsRebuild)
    {
        cudau::Buffer& asScratchMem = renderContext_->getASScratchBuffer();
        mesh->optixGas.rebuild (
            stream,
            mesh->optixGasMem,
            asScratchMem);

        mesh->needsRebuild = false;
    }
}

OptixTraversableHandle TriangleMeshHandler::getTraversableHandle (
    OptiXTriMesh* mesh) const
{
    return mesh->optixGas.getHandle();
}

shared::GeometryInstanceData* TriangleMeshHandler::getGeometryInstanceData (uint32_t slot)
{
    if (slot >= maxNumGeometryInstances)
        return nullptr;

    shared::GeometryInstanceData* geomInstDataOnHost =
        geomInstDataBuffer_.getMappedPointer();

    return &geomInstDataOnHost[slot];
}

void TriangleMeshHandler::destroyTriMeshSurface (OptiXTriMeshSurface* surface)
{
    if (!surface)
        return;

    // Free the slot
    geomInstSlotFinder_.setNotInUse (surface->geomInstSlot);

    // Clean up the surface
    surface->finalize();

    // Remove from tracking
    auto it = std::find (triMeshSurfaces_.begin(), triMeshSurfaces_.end(), surface);
    if (it != triMeshSurfaces_.end())
        triMeshSurfaces_.erase (it);

    delete surface;
}

void TriangleMeshHandler::destroyTriMesh (OptiXTriMesh* mesh)
{
    if (!mesh)
        return;

    // Clean up OptiX resources
    mesh->optixGas.destroy();
    mesh->optixGasMem.finalize();

    // Remove from tracking
    auto it = std::find (triMeshes_.begin(), triMeshes_.end(), mesh);
    if (it != triMeshes_.end())
        triMeshes_.erase (it);

    delete mesh;
}

OptiXTriMesh* TriangleMeshHandler::createTriMeshFromModelWithMaterials (
    sabi::CgModelPtr model,
    const std::filesystem::path& materialFolder)
{
    if (!model)
    {
        LOG (WARNING) << "TriangleMeshHandler::createTriMeshFromModelWithMaterials: null model";
        return nullptr;
    }
    
    // Access the DisneyMaterialHandler through RenderContext's handlers
    auto disneyMaterialHandler = renderContext_->getHandlers().disneyMaterialHandler;
    if (!disneyMaterialHandler)
    {
        LOG (WARNING) << "TriangleMeshHandler::createTriMeshFromModelWithMaterials: DisneyMaterialHandler not available";
        return nullptr;
    }
    
    // Create Disney materials for each surface
    std::vector<DisneyMaterial*> disneyMaterials;
    std::vector<optixu::Material> optixMaterials;
    
    for (uint32_t surfaceIndex = 0; surfaceIndex < model->S.size(); ++surfaceIndex)
    {
        const auto& surface = model->S[surfaceIndex];
        
        // Create OptiX material using DisneyMaterialHandler
        auto [optixMat, materialSlot] = disneyMaterialHandler->createDisneyMaterial(
            surface.cgMaterial, 
            materialFolder, 
            model);
        
        optixMaterials.push_back(optixMat);
        
        // Get the DisneyMaterial pointer from the handler
        // Note: We need to access the internal DisneyMaterial from the handler
        // For now, we'll pass nullptr and rely on material slots in the shader
        disneyMaterials.push_back(nullptr);
    }
    
    // Create the mesh with the materials
    OptiXTriMesh* mesh = createTriMeshFromModel(model, disneyMaterials);
    
    // Store the OptiX materials with the surfaces for later use
    if (mesh)
    {
        uint32_t matIdx = 0;
        for (auto* surface : mesh->surfaces)
        {
            if (matIdx < optixMaterials.size())
            {
                // Set the OptiX material on the geometry instance
                surface->optixGeomInst.setMaterialCount(1, optixu::BufferView());
                surface->optixGeomInst.setMaterial(0, 0, optixMaterials[matIdx]);
                matIdx++;
            }
        }
    }
    
    return mesh;
}

void TriangleMeshHandler::finalize()
{
    // Clean up geometry surfaces
    for (auto* surface : triMeshSurfaces_)
    {
        surface->finalize();
        delete surface;
    }
    triMeshSurfaces_.clear();

    // Clean up OptiX trimeshes
    for (auto* mesh : triMeshes_)
    {
        mesh->optixGas.destroy();
        mesh->optixGasMem.finalize();
        delete mesh;
    }
    triMeshes_.clear();

    // Clean up buffers
    geomInstDataBuffer_.finalize();
}