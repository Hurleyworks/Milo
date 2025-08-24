// OptiXModel2 - Implementation using TriangleMesh and TriangleMeshSurface structures
#include "OptiXModel2.h"
#include "ModelUtilities.h"

using Eigen::Vector2f;
using sabi::CgModel;
using sabi::CgModelList;
using sabi::CgModelPtr;
using sabi::CgModelSurface;

// Helper to find texture folder (reused from original)
static fs::path findTextureFolder(const CgModelPtr& model)
{
    if (!model || model->images.empty())
        return fs::path();
    
    fs::path imagesFolder = model->contentDirectory / "images";
    if (!fs::exists(imagesFolder))
    {
        LOG(WARNING) << "Images folder not found in " << model->contentDirectory;
        return fs::path();
    }
    
    const std::string& imageUri = model->images[0].uri;
    if (imageUri.empty())
    {
        LOG(WARNING) << "First image has empty URI";
        return fs::path();
    }
    
    fs::path searchTarget = fs::path(imageUri).filename();
    LOG(DBUG) << "Searching for texture: " << searchTarget;
    
    try
    {
        for (const auto& entry : fs::recursive_directory_iterator(imagesFolder))
        {
            if (entry.is_regular_file() && entry.path().filename() == searchTarget)
            {
                LOG(DBUG) << "Found texture at: " << entry.path();
                return entry.path().parent_path();
            }
        }
    }
    catch (const fs::filesystem_error& e)
    {
        LOG(WARNING) << "Error searching images folder: " << e.what();
        return fs::path();
    }
    
    LOG(WARNING) << "Could not find texture " << searchTarget << " in " << imagesFolder;
    return fs::path();
}

void OptiXTriangleModel2::createGeometryAndMaterials(RenderContextPtr ctx, RenderableNode& node)
{
    CgModelPtr model = node->getModel();
    if (!model)
        throw std::runtime_error("Node has no cgModel: " + node->getName());
    
    // Get model path and space-time info
    fs::path modelPath = node->description().modelPath;
    SpaceTime& st = node->getSpaceTime();
    
    // Initialize the TriangleMesh structure
    triangleMesh.ref_count = 1;  // Start with ref count of 1
    triangleMesh.is_emissive = false;
    
    // Convert vertices to shared::Vertex format (shared by all surfaces)
    std::vector<shared::Vertex> vertices = populateVertices(model);
    if (!vertices.size())
        throw std::runtime_error("Populate vertices failed: " + modelPath.string());
    
    // Update AABB from vertices
    for (const auto& v : vertices)
    {
        triangleMesh.aabb.unify(v.position);
    }
    
    // Initialize shared vertex buffer
    triangleMesh.vertex_buffer.initialize(ctx->cuCtx, cudau::BufferType::Device, vertices);
    
    // Enable deformation if displacement vectors are present
    if (model->VD.cols() > 0)
    {
        LOG(DBUG) << "Enabling deformation for " << node->getName()
                  << " with " << model->VD.cols() << " displacement vectors";
        enableDeformation(ctx);
        
        if (originalVertexBuffer.numElements() != model->VD.cols())
        {
            LOG(WARNING) << "Vertex count mismatch - vertices: "
                        << originalVertexBuffer.numElements()
                        << " displacements: " << model->VD.cols();
        }
    }
    
    // Get content folder for material creation
    fs::path contentFolder = ctx->properties.renderProps->getVal<std::string>(RenderKey::ContentFolder);
    
    // Process each surface as a TriangleMeshSurface
    uint32_t materialCount = model->S.size();
    std::vector<uint8_t> allMaterialIDs;  // For multi-material support
    
    for (size_t surfIdx = 0; surfIdx < model->S.size(); ++surfIdx)
    {
        const auto& surface = model->S[surfIdx];
        
        // Check for emissive material
        if (surface.cgMaterial.emission.luminous > 0.0f)
        {
            triangleMesh.is_emissive = true;
            LOG(DBUG) << "Surface " << surfIdx << " is emissive";
        }
        
        // Get triangle indices for this surface
        const MatrixXu& F = surface.indices();
        
        // Convert to shared::Triangle format
        std::vector<shared::Triangle> triangles;
        triangles.reserve(F.cols());
        
        for (int i = 0; i < F.cols(); ++i)
        {
            Vector3u tri = F.col(i);
            triangles.emplace_back(shared::Triangle(tri.x(), tri.y(), tri.z()));
            
            // Track material IDs if multi-material
            if (materialCount > 1)
            {
                allMaterialIDs.push_back(surfIdx);
            }
        }
        
        if (triangles.empty())
        {
            LOG(DBUG) << "Skipping surface " << surfIdx << " - no triangles";
            continue;
        }
        
        // Create TriangleMeshSurface
        TriangleMeshSurface meshSurface;
        
        // Initialize triangle buffer for this surface
        meshSurface.triangle_buffer.initialize(ctx->cuCtx, cudau::BufferType::Device, triangles);
        
        // Allocate material slot from handler if available
        if (ctx->getHandlers().triangleMeshHandler)
        {
            meshSurface.material_slot = ctx->getHandlers().triangleMeshHandler->allocateMaterialSlot();
            if (meshSurface.material_slot == UINT32_MAX)
            {
                LOG(WARNING) << "Failed to allocate material slot for surface " << surfIdx;
                meshSurface.material_slot = 0;  // Fallback
            }
        }
        else
        {
            meshSurface.material_slot = surfIdx;  // Simple indexing if no handler
        }
        
        // Create OptiX geometry instance for this surface
        meshSurface.optix_geom_inst = ctx->scene.createGeometryInstance();
        meshSurface.optix_geom_inst.setVertexBuffer(triangleMesh.vertex_buffer);
        meshSurface.optix_geom_inst.setTriangleBuffer(meshSurface.triangle_buffer);
        
        // Create and set material
        optixu::Material mat = ctx->handlers->mat->createDisneyMaterial(
            surface.cgMaterial, contentFolder, model);
        
        meshSurface.optix_geom_inst.setNumMaterials(1, optixu::BufferView());
        meshSurface.optix_geom_inst.setMaterial(0, 0, mat);
        meshSurface.optix_geom_inst.setUserData(surfIdx);
        
        // Update surface AABB (optional, for culling)
        for (const auto& tri : triangles)
        {
            meshSurface.aabb.unify(vertices[tri.index0].position);
            meshSurface.aabb.unify(vertices[tri.index1].position);
            meshSurface.aabb.unify(vertices[tri.index2].position);
        }
        
        // Add to mesh
        triangleMesh.geom_instances.push_back(std::move(meshSurface));
        
        LOG(DBUG) << "Created surface " << surfIdx 
                  << " with " << triangles.size() << " triangles"
                  << ", material slot " << triangleMesh.geom_instances.back().material_slot;
    }
    
    if (triangleMesh.geom_instances.empty())
    {
        throw std::runtime_error("No valid surfaces found in model");
    }
    
    // Initialize material index buffer for multi-material models
    if (materialCount > 1 && !allMaterialIDs.empty())
    {
        materialIndexBuffer.initialize(ctx->cuCtx, cudau::BufferType::Device, allMaterialIDs);
    }
    
    // Initialize light distribution
    uint32_t totalTriangles = 0;
    for (const auto& surface : triangleMesh.geom_instances)
    {
        totalTriangles += surface.triangle_buffer.numElements();
    }
    
    emitterPrimDist.initialize(
        ctx->cuCtx, cudau::BufferType::Device, nullptr, totalTriangles);
    
    // Build the GAS
    buildGAS(ctx, 1);  // Default to 1 ray type
    
    LOG(INFO) << "Created TriangleMesh with " << triangleMesh.geom_instances.size()
              << " surfaces, " << vertices.size() << " vertices"
              << (triangleMesh.is_emissive ? " (EMISSIVE)" : "");
    
    // Register with area light handler if emissive
    if (triangleMesh.is_emissive && ctx->handlers->areaLight)
    {
        ctx->handlers->areaLight->addGeometry(node->getClientID());
    }
}

void OptiXTriangleModel2::buildGAS(RenderContextPtr ctx, uint32_t numRayTypes)
{
    // Create the GAS
    triangleMesh.gas = ctx->scene.createGeometryAccelerationStructure();
    
    // Configure GAS
    triangleMesh.gas.setConfiguration(
        hasDeformation() ? optixu::ASTradeoff::PreferFastBuild : optixu::ASTradeoff::PreferFastTrace,
        hasDeformation() ? optixu::AllowUpdate::Yes : optixu::AllowUpdate::No,
        hasDeformation() ? optixu::AllowCompaction::No : optixu::AllowCompaction::Yes);
    
    triangleMesh.gas.setNumMaterialSets(MATERIAL_SETS_V2);
    for (int i = 0; i < MATERIAL_SETS_V2; ++i)
    {
        triangleMesh.gas.setNumRayTypes(i, numRayTypes);
    }
    
    // Add all geometry instances to GAS
    for (auto& surface : triangleMesh.geom_instances)
    {
        triangleMesh.gas.addChild(surface.optix_geom_inst);
        surface.optix_geom_inst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        
        // Set geometry data
        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = triangleMesh.vertex_buffer.getDevicePointer();
        geomData.triangleBuffer = surface.triangle_buffer.getDevicePointer();
        surface.optix_geom_inst.setUserData(geomData);
    }
    
    // Prepare and build GAS
    OptixAccelBufferSizes bufferSizes;
    triangleMesh.gas.prepareForBuild(&bufferSizes);
    
    // Ensure scratch buffer is large enough
    size_t maxScratchSize = std::max(bufferSizes.tempSizeInBytes,
                                     bufferSizes.tempUpdateSizeInBytes);
    if (maxScratchSize > ctx->asBuildScratchMem.sizeInBytes())
    {
        ctx->asBuildScratchMem.resize(maxScratchSize, 1, ctx->cuStr);
    }
    
    // Allocate GAS memory
    triangleMesh.gas_mem.initialize(ctx->cuCtx, cudau::BufferType::Device, 
                                   bufferSizes.outputSizeInBytes, 1);
    
    // Build the GAS
    triangleMesh.gas.rebuild(ctx->cuStr, triangleMesh.gas_mem, ctx->asBuildScratchMem);
    
    LOG(INFO) << "Built GAS with handle: " << triangleMesh.gas.getHandle();
}

void OptiXTriangleModel2::extractVertexPositions(MatrixXf& V)
{
    uint32_t vertexCount = triangleMesh.vertex_buffer.numElements();
    V.resize(3, vertexCount);
    
    triangleMesh.vertex_buffer.map();
    const shared::Vertex* const vertices = triangleMesh.vertex_buffer.getMappedPointer();
    
    for (int i = 0; i < vertexCount; ++i)
    {
        const auto& v = vertices[i];
        V.col(i) = Eigen::Vector3f(v.position.x, v.position.y, v.position.z);
    }
    
    triangleMesh.vertex_buffer.unmap();
}

void OptiXTriangleModel2::extractTriangleIndices(MatrixXu& F)
{
    // Count total triangles across all surfaces
    uint32_t totalTriangles = 0;
    for (const auto& surface : triangleMesh.geom_instances)
    {
        totalTriangles += surface.triangle_buffer.numElements();
    }
    
    F.resize(3, totalTriangles);
    
    // Extract triangles from each surface
    uint32_t offset = 0;
    for (const auto& surface : triangleMesh.geom_instances)
    {
        surface.triangle_buffer.map();
        const shared::Triangle* const triangles = surface.triangle_buffer.getMappedPointer();
        uint32_t count = surface.triangle_buffer.numElements();
        
        for (uint32_t i = 0; i < count; ++i)
        {
            const auto& tri = triangles[i];
            F.col(offset + i) = Vector3u(tri.index0, tri.index1, tri.index2);
        }
        
        surface.triangle_buffer.unmap();
        offset += count;
    }
}

void OptiXTriangleModel2::enableDeformation(RenderContextPtr ctx)
{
    if (triangleMesh.vertex_buffer.numElements() == 0)
        return;
    
    LOG(DBUG) << "Enabling deformation for triangle mesh";
    originalVertexBuffer = triangleMesh.vertex_buffer.copy();
}

void OptiXTriangleModel2::updateDeformedGeometry(RenderContextPtr ctx)
{
    if (!hasDeformation())
        return;
    
    // Rebuild GAS after vertex modifications
    triangleMesh.gas.rebuild(ctx->cuStr, triangleMesh.gas_mem, ctx->asBuildScratchMem);
    
    LOG(DBUG) << "Updated deformed geometry";
}

optixu::Material OptiXTriangleModel2::getMaterialAt(uint32_t surfaceIdx, uint32_t matSetIdx)
{
    if (surfaceIdx >= triangleMesh.geom_instances.size())
    {
        LOG(WARNING) << "Invalid surface index: " << surfaceIdx;
        return optixu::Material();
    }
    
    return triangleMesh.geom_instances[surfaceIdx].optix_geom_inst.getMaterial(matSetIdx, 0);
}