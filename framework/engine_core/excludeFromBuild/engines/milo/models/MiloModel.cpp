#include "../../../model/ModelUtilities.h"
#include "MiloModel.h"
#include "../handlers/MiloMaterialHandler.h"

using Eigen::Vector2f;
using sabi::CgModel;
using sabi::CgModelList;
using sabi::CgModelPtr;
using sabi::CgModelSurface;

// Implementation for MiloTriangleModel
void MiloTriangleModel::createGeometry(RenderContextPtr ctx, RenderableNode& node, optixu::Scene* scene)
{
    CgModelPtr model = node->getModel();
    if (!model)
        throw std::runtime_error("Node has no cgModel: " + node->getName());

    // Get model path and texture folder
    fs::path modelPath = node->description().modelPath;
    SpaceTime& st = node->getSpaceTime();

    MatrixXu F;
    model->getAllSurfaceIndices(F);

    // Create geometry instance from scene
    if (!scene)
        throw std::runtime_error("Scene not provided for geometry creation");
    
    geomInst = scene->createGeometryInstance();

    // Create OptiX triangles
    std::vector<shared::Triangle> triangles;
    triangles.reserve(F.cols());
    for (int i = 0; i < F.cols(); ++i)
    {
        auto tri = F.col(i);
        triangles.emplace_back(shared::Triangle(tri(0), tri(1), tri(2)));
    }

    // Create OptiX vertices
    std::vector<shared::Vertex> vertices = populateVertices(model);
    if (!vertices.size())
        throw std::runtime_error("Populate vertices failed: " + modelPath.string());

    // Initialize GPU buffers
    triangleBuffer.initialize(ctx->getCudaContext(), cudau::BufferType::Device, triangles);
    vertexBuffer.initialize(ctx->getCudaContext(), cudau::BufferType::Device, vertices);

    if (model->VD.cols() > 0)
    {
        enableDeformation(ctx);

        // Verify buffer sizes
        if (originalVertexBuffer.numElements() != model->VD.cols())
        {
            LOG(WARNING) << "Vertex count mismatch - vertices: "
                         << originalVertexBuffer.numElements()
                         << " displacements: " << model->VD.cols();
        }
    }

    fs::path contentFolder = ctx->getPropertyService().renderProps->getVal<std::string>(RenderKey::ContentFolder);

    // 1 material per surface
    uint32_t materialCount = model->S.size();
    std::vector<uint8_t> materialIDs(F.cols());
    if (materialCount > 1)
    {
        uint8_t id = 0;
        uint32_t index = 0;
        for (const auto& s : model->S)
        {
            const MatrixXu& indices = s.indices();
            for (int i = 0; i < indices.cols(); ++i)
                materialIDs[index++] = id;

            ++id;
        }
    }

    if (materialCount > 1)
    {
        materialIndexBuffer.initialize(ctx->getCudaContext(), cudau::BufferType::Device, materialIDs);
        geomInst.setNumMaterials(materialCount, materialIndexBuffer, optixu::IndexSize::k1Byte);
    }
    else
        geomInst.setNumMaterials(1, optixu::BufferView());

    // Set vertex and triangle buffers
    geomInst.setVertexBuffer(vertexBuffer);
    geomInst.setTriangleBuffer(triangleBuffer);

    // Materials will be created separately by MiloMaterialHandler in MiloModelHandler::addCgModel
    
    emitterPrimDist.initialize(
        ctx->getCudaContext(), cudau::BufferType::Device, nullptr, static_cast<uint32_t>(triangles.size()));

    geomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
    
    // Set the geometry instance slot as user data
    // This allows the GPU code to find the geometry data in the global buffer
    LOG(DBUG) << "Setting geometry instance user data to slot " << geomInstSlot_;
    geomInst.setUserData(geomInstSlot_);
}

void MiloTriangleModel::createGAS(RenderContextPtr ctx, optixu::Scene* scene, uint32_t numRayTypes)
{
    if (!scene)
        throw std::runtime_error("Scene not provided for GAS creation");
        
    gasData.gas = scene->createGeometryAccelerationStructure();
    gasData.gas.setConfiguration(
        optixu::ASTradeoff::PreferFastBuild, // Changed from PreferFastTrace for deformable meshes
        optixu::AllowUpdate::Yes,            // Changed from No to Yes to allow updates
        optixu::AllowCompaction::No);        // Changed from Yes to No since we'll update frequently

    gasData.gas.setNumMaterialSets(MiloConstants::MATERIAL_SETS);
    for (int i = 0; i < MiloConstants::MATERIAL_SETS; ++i)
        gasData.gas.setNumRayTypes(i, numRayTypes);

    gasData.gas.addChild(geomInst);
    OptixAccelBufferSizes bufferSizes;
    gasData.gas.prepareForBuild(&bufferSizes);

    // Make sure scratch buffer is large enough for updates
    size_t maxScratchSize = std::max(bufferSizes.tempSizeInBytes,
                                    bufferSizes.tempUpdateSizeInBytes);
    
    cudau::Buffer& scratchMem = ctx->getASBuildScratchMem();
    if (maxScratchSize > scratchMem.sizeInBytes())
        scratchMem.resize(maxScratchSize, 1, ctx->getCudaStream());

    gasData.gasMem.initialize(ctx->getCudaContext(), cudau::BufferType::Device, bufferSizes.outputSizeInBytes, 1);
}

void MiloTriangleModel::extractVertexPositions(MatrixXf& V)
{
    uint32_t vertexCount = vertexBuffer.numElements();

    V.resize(3, vertexCount);

    vertexBuffer.map();

    const shared::Vertex* const vertices = vertexBuffer.getMappedPointer();
    for (int i = 0; i < vertexCount; ++i)
    {
        shared::Vertex v = vertices[i];
        V.col(i) = Eigen::Vector3f(v.position.x, v.position.y, v.position.z);
    }

    vertexBuffer.unmap();
}

void MiloTriangleModel::extractTriangleIndices(MatrixXu& F)
{
    uint32_t triangleCount = triangleBuffer.numElements();

    F.resize(3, triangleCount);

    triangleBuffer.map();

    const shared::Triangle* const triangles = triangleBuffer.getMappedPointer();
    for (int i = 0; i < triangleCount; ++i)
    {
        shared::Triangle t = triangles[i];
        F.col(i) = Eigen::Vector3<unsigned int>(t.index0, t.index1, t.index2);
    }

    triangleBuffer.unmap();
}