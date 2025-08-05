#include "MeshOps.h"

/*
    normal.cpp: Helper routines for computing vertex normals

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#// Define constants for multi-threading and floating point operations
#define GRAIN_SIZE 1024
#if defined(_WIN32)
#define RCPOVERFLOW_FLT 2.93873587705571876e-39f
#define RCPOVERFLOW_DBL 5.56268464626800345e-309
#else
#define RCPOVERFLOW_FLT 0x1p-128f
#define RCPOVERFLOW_DBL 0x1p-1024
#endif

#if defined(SINGLE_PRECISION)
#define RCPOVERFLOW RCPOVERFLOW_FLT
#else
#define RCPOVERFLOW RCPOVERFLOW_DBL
#endif

using Eigen::Vector3f;
using sabi::SpaceTime;

constexpr float epsilon = 1e-6;

// Prepare mesh for flat shading by duplicating vertices at edges
// Prepare mesh for flat shading by duplicating vertices at edges
void MeshOps::prepareForFlatShading (CgModelPtr& model)
{
    MatrixXf originalV = model->V;
    MatrixXf newV;
    MatrixXf originalUV0;
    bool hasUVs = model->UV0.cols() > 0;

    if (hasUVs)
    {
        originalUV0 = model->UV0;
    }

    // Count total number of triangles across all surfaces
    size_t totalTriangles = 0;
    for (const auto& surface : model->S)
    {
        totalTriangles += surface.triangleCount();
    }

    // Each triangle will have its own unique vertices
    newV.resize (3, totalTriangles * 3);
    if (hasUVs)
    {
        model->UV0.resize (2, totalTriangles * 3);
    }

    size_t currentVertex = 0;

    // Process each surface
    for (auto& surface : model->S)
    {
        MatrixXu& F = surface.F;
        MatrixXu newSurfaceF;
        newSurfaceF.resize (3, F.cols());

        // For each triangle
        for (int i = 0; i < F.cols(); i++)
        {
            // Copy vertices and UVs
            for (int j = 0; j < 3; j++)
            {
                newV.col (currentVertex + j) = originalV.col (F (j, i));
                if (hasUVs)
                {
                    model->UV0.col (currentVertex + j) = originalUV0.col (F (j, i));
                }
                newSurfaceF (j, i) = currentVertex + j;
            }
            currentVertex += 3;
        }

        surface.F = newSurfaceF;
        surface.vertexCount = newSurfaceF.cols() * 3;
    }

    model->V = newV;
}
#if 0
// Prepare mesh for flat shading by duplicating vertices at edges
void MeshOps::prepareForFlatShading (CgModelPtr& model)
{
    MatrixXf originalV = model->V;
    MatrixXf newV;
    MatrixXu newF;

    // Count total number of triangles across all surfaces
    size_t totalTriangles = 0;
    for (const auto& surface : model->S)
    {
        totalTriangles += surface.triangleCount();
    }

    // Each triangle will have its own unique vertices
    newV.resize (3, totalTriangles * 3);

    size_t currentVertex = 0;

    // Process each surface
    for (auto& surface : model->S)
    {
        MatrixXu& F = surface.F;
        MatrixXu newSurfaceF;
        newSurfaceF.resize (3, F.cols());

        // For each triangle
        for (int i = 0; i < F.cols(); i++)
        {
            // Copy vertices
            for (int j = 0; j < 3; j++)
            {
                newV.col (currentVertex + j) = originalV.col (F (j, i));
                newSurfaceF (j, i) = currentVertex + j;
            }
            currentVertex += 3;
        }

        surface.F = newSurfaceF;
        surface.vertexCount = newSurfaceF.cols() * 3;
    }

    model->V = newV;
}
#endif
// Generates vertex and face normals for a mesh
// If flatShaded is true, assigns face normals to vertices instead of computing smooth vertex normals
// Generates vertex and face normals for a mesh
// If flatShaded is true, duplicates vertices at shared edges to achieve flat shading
void MeshOps::generate_normals (const MatrixXu& F, const MatrixXf& V, MatrixXf& N, MatrixXf& FN,
                                bool deterministic, bool flatShaded)
{
    std::atomic<uint32_t> badFaces (0);

    N.resize (V.rows(), V.cols());
    N.setZero();

    FN.resize (F.rows(), F.cols());
    FN.setZero();

    BS::thread_pool pool;

    // First compute face normals
    auto computeFaceNormals = [&] (const uint32_t start, const uint32_t end)
    {
        for (uint32_t f = start; f < end; ++f)
        {
            Vector3f v0 = V.col (F (0, f));
            Vector3f v1 = V.col (F (1, f));
            Vector3f v2 = V.col (F (2, f));

            Vector3f d0 = v1 - v0;
            Vector3f d1 = v2 - v0;
            Vector3f fn = d0.cross (d1);

            Float norm = fn.norm();
            if (norm < RCPOVERFLOW)
            {
                badFaces++;
                continue;
            }

            fn /= norm;
            FN.col (f) = fn;

            if (flatShaded)
            {
                // For flat shading, assign the exact same face normal to all vertices
                // This ensures no normal interpolation across face boundaries
                N.col (F (0, f)) = fn;
                N.col (F (1, f)) = fn;
                N.col (F (2, f)) = fn;
            }
        }
    };

    pool.detach_blocks (0u, (uint32_t)F.cols(), computeFaceNormals, GRAIN_SIZE);
    pool.wait();

    if (!flatShaded)
    {
        // Clear vertex normals before computing smooth normals
        N.setZero();

        // Standard angle-weighted vertex normal computation for smooth shading
        auto computeSmoothNormals = [&] (const uint32_t start, const uint32_t end)
        {
            for (uint32_t f = start; f < end; ++f)
            {
                for (int i = 0; i < 3; ++i)
                {
                    Vector3f d0 = V.col (F ((i + 1) % 3, f)) - V.col (F (i, f));
                    Vector3f d1 = V.col (F ((i + 2) % 3, f)) - V.col (F (i, f));
                    Float angle = wabi::fast_acos (d0.dot (d1) / std::sqrt (d0.squaredNorm() * d1.squaredNorm()));
                    Vector3f fn = FN.col (f) * angle;

                    for (uint32_t k = 0; k < 3; ++k)
                    {
                        mace::atomicAdd (&N.coeffRef (k, F (i, f)), fn[k]);
                    }
                }
            }
        };

        pool.detach_blocks (0u, (uint32_t)F.cols(), computeSmoothNormals, GRAIN_SIZE);
        pool.wait();

        // Normalize smooth vertex normals
        auto normalizeNormals = [&] (const uint32_t start, const uint32_t end)
        {
            for (uint32_t i = start; i < end; ++i)
            {
                Float norm = N.col (i).norm();
                if (norm < RCPOVERFLOW)
                {
                    N.col (i) = Vector3f::UnitX();
                }
                else
                {
                    N.col (i) /= norm;
                }
            }
        };

        pool.detach_blocks (0u, (uint32_t)V.cols(), normalizeNormals);
        pool.wait();
    }
}
#if 0

void MeshOps::generate_normals (const MatrixXu& F, const MatrixXf& V, MatrixXf& N, MatrixXf& FN, bool deterministic, bool flatShaded)
{
    // ScopedStopWatch sw("GENERATE NORMALS"); // Start timer

    std::atomic<uint32_t> badFaces (0); // Counter for degenerate faces

    N.resize (V.rows(), V.cols()); // Prepare vertex normal matrix
    N.setZero();

    FN.resize (F.rows(), F.cols()); // Prepare face normal matrix
    FN.setZero();

    BS::thread_pool pool; // Initialize thread pool

    // Multi-threaded computation of face and vertex normals
    auto map = [&] (const uint32_t start, const uint32_t end)
    {
        for (uint32_t f = start; f < end; ++f)
        {
            Vector3f fn = Vector3f::Zero();
            for (int i = 0; i < 3; ++i)
            {
                Vector3f v0 = V.col (F (i, f)),
                         v1 = V.col (F ((i + 1) % 3, f)),
                         v2 = V.col (F ((i + 2) % 3, f)),
                         d0 = v1 - v0,
                         d1 = v2 - v0;

                if (i == 0)
                {
                    fn = d0.cross (d1);
                    Float norm = fn.norm();
                    if (norm < RCPOVERFLOW)
                    {
                        badFaces++;
                        break;
                    }
                    FN.col (f) = fn.normalized();
                    fn /= norm;
                }

                Float angle = wabi::fast_acos (d0.dot (d1) / std::sqrt (d0.squaredNorm() * d1.squaredNorm()));
                for (uint32_t k = 0; k < 3; ++k)
                    mace::atomicAdd (&N.coeffRef (k, F (i, f)), fn[k] * angle);
            }
        }
    };

    pool.detach_blocks (0u, (uint32_t)F.cols(), map, GRAIN_SIZE); // Execute in parallel

    // must wait here because the normalize task depends on this task being completed
    pool.wait();

    // Normalize the vertex normals
    pool.detach_blocks (0u, (uint32_t)V.cols(),
                        [&] (const uint32_t start, const uint32_t end)
                        {
                            for (uint32_t i = start; i < end; ++i)
                            {
                                Float norm = N.col (i).norm();
                                if (norm < RCPOVERFLOW)
                                {
                                    N.col (i) = Vector3f::UnitX();
                                }
                                else
                                {
                                    N.col (i) /= norm;
                                }
                            }
                        });

    pool.wait();
}
#endif

void MeshOps::generate_normals (CgModelPtr& cgModel, bool flatShaded)
{
    if (flatShaded)
    {
        // First unweld vertices so each face has unique vertices
        prepareForFlatShading (cgModel);
    }

    MatrixXu allIndices;
    cgModel->getAllSurfaceIndices (allIndices);
    MeshOps::generate_normals (allIndices, cgModel->V, cgModel->N, cgModel->FN, false, flatShaded);
}

void MeshOps::processCgModel (RenderableNode& node, MeshOptions meshOptions, LoadStrategyPtr loadStrategy)
{
    CgModelPtr model = node->getModel();
    if (!model)
    {
        LOG (CRITICAL) << "Node does not have a cgModel";
        return;
    }

    SpaceTime& spacetime = node->getSpaceTime();

    AlignedBox3f modelBound;
    modelBound.min() = model->V.rowwise().minCoeff();
    modelBound.max() = model->V.rowwise().maxCoeff();
    float scale = 1.0f;

    if (modelBound.isEmpty())
        throw std::runtime_error ("Empty bounding box detected for " + node->getName());

    // this might change scale!
    if ((meshOptions & MeshOptions::NormalizeSize) == MeshOptions::NormalizeSize)
        MeshOps::normalizeSize (model, modelBound, scale);

    if ((meshOptions & MeshOptions::CenterVertices) == MeshOptions::CenterVertices)
        MeshOps::centerVertices (model, modelBound, scale);

    // recalc new modelBound
    modelBound.min() = model->V.rowwise().minCoeff();
    modelBound.max() = model->V.rowwise().maxCoeff();
    spacetime.modelBound = modelBound;

    if ((meshOptions & MeshOptions::LoadStrategy) == MeshOptions::LoadStrategy)
        loadStrategy->addNextItem (spacetime);

    spacetime.updateWorldBounds (true);

    if ((meshOptions & MeshOptions::RestOnGround) == MeshOptions::RestOnGround)
        spacetime.worldTransform.translation().y() = -spacetime.worldBound.min().y();

    spacetime.updateWorldBounds (true);
    spacetime.startTransform = spacetime.worldTransform;

    // create vertex normals and face normals  if they aren't there
    if (!model->N.cols() || !model->FN.cols())
    {
        MatrixXu allIndices;
        model->getAllSurfaceIndices (allIndices);
        MeshOps::generate_normals (allIndices, model->V, model->N, model->FN, false);
    }

    if (!model->isValid())
        throw std::runtime_error ("Invalid model");
}

void MeshOps::unweldMesh (CgModelPtr& model)
{
    // model->triangleCount might need to be computed
    MatrixXu allTris;
    if (!model->triangleCount())
    {
        model->getAllSurfaceIndices (allTris);
    }

    MatrixXf newVertices;

    // unwelded vertex count is 3 * total number of triangles
    newVertices.resize (3, model->triangleCount() * 3);

    // add all vertices in each triangle to unwlded vertics array VU
    uint32_t nextVertexIndex = 0;
    for (auto& s : model->S)
    {
        uint32_t triCount = s.triangleCount();
        MatrixXu& tris = s.indices();

        MatrixXu newTris;
        newTris.resize (3, triCount);

        for (int i = 0; i < triCount; i++)
        {
            Vector3u tri = tris.col (i);

            // get the 3 vertex positions for this tri
            const Vector3f& p0 = model->V.col (tri.x());
            const Vector3f& p1 = model->V.col (tri.y());
            const Vector3f& p2 = model->V.col (tri.z());

            // add this tris 3 vertices to
            // the new vertex array and make
            // a new triangle
            Vector3u newTri;

            newTri.x() = nextVertexIndex;
            newVertices.col (nextVertexIndex++) = p0;

            newTri.y() = nextVertexIndex;
            newVertices.col (nextVertexIndex++) = p1;

            newTri.z() = nextVertexIndex;
            newVertices.col (nextVertexIndex++) = p2;

            newTris.col (i) = newTri;
        }

        tris = newTris;

        // set this or the renderer will crash FIXME
        s.vertexCount = newVertices.cols();
    }

    model->V = newVertices;
}

void MeshOps::centerVertices (CgModelPtr model, const AlignedBox3f& modelBound, float scale)
{
    int pointCount = model->V.cols();
    Vector3f center = modelBound.center();
    for (int i = 0; i < pointCount; i++)
    {
        Vector3f pnt = model->V.col (i);
        pnt -= center;
        pnt *= scale;
        model->V.col (i) = pnt;
    }
}

void MeshOps::normalizeSize (CgModelPtr model, const AlignedBox3f& modelBound, float& scale)
{
    Eigen::Vector3f edges = modelBound.max() - modelBound.min();
    float maxEdge = std::max (edges.x(), std::max (edges.y(), edges.z()));
    scale = 1.0f / maxEdge; // max
}

void MeshOps::resizeModel (CgModelPtr model, const Eigen::Vector3f& targetSize)
{
    // Calculate the current bounding box of the model
    Eigen::AlignedBox3f modelBound;
    modelBound.min() = model->V.rowwise().minCoeff();
    modelBound.max() = model->V.rowwise().maxCoeff();

    // Calculate the current size of the model
    Eigen::Vector3f currentSize = modelBound.max() - modelBound.min();

    // Identify degenerate dimensions (within an epsilon)

    bool isXDegenerate = currentSize.x() < epsilon;
    bool isYDegenerate = currentSize.y() < epsilon;
    bool isZDegenerate = currentSize.z() < epsilon;

    // Calculate scale factors for each non-degenerate dimension
    Eigen::Vector3f scaleFactor (1.0, 1.0, 1.0); // Initialize to 1.0 for no scaling effect by default
    if (!isXDegenerate) scaleFactor.x() = targetSize.x() / currentSize.x();
    if (!isYDegenerate) scaleFactor.y() = targetSize.y() / currentSize.y();
    if (!isZDegenerate) scaleFactor.z() = targetSize.z() / currentSize.z();

    // Apply scaling to each vertex
    for (int i = 0; i < model->V.cols(); ++i)
    {
        Eigen::Vector3f p = model->V.col (i);
        p.x() *= scaleFactor.x();
        p.y() *= scaleFactor.y();
        p.z() *= scaleFactor.z();
        model->V.col (i) = p;
    }
}

CgModelPtr MeshOps::createTexturedTriangle (const fs::path& pngImagePath)
{
    CgModelPtr triangle = CgModel::create();

    // Define the vertices of the triangle to match redTri.lwo
    triangle->V.resize (3, 3);
    triangle->V.col (0) = Eigen::Vector3f (0, 1, 0);   // (0.0, 1.0, 0.0)
    triangle->V.col (1) = Eigen::Vector3f (1, -1, 0);  // (1.0, -1.0, 0.0)
    triangle->V.col (2) = Eigen::Vector3f (-1, -1, 0); // (-1.0, -1.0, 0.0)

    // Create a single surface for the triangle
    triangle->S.resize (1);
    auto& surface = triangle->S[0];
    surface.name = "FRED";

    // Set up indices for this face
    surface.F.resize (3, 1);
    surface.F (0, 0) = 0;
    surface.F (1, 0) = 1;
    surface.F (2, 0) = 2;

    // Set up the material properties using CgMaterial
    auto& material = surface.cgMaterial;
    material.name = "Textured_Material";
    material.core.baseColor = Vector3f (1.0f, 1.0f, 1.0f); // White color (will be multiplied with texture)
    material.metallic.metallic = 0.0f;
    material.core.roughness = 0.5f;

    // Create and set up the sampler
    CgSampler sampler;
    sampler.magFilter = CgFilter::Linear;             // 9729
    sampler.minFilter = CgFilter::LinearMipMapLinear; // 9987
    sampler.wrapS = CgWrap::Repeat;                   // 10497
    sampler.wrapT = CgWrap::Repeat;                   // 10497
    triangle->cgSamplers.push_back (sampler);

    // Create and set up the texture
    CgTexture texture;
    texture.imageIndex = triangle->cgImages.size(); // Index of the image we're about to add
    triangle->cgTextures.push_back (texture);

    // Create and set up the image
    CgImage image;
    image.uri = pngImagePath.generic_string(); // Replace with actual texture path
    image.mimeType = "image/png";              // Adjust based on your image type
    triangle->cgImages.push_back (image);

    // Set the base color texture
    CgTextureInfo textureInfo;
    textureInfo.textureIndex = triangle->cgTextures.size() - 1; // Index of the texture we just added
    textureInfo.texCoordIndex = 0;                              // Use the first set of texture coordinates (UV0)
    material.core.baseColorTexture = textureInfo;

    surface.vertexCount = 3;

    // Update the total triangle count
    triangle->triCount = 1;

    // Create vertex normals (flat normal for simplicity)
    triangle->N.resize (3, 3);
    Eigen::Vector3f edge1 = triangle->V.col (1) - triangle->V.col (0);
    Eigen::Vector3f edge2 = triangle->V.col (2) - triangle->V.col (0);
    Eigen::Vector3f normal = edge1.cross (edge2).normalized();
    triangle->N.col (0) = normal;
    triangle->N.col (1) = normal;
    triangle->N.col (2) = normal;

    // Add UV coordinates matching LightWave's exact UV values
    triangle->UV0.resize (2, 3);
    triangle->UV0.col (0) = Eigen::Vector2f (0.5f, 1.0f); // Top vertex
    triangle->UV0.col (1) = Eigen::Vector2f (1.0f, 0.0f); // Right vertex
    triangle->UV0.col (2) = Eigen::Vector2f (0.0f, 0.0f); // Left vertex

    LOG (DBUG) << "Created a single triangle with a textured surface";
    return triangle;
}

#if 0
CgModelPtr MeshOps::createTexturedTriangle (const fs::path& pngImagePath)
{
    CgModelPtr triangle = CgModel::create();
    // Define the vertices of the triangle to match redTri.lwo
    triangle->V.resize (3, 3);
    triangle->V.col (0) = Eigen::Vector3f (0, 1, 0);   // (0.0, 1.0, 0.0)
    triangle->V.col (1) = Eigen::Vector3f (1, -1, 0);  // (1.0, -1.0, 0.0)
    triangle->V.col (2) = Eigen::Vector3f (-1, -1, 0); // (-1.0, -1.0, 0.0)

    // Create a single surface for the triangle
    triangle->S.resize (1);
    auto& surface = triangle->S[0];
    surface.name = "FRED";

    // Set up indices for this face
    surface.F.resize (3, 1);
    surface.F (0, 0) = 0;
    surface.F (1, 0) = 1;
    surface.F (2, 0) = 2;

    // Set up the material properties
    surface.material.name = "Textured_Material";
    surface.material.pbrMetallicRoughness.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f}; // White color (will be multiplied with texture)
    surface.material.pbrMetallicRoughness.metallicFactor = 0.0f;
    surface.material.pbrMetallicRoughness.roughnessFactor = 0.5f;

    // Create and set up the sampler
    Sampler sampler;
    sampler.magFilter = 9729; // GL_LINEAR
    sampler.minFilter = 9987; // GL_LINEAR_MIPMAP_LINEAR
    sampler.wrapS = 10497;    // GL_REPEAT
    sampler.wrapT = 10497;    // GL_REPEAT
    triangle->samplers.push_back (sampler);

    // Create and set up the texture
    Texture texture;
    texture.source = triangle->images.size(); // Index of the image we're about to add
    triangle->textures.push_back (texture);

    // Create and set up the image
    Image image;
    image.uri = pngImagePath.generic_string(); // Replace with actual texture path
    image.mimeType = "image/png";              // Adjust based on your image type
    triangle->images.push_back (image);

    // Set the base color texture
    TextureInfo textureInfo;
    textureInfo.textureIndex = triangle->textures.size() - 1; // Index of the texture we just added
    textureInfo.texCoord = 0;                                 // Use the first set of texture coordinates (UV0)
    surface.material.pbrMetallicRoughness.baseColorTexture = textureInfo;

    surface.vertexCount = 3;
    // Update the total triangle count
    triangle->triCount = 1;

    // Create vertex normals (flat normal for simplicity)
    triangle->N.resize (3, 3);
    Eigen::Vector3f edge1 = triangle->V.col (1) - triangle->V.col (0);
    Eigen::Vector3f edge2 = triangle->V.col (2) - triangle->V.col (0);
    Eigen::Vector3f normal = edge1.cross (edge2).normalized();
    triangle->N.col (0) = normal;
    triangle->N.col (1) = normal;
    triangle->N.col (2) = normal;

    // Add UV coordinates matching LightWave's exact UV values
    triangle->UV0.resize (2, 3);
    triangle->UV0.col (0) = Eigen::Vector2f (0.5f, 1.0f); // Top vertex
    triangle->UV0.col (1) = Eigen::Vector2f (1.0f, 0.0f); // Right vertex
    triangle->UV0.col (2) = Eigen::Vector2f (0.0f, 0.0f); // Left vertex

    LOG (DBUG) << "Created a single triangle with a textured surface";
    return triangle;
}
#endif
CgModelPtr MeshOps::createTriangle()
{
    CgModelPtr triangle = CgModel::create();

    // Define the vertices of the triangle to match redTri.lwo
    triangle->V.resize (3, 3);
    triangle->V.col (0) = Eigen::Vector3f (-1, -1, 0); // (-1.0, -1.0, 0.0)
    triangle->V.col (1) = Eigen::Vector3f (0, 1, 0);   // (0.0, 1.0, 0.0)
    triangle->V.col (2) = Eigen::Vector3f (1, -1, 0);  // (1.0, -1.0, 0.0)

    // Create a single surface for the triangle
    triangle->S.resize (1);

    auto& surface = triangle->S[0];
    surface.name = "Triangle_Face";

    // Set up indices for this face
    surface.F.resize (3, 1);
    surface.F (0, 0) = 1;
    surface.F (1, 0) = 2;
    surface.F (2, 0) = 0;

    // Set up the material properties
    surface.material.name = "Blue_Material";
    surface.material.pbrMetallicRoughness.baseColorFactor = {0.0f, 0.0f, 1.0f, 1.0f}; // blue color
    surface.material.pbrMetallicRoughness.metallicFactor = 0.0f;
    surface.material.pbrMetallicRoughness.roughnessFactor = 0.0f;

    surface.vertexCount = 3;

    // Update the total triangle count
    triangle->triCount = 1;

    // Create vertex normals (flat normal for simplicity)
    triangle->N.resize (3, 3);
    Eigen::Vector3f edge1 = triangle->V.col (1) - triangle->V.col (0);
    Eigen::Vector3f edge2 = triangle->V.col (2) - triangle->V.col (0);
    Eigen::Vector3f normal = edge1.cross (edge2).normalized();
    triangle->N.col (0) = normal;
    triangle->N.col (1) = normal;
    triangle->N.col (2) = normal;

    // Add UV coordinates (simple planar mapping)
    triangle->UV0.resize (2, 3);
    triangle->UV0.col (0) = Eigen::Vector2f (0, 0);
    triangle->UV0.col (1) = Eigen::Vector2f (1, 0);
    triangle->UV0.col (2) = Eigen::Vector2f (0.5f, 1);

    // LOG (DBUG) << "Created a single triangle with a red surface";

    return triangle;
}

// Creates a cube with vertices and normals matching LightWave's internal format
// Creates a cube with vertices and normals matching LightWave's internal format
// Creates a cube with vertices and normals matching LightWave's format
CgModelPtr MeshOps::createCube (float size)
{
    auto model = CgModel::create();
    const float s = size * 0.5f;

    // Define vertices in exact LightWave point order (0-based indices)
    model->V.resize (3, 8);
    model->V.col (0) = Vector3f (-s, -s, -s); // Back bottom left
    model->V.col (1) = Vector3f (-s, -s, s);  // Front bottom left
    model->V.col (2) = Vector3f (-s, s, -s);  // Back top left
    model->V.col (3) = Vector3f (-s, s, s);   // Front top left
    model->V.col (4) = Vector3f (s, -s, -s);  // Back bottom right
    model->V.col (5) = Vector3f (s, -s, s);   // Front bottom right
    model->V.col (6) = Vector3f (s, s, -s);   // Back top right
    model->V.col (7) = Vector3f (s, s, s);    // Front top right

    // Set normals
    model->N.resize (3, 8);
    for (int i = 0; i < 8; i++)
    {
        model->N.col (i) = -model->V.col (i).normalized();
    }

    // Create surface with triangles
    CgModelSurface surface;
    surface.F.resize (3, 12);

    // Define faces using the corrected vertex order
    int faces[12][3] = {
        {0, 1, 3}, {0, 3, 2}, // First two triangles from hex dump
        {1, 5, 7},
        {1, 7, 3}, // Next two triangles
        {5, 4, 6},
        {5, 6, 7}, // And so on...
        {4, 0, 2},
        {4, 2, 6},
        {2, 3, 7},
        {2, 7, 6},
        {1, 0, 4},
        {1, 4, 5}};

    for (int i = 0; i < 12; i++)
    {
        surface.F.col (i) = Vector3u (faces[i][0], faces[i][1], faces[i][2]);
    }

    model->S.push_back (surface);
    return model;
}

#if 0
CgModelPtr MeshOps::createCube()
{
    CgModelPtr cube = CgModel::create();

    // Define the vertices of the cube (shared vertices for adjacent faces)
    cube->V.resize (3, 8);
    cube->V << -.5, -.5, -.5, -.5, .5, .5, .5, .5,
        -.5, -.5, .5, .5, -.5, -.5, .5, .5,
        -.5, .5, -.5, .5, -.5, .5, -.5, .5;

    // Create 6 surfaces, one for each face of the cube
    cube->S.resize (6);

    // Define the indices for each face (now using triangles)
#if 0
    std::vector<std::vector<unsigned int>> faceIndices = {
        {0, 1, 2, 2, 1, 3}, // Front face
        {4, 6, 5, 5, 6, 7}, // Back face
        {0, 4, 1, 1, 4, 5}, // Left face
        {2, 3, 6, 6, 3, 7}, // Right face
        {0, 2, 4, 4, 2, 6}, // Bottom face
        {1, 5, 3, 3, 5, 7}  // Top face
    };
#endif
    std::vector<std::vector<unsigned int>> faceIndices = {
        {0, 1, 3, 1, 2, 3}, // Directly copied from LW's hex dump
        {4, 5, 6, 5, 7, 6}, // Directly copied from LW's hex dump
        {0, 1, 4, 1, 5, 4}, // Directly copied from LW's hex dump
        {2, 3, 6, 3, 7, 6}, // Directly copied from LW's hex dump
        {4, 0, 3, 3, 7, 4}, // Directly copied from LW's hex dump
        {5, 4, 7, 4, 6, 7}  // Directly copied from LW's hex dump
    };
    // Define colors for each face
    std::vector<std::array<float, 4>> colors = {
        {1.0f, 0.0f, 0.0f, 1.0f}, // Red
        {0.0f, 1.0f, 0.0f, 1.0f}, // Green
        {0.0f, 0.0f, 1.0f, 1.0f}, // Blue
        {1.0f, 1.0f, 0.0f, 1.0f}, // Yellow
        {1.0f, 0.0f, 1.0f, 1.0f}, // Magenta
        {0.0f, 1.0f, 1.0f, 1.0f}  // Cyan
    };

    for (int i = 0; i < 6; ++i)
    {
        auto& surface = cube->S[i];
        surface.name = "Face_" + std::to_string (i+1);

        // Set up indices for this face
        surface.F.resize (3, 2);
        for (int j = 0; j < 2; ++j)
        {
            surface.F.col (j) << faceIndices[i][j * 3], faceIndices[i][j * 3 + 1], faceIndices[i][j * 3 + 2];
        }

        surface.material.name = "Material_" + std::to_string (i);
        surface.material.pbrMetallicRoughness.baseColorFactor = colors[i];
        surface.material.pbrMetallicRoughness.metallicFactor = 0.0f;
        surface.material.pbrMetallicRoughness.roughnessFactor = 0.5f;

        surface.vertexCount = 4;
    }

    // Update the total triangle count
    cube->triCount = 12; // 2 triangles per face, 6 faces

    // Create vertex normals (not averaged, for simplicity)
    cube->N = cube->V.colwise().normalized();

    // Add UV coordinates (simple planar mapping for each face)
    //cube->UV0.resize (2, 24); // 4 vertices per face, 6 faces
    //for (int i = 0; i < 6; ++i)
    //{
    //    cube->UV0.col (i * 4 + 0) << 0, 0;
    //    cube->UV0.col (i * 4 + 1) << 1, 0;
    //    cube->UV0.col (i * 4 + 2) << 0, 1;
    //    cube->UV0.col (i * 4 + 3) << 1, 1;
    //}

    LOG (DBUG) << "Created a cube with 6 surfaces using shared vertices and triangles";

    return cube;
}
#endif
CgModelPtr MeshOps::createTexturedQuad (const fs::path& pngImagePath)
{
    CgModelPtr quad = CgModel::create();

    // Define the vertices of the quad (2 triangles)
    quad->V.resize (3, 4);
    quad->V.col (0) = Eigen::Vector3f (-1, -1, 0); // Bottom-left
    quad->V.col (1) = Eigen::Vector3f (1, -1, 0);  // Bottom-right
    quad->V.col (2) = Eigen::Vector3f (1, 1, 0);   // Top-right
    quad->V.col (3) = Eigen::Vector3f (-1, 1, 0);  // Top-left

    // Create a single surface for the quad
    quad->S.resize (1);
    auto& surface = quad->S[0];
    surface.name = "Quad_Face";

    // Set up indices for the two triangles
    surface.F.resize (3, 2);
    surface.F.col (0) << 0, 1, 2; // First triangle
    surface.F.col (1) << 0, 2, 3; // Second triangle

    // Set up the material properties
    surface.material.name = "Textured_Material";
    surface.material.pbrMetallicRoughness.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f}; // White color (will be multiplied with texture)
    surface.material.pbrMetallicRoughness.metallicFactor = 0.0f;
    surface.material.pbrMetallicRoughness.roughnessFactor = 0.5f;

    // Create and set up the sampler
    Sampler sampler;
    sampler.magFilter = 9729; // GL_LINEAR
    sampler.minFilter = 9987; // GL_LINEAR_MIPMAP_LINEAR
    sampler.wrapS = 10497;    // GL_REPEAT
    sampler.wrapT = 10497;    // GL_REPEAT
    quad->samplers.push_back (sampler);

    // Create and set up the texture
    Texture texture;
    texture.source = quad->images.size();        // Index of the image we're about to add
    texture.sampler = quad->samplers.size() - 1; // Index of the sampler we just added
    quad->textures.push_back (texture);

    // Create and set up the image
    Image image;
    image.uri = pngImagePath.string();
    image.mimeType = "image/png";
    quad->images.push_back (image);

    // Set the base color texture
    TextureInfo textureInfo;
    textureInfo.textureIndex = quad->textures.size() - 1; // Index of the texture we just added
    textureInfo.texCoord = 0;                             // Use the first set of texture coordinates (UV0)
    surface.material.pbrMetallicRoughness.baseColorTexture = textureInfo;

    surface.vertexCount = 4;

    // Update the total triangle count
    quad->triCount = 2;

    // Create vertex normals (flat normal for simplicity)
    quad->N.resize (3, 4);
    Eigen::Vector3f normal (0, 0, 1); // Facing positive Z direction
    quad->N.col (0) = normal;
    quad->N.col (1) = normal;
    quad->N.col (2) = normal;
    quad->N.col (3) = normal;

    quad->UV0.resize (2, 4);
    quad->UV0.col (0) = Eigen::Vector2f (0, 1); // Bottom-left
    quad->UV0.col (1) = Eigen::Vector2f (1, 1); // Bottom-right
    quad->UV0.col (2) = Eigen::Vector2f (1, 0); // Top-right
    quad->UV0.col (3) = Eigen::Vector2f (0, 0); // Top-left

    LOG (DBUG) << "Created a quad with a textured surface using image: " << pngImagePath.string();
    return quad;
}

void MeshOps::debugCgModel (const CgModelPtr& model, DebugFlags flags)
{
    if (!model)
    {
        LOG (WARNING) << "CgModel is null!";
        return;
    }

    LOG (INFO) << "Debugging CgModel:";

    // Vertices
    if ((flags & DebugFlags::Vertices) == DebugFlags::Vertices)
    {
        LOG (INFO) << "Vertices: " << model->V.cols();
        for (int i = 0; i < std::min (24, static_cast<int> (model->V.cols())); ++i)
        {
            LOG (INFO) << "  V" << i << ": (" << model->V (0, i) << ", " << model->V (1, i) << ", " << model->V (2, i) << ")";
        }
        if (model->V.cols() > 24) LOG (INFO) << "  ... (truncated)";
    }

    // Normals
    if ((flags & DebugFlags::Normals) == DebugFlags::Normals)
    {
        LOG (INFO) << "Normals: " << model->N.cols();
        for (int i = 0; i < std::min (24, static_cast<int> (model->N.cols())); ++i)
        {
            LOG (INFO) << "  N" << i << ": (" << model->N (0, i) << ", " << model->N (1, i) << ", " << model->N (2, i) << ")";
        }
        if (model->N.cols() > 24) LOG (INFO) << "  ... (truncated)";
    }

    // UVs
    if ((flags & DebugFlags::UVs) == DebugFlags::UVs)
    {
        LOG (INFO) << "UVs: " << model->UV0.cols();
        for (int i = 0; i < std::min (36, static_cast<int> (model->UV0.cols())); ++i)
        {
            LOG (INFO) << "  UV" << i << ": (" << model->UV0 (0, i) << ", " << model->UV0 (1, i) << ")";
        }
        if (model->UV0.cols() > 36) LOG (INFO) << "  ... (truncated)";
    }

    // Triangle Indices
    if ((flags & DebugFlags::Indices) == DebugFlags::Indices)
    {
        LOG (INFO) << "Triangle Indices:";
        for (size_t s = 0; s < model->S.size(); ++s)
        {
            const auto& surface = model->S[s];
            LOG (INFO) << "  Surface " << s << " indices:";
            for (int i = 0; i < std::min (24, static_cast<int> (surface.F.cols())); ++i)
            {
                const auto& tri = surface.F.col (i);
                LOG (INFO) << "    Triangle " << i << ": ("
                           << tri[0] << ", " << tri[1] << ", " << tri[2] << ")";
            }
            if (surface.F.cols() > 24) LOG (INFO) << "    ... (truncated)";
        }
    }

    // Surfaces and Materials
    if ((flags & DebugFlags::Surfaces) == DebugFlags::Surfaces)
    {
        LOG (INFO) << "Surfaces: " << model->S.size();
        for (size_t i = 0; i < model->S.size(); ++i)
        {
            const auto& surface = model->S[i];
            LOG (INFO) << "Surface " << i << "::" << surface.name << ":";
            LOG (INFO) << "  Triangles: " << surface.triangleCount();

            if ((flags & DebugFlags::Materials) == DebugFlags::Materials)
            {
                const auto& mat = surface.cgMaterial;
                LOG (INFO) << "  Material:";
                LOG (INFO) << "    Name: " << mat.name;
                LOG (INFO) << "    Base Color: "
                           << mat.core.baseColor.x() << ", "
                           << mat.core.baseColor.y() << ", "
                           << mat.core.baseColor.z();
                LOG (INFO) << "    Metallic: " << mat.metallic.metallic;
                LOG (INFO) << "    Roughness: " << mat.core.roughness;
                LOG (INFO) << "    Specular: " << mat.core.specular;
                LOG (INFO) << "    Thin Walled: " << (mat.transparency.thin ? "Yes" : "No");
                LOG (INFO) << "    Transparency: " << mat.transparency.transparency;
                LOG (INFO) << "    IOR: " << mat.transparency.refractionIndex;

                if (mat.core.baseColorTexture)
                {
                    LOG (INFO) << "    Base Color Texture Index: " << mat.core.baseColorTexture->textureIndex;
                }
            }
        }
    }

    // Textures
    if ((flags & DebugFlags::Textures) == DebugFlags::Textures)
    {
        LOG (INFO) << "Textures: " << model->cgTextures.size();
        for (size_t i = 0; i < model->cgTextures.size(); ++i)
        {
            const auto& texture = model->cgTextures[i];
            LOG (INFO) << "Texture " << i << ":";
            LOG (INFO) << "  Name: " << texture.name;
            if (texture.imageIndex)
                LOG (INFO) << "  Image Index: " << *texture.imageIndex;
            if (texture.samplerIndex)
                LOG (INFO) << "  Sampler Index: " << *texture.samplerIndex;
        }
    }

    // Images
    if ((flags & DebugFlags::Images) == DebugFlags::Images)
    {
        LOG (INFO) << "Images: " << model->cgImages.size();
        for (size_t i = 0; i < model->cgImages.size(); ++i)
        {
            const auto& image = model->cgImages[i];
            LOG (INFO) << "Image " << i << ":";
            LOG (INFO) << "  Name: " << image.name;
            LOG (INFO) << "  URI: " << image.uri;
            LOG (INFO) << "  MIME Type: " << image.mimeType;
            LOG (INFO) << "  Index: " << image.index;
        }
    }

    // Samplers
    if ((flags & DebugFlags::Samplers) == DebugFlags::Samplers)
    {
        LOG (INFO) << "Samplers: " << model->cgSamplers.size();
        for (size_t i = 0; i < model->cgSamplers.size(); ++i)
        {
            const auto& sampler = model->cgSamplers[i];
            LOG (INFO) << "Sampler " << i << ":";
            if (sampler.magFilter)
                LOG (INFO) << "  Mag Filter: " << static_cast<int> (*sampler.magFilter);
            if (sampler.minFilter)
                LOG (INFO) << "  Min Filter: " << static_cast<int> (*sampler.minFilter);
            LOG (INFO) << "  Wrap S: " << static_cast<int> (sampler.wrapS);
            LOG (INFO) << "  Wrap T: " << static_cast<int> (sampler.wrapT);
            LOG (INFO) << "  Name: " << sampler.name;
        }
    }
}

// Writes UV coordinates from a CgModel to a text file
// Returns true if successful, false if file couldn't be opened or model is invalid
bool MeshOps::dumpUVCoordinates (const CgModelPtr& model, const fs::path& outputPath)
{
    if (!model || model->UV0.cols() == 0)
    {
        LOG (WARNING) << "Invalid model or no UV coordinates present";
        return false;
    }

    std::ofstream outFile (outputPath);
    if (!outFile)
    {
        LOG (WARNING) << "Failed to open output file: " << outputPath;
        return false;
    }

    outFile << "Vertex Count: " << model->UV0.cols() << "\n\n";
    outFile << "Format: VertexIndex U V\n\n";

    for (int i = 0; i < model->UV0.cols(); ++i)
    {
        outFile << std::setw (6) << i << " "
                << std::setprecision (6) << std::fixed
                << std::setw (9) << model->UV0 (0, i) << " "
                << std::setw (9) << model->UV0 (1, i) << "\n";
    }

    // Write UV1 coordinates if they exist
    if (model->UV1.cols() > 0)
    {
        outFile << "\nSecondary UV Set (UV1):\n\n";
        for (int i = 0; i < model->UV1.cols(); ++i)
        {
            outFile << std::setw (6) << i << " "
                    << std::setprecision (6) << std::fixed
                    << std::setw (9) << model->UV1 (0, i) << " "
                    << std::setw (9) << model->UV1 (1, i) << "\n";
        }
    }

    return true;
}


// Creates a luminous rectangle mesh light from 2 triangles
CgModelPtr MeshOps::createLuminousRectangle (float width, float height,
                                             const Eigen::Vector3f& luminousColor,
                                             float luminousIntensity)
{
    CgModelPtr rectangle = CgModel::create();

    float halfWidth = width * 0.5f;
    float halfHeight = height * 0.5f;

    // Define the vertices of the rectangle (2 triangles forming a quad)
    rectangle->V.resize (3, 4);
    rectangle->V.col (0) = Eigen::Vector3f (-halfWidth, -halfHeight, 0.0f); // Bottom-left
    rectangle->V.col (1) = Eigen::Vector3f (halfWidth, -halfHeight, 0.0f);  // Bottom-right
    rectangle->V.col (2) = Eigen::Vector3f (halfWidth, halfHeight, 0.0f);   // Top-right
    rectangle->V.col (3) = Eigen::Vector3f (-halfWidth, halfHeight, 0.0f);  // Top-left

    // Create a single surface for the rectangle
    rectangle->S.resize (1);
    auto& surface = rectangle->S[0];
    surface.name = "LuminousRectangle";

    // Set up indices for the two triangles
    surface.F.resize (3, 2);
    surface.F.col (0) = Vector3u (0, 1, 2); // First triangle: bottom-left, bottom-right, top-right
    surface.F.col (1) = Vector3u (0, 2, 3); // Second triangle: bottom-left, top-right, top-left

    // Set up the luminous material using CgMaterial
    auto& material = surface.cgMaterial;
    material.name = "LuminousMaterial";

    // Core properties - keep base color neutral since we want emission to dominate
    material.core.baseColor = Eigen::Vector3f (0.1f, 0.1f, 0.1f); // Very dark base
    material.core.roughness = 0.9f;                               // Rough surface
    material.core.specular = 0.0f;                                // No specular reflection

    // Metallic properties - non-metallic for light emission
    material.metallic.metallic = 0.0f;

    // Emission properties - this makes it luminous
    material.emission.luminous = luminousIntensity;
    material.emission.luminousColor = luminousColor;

    // Transparency properties - opaque light
    material.transparency.thin = false;
    material.transparency.transparency = 0.0f;

    // Material flags - unlit so it doesn't receive lighting
    material.flags.unlit = true;
    material.flags.alphaMode = AlphaMode::Opaque;

    surface.vertexCount = 4;

    // Update the total triangle count
    rectangle->triCount = 2;

    // Create vertex normals (all pointing in positive Z direction)
    rectangle->N.resize (3, 4);
    Eigen::Vector3f normal (0.0f, 0.0f, 1.0f);
    rectangle->N.col (0) = normal;
    rectangle->N.col (1) = normal;
    rectangle->N.col (2) = normal;
    rectangle->N.col (3) = normal;

    // Create UV coordinates for the rectangle
    rectangle->UV0.resize (2, 4);
    rectangle->UV0.col (0) = Eigen::Vector2f (0.0f, 0.0f); // Bottom-left
    rectangle->UV0.col (1) = Eigen::Vector2f (1.0f, 0.0f); // Bottom-right
    rectangle->UV0.col (2) = Eigen::Vector2f (1.0f, 1.0f); // Top-right
    rectangle->UV0.col (3) = Eigen::Vector2f (0.0f, 1.0f); // Top-left

    return rectangle;
}

// Creates a RenderableNode with a luminous rectangle mesh light
RenderableNode MeshOps::createLuminousRectangleNode (const std::string& name,
                                                     float width, float height,
                                                     const Eigen::Vector3f& luminousColor,
                                                     float luminousIntensity)
{
    // Create the luminous rectangle geometry
    CgModelPtr rectangleModel = createLuminousRectangle (width, height, luminousColor, luminousIntensity);

    // Create a new WorldItem (RenderableNode)
    RenderableNode node = WorldItem::create();

    // Set the name
    node->setName (name);

    // Attach the CgModel to the node
    node->setModel (rectangleModel);

    // Set up the renderable description
    RenderableDesc& desc = node->description();
    desc.bodyType = BodyType::Static;  // Mesh lights are typically static
    desc.shape = CollisionShape::None; // No collision for lights

    // Set up the renderable state
    RenderableState& state = node->getState();
    state.state |= PRenderableState::Visible;
    state.state |= PRenderableState::Pickable;

    // Initialize space-time properties
    SpaceTime& spacetime = node->getSpaceTime();
    spacetime.reset();

    // Calculate and set the model bounds
    spacetime.modelBound.min() = rectangleModel->V.rowwise().minCoeff();
    spacetime.modelBound.max() = rectangleModel->V.rowwise().maxCoeff();
    spacetime.updateWorldBounds (true);

    // Set the start transform to current world transform
    spacetime.startTransform = spacetime.worldTransform;

    return node;
}


// Creates a ground plane from 2 triangles lying flat on XZ plane
CgModelPtr MeshOps::createGroundPlane (float width, float depth,
                                       const Eigen::Vector3f& baseColor,
                                       float roughness)
{
    CgModelPtr groundPlane = CgModel::create();

    float halfWidth = width * 0.5f;
    float halfDepth = depth * 0.5f;

    // Define the vertices of the ground plane (lying flat on XZ plane, Y=0)
    groundPlane->V.resize (3, 4);
    groundPlane->V.col (0) = Eigen::Vector3f (-halfWidth, 0.0f, -halfDepth); // Back-left
    groundPlane->V.col (1) = Eigen::Vector3f (halfWidth, 0.0f, -halfDepth);  // Back-right
    groundPlane->V.col (2) = Eigen::Vector3f (halfWidth, 0.0f, halfDepth);   // Front-right
    groundPlane->V.col (3) = Eigen::Vector3f (-halfWidth, 0.0f, halfDepth);  // Front-left

    // Create a single surface for the ground plane
    groundPlane->S.resize (1);
    auto& surface = groundPlane->S[0];
    surface.name = "GroundPlaneSurface";

    // Set up indices for the two triangles (counter-clockwise winding when viewed from above)
    surface.F.resize (3, 2);
    surface.F.col (0) = Vector3u (0, 1, 2); // First triangle: back-left, back-right, front-right
    surface.F.col (1) = Vector3u (0, 2, 3); // Second triangle: back-left, front-right, front-left

    // Set up the material using CgMaterial
    auto& material = surface.cgMaterial;
    material.name = "GroundPlaneMaterial";

    // Core properties - light blue matte surface
    material.core.baseColor = baseColor;
    material.core.roughness = roughness; // High roughness for matte appearance
    material.core.specular = 0.1f;       // Very low specular for matte look
    material.core.specularTint = 0.0f;   // No specular tint

    // Metallic properties - non-metallic ground surface
    material.metallic.metallic = 0.0f;
    material.metallic.anisotropic = 0.0f;
    material.metallic.anisotropicRotation = 0.0f;

    // Emission properties - no emission (receives light)
    material.emission.luminous = 0.0f;
    material.emission.luminousColor = Eigen::Vector3f (0.0f, 0.0f, 0.0f);

    // Transparency properties - fully opaque
    material.transparency.thin = false;
    material.transparency.transparency = 0.0f;
    material.transparency.refractionIndex = 1.5f;

    // Subsurface properties - no subsurface scattering
    material.subsurface.subsurface = 0.0f;

    // Material flags - receives lighting and shadows
    material.flags.unlit = false; // Receives lighting
    material.flags.alphaMode = AlphaMode::Opaque;
    material.flags.alphaCutoff = 0.5f;

    // Enable double-sided rendering for ground plane
    material.doubleSided = true;

    surface.vertexCount = 4;

    // Update the total triangle count
    groundPlane->triCount = 2;

    // Create vertex normals (all pointing upward in +Y direction)
    groundPlane->N.resize (3, 4);
    Eigen::Vector3f upNormal (0.0f, 1.0f, 0.0f);
    groundPlane->N.col (0) = upNormal;
    groundPlane->N.col (1) = upNormal;
    groundPlane->N.col (2) = upNormal;
    groundPlane->N.col (3) = upNormal;

    // Create UV coordinates for the ground plane (useful for texturing)
    groundPlane->UV0.resize (2, 4);
    groundPlane->UV0.col (0) = Eigen::Vector2f (0.0f, 0.0f); // Back-left
    groundPlane->UV0.col (1) = Eigen::Vector2f (1.0f, 0.0f); // Back-right
    groundPlane->UV0.col (2) = Eigen::Vector2f (1.0f, 1.0f); // Front-right
    groundPlane->UV0.col (3) = Eigen::Vector2f (0.0f, 1.0f); // Front-left

    return groundPlane;
}

// Creates a RenderableNode with a ground plane
RenderableNode MeshOps::createGroundPlaneNode (const std::string& name,
                                               float width, float depth,
                                               const Eigen::Vector3f& baseColor,
                                               float roughness)
{
    // Create the ground plane geometry
    CgModelPtr planeModel = createGroundPlane (width, depth, baseColor, roughness);

    // Create a new WorldItem (RenderableNode)
    RenderableNode node = WorldItem::create();

    // Set the name
    node->setName (name);

    // Attach the CgModel to the node
    node->setModel (planeModel);

    // Set up the renderable description
    RenderableDesc& desc = node->description();
    desc.bodyType = BodyType::Static; // Ground planes are static
    desc.shape = CollisionShape::Mesh; // Use box collision for ground
    desc.mass = 0.0;                  // Static objects have zero mass
    desc.bounciness = 0.1;            // Low bounce for ground
    desc.staticFriction = 0.8;        // High static friction
    desc.dynamicFriction = 0.6;       // Medium dynamic friction

    // Set up the renderable state
    RenderableState& state = node->getState();
    state.state |= PRenderableState::Visible;
    state.state |= PRenderableState::Pickable;

    // Initialize space-time properties
    SpaceTime& spacetime = node->getSpaceTime();
    spacetime.reset();

    // Calculate and set the model bounds
    spacetime.modelBound.min() = planeModel->V.rowwise().minCoeff();
    spacetime.modelBound.max() = planeModel->V.rowwise().maxCoeff();
    spacetime.updateWorldBounds (true);

    // Set the start transform to current world transform
    spacetime.startTransform = spacetime.worldTransform;

    return node;
}