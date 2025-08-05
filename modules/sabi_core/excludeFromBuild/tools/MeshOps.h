#pragma once

#include "../../sabi_core.h"

using Eigen::AlignedBox3f;
using sabi::CgModelPtr;
using sabi::RenderableNode;
using sabi::LoadStrategyPtr;
using sabi::MeshOptions;
using sabi::RenderableNode;

// Define debug flags as an enum class for type safety
enum class DebugFlags
{
    None = 0,
    Vertices = 1 << 0,
    Indices = 1 << 1, 
    Normals = 1 << 2,
    UVs = 1 << 3,
    Surfaces = 1 << 4,
    Materials = 1 << 5,
    Textures = 1 << 6,
    Images = 1 << 7,
    Samplers = 1 << 8,
    All = Vertices | Indices | Normals | UVs | Surfaces | Materials | Textures | Images | Samplers
};

// Enable bitwise operations for DebugFlags
inline DebugFlags operator| (DebugFlags a, DebugFlags b)
{
    return static_cast<DebugFlags> (static_cast<uint32_t> (a) | static_cast<uint32_t> (b));
}

inline DebugFlags operator& (DebugFlags a, DebugFlags b)
{
    return static_cast<DebugFlags> (static_cast<uint32_t> (a) & static_cast<uint32_t> (b));
}

struct MeshOps
{
    static void unweldMesh(CgModelPtr& model);
    static void centerVertices (CgModelPtr model, const AlignedBox3f& modelBound, float scale);
    static void normalizeSize (CgModelPtr model, const AlignedBox3f& modelBound, float& scale);
    static void resizeModel (CgModelPtr model, const Eigen::Vector3f& targetSize);
    static void generate_normals (const MatrixXu& F, const MatrixXf& V, MatrixXf& N, MatrixXf& FN,
                                  bool deterministic, bool flatShaded = false);
    static void generate_normals (CgModelPtr& cgModel, bool flatShaded = false);
    static void prepareForFlatShading (CgModelPtr& model);
    static void processCgModel(RenderableNode& node, MeshOptions meshOptions, LoadStrategyPtr loadStrategy = nullptr);
    static CgModelPtr createTriangle();
    static CgModelPtr createTexturedTriangle (const fs::path& pngImagePath);
    static CgModelPtr createCube(float size = 1.0f);
    static CgModelPtr createTexturedQuad (const fs::path& pngImagePath);
    static void debugCgModel (const CgModelPtr& model, DebugFlags flags = DebugFlags::All);
    static bool dumpUVCoordinates (const CgModelPtr& model, const fs::path& outputPath);

      // Creates a luminous rectangle mesh light from 2 triangles
    static CgModelPtr createLuminousRectangle (float width = 2.0f, float height = 2.0f,
                                               const Eigen::Vector3f& luminousColor = Eigen::Vector3f (1.0f, 1.0f, 1.0f),
                                               float luminousIntensity = 1.0f);

    // Creates a RenderableNode with a luminous rectangle mesh light
    static RenderableNode createLuminousRectangleNode (const std::string& name = "MeshLight",
                                                       float width = 2.0f, float height = 2.0f,
                                                       const Eigen::Vector3f& luminousColor = Eigen::Vector3f (1.0f, 1.0f, 1.0f),
                                                       float luminousIntensity = 1.0f);

     // Creates a ground plane from 2 triangles lying flat on XZ plane
    static CgModelPtr createGroundPlane (float width = 20.0f, float depth = 20.0f,
                                         const Eigen::Vector3f& baseColor = Eigen::Vector3f (0.7f, 0.9f, 1.0f),
                                         float roughness = 0.9f);

    // Creates a RenderableNode with a ground plane
    static RenderableNode createGroundPlaneNode (const std::string& name = "GroundPlane",
                                                 float width = 20.0f, float depth = 20.0f,
                                                 const Eigen::Vector3f& baseColor = Eigen::Vector3f (0.7f, 0.9f, 1.0f),
                                                 float roughness = 0.9f);
};