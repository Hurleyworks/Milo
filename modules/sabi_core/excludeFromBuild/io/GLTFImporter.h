#pragma once

// A comprehensive importer for glTF (GL Transmission Format) files that converts
// them into the CgModel format. This class leverages the fastgltf library for
// parsing and provides complete support for both .gltf and .glb file formats.
//
// Features:
// - Imports geometry (vertices, indices, normals, UVs)
// - Imports PBR materials and textures
// - Handles node hierarchies and transforms
// - Supports animations
// - Can combine multiple meshes into a single model
//
// Example usage:
//   GLTFImporter importer;
//   auto [model, animations] = importer.importModel("path/to/model.gltf");

using sabi::CgMaterial;
using sabi::CgModel;
using sabi::CgModelList;
using sabi::CgModelPtr;

class GLTFImporter
{
 public:
    GLTFImporter() = default;
    ~GLTFImporter() = default;

    // Primary entry point for importing a glTF file
    // Returns a pair containing the model and its animations
    // Throws std::runtime_error if import fails
    std::pair<CgModelPtr, std::vector<Animation>> importModel (const std::string& filePath);

 private:
    // Internal state
    CgModelList models;                        // Imported models
    GltfAnimationExporter animationExporter;   // Animation export handler
    std::unordered_set<std::string> usedNames; // Name uniqueness tracking

    // Asset loading
    // Parses glTF/GLB file into fastgltf::Asset
    fastgltf::Asset loadGLTF (const std::string& filePath);

    // Scene handling
    void processScenes (const fastgltf::Asset& asset);
    void processNode (const fastgltf::Asset& asset, const fastgltf::Node& node,
                      const Eigen::Affine3f& parentTransform);

    // Geometry import
    void importPrimitive (const fastgltf::Asset& asset, const fastgltf::Mesh& mesh,
                          const fastgltf::Primitive& primitive,
                          const Eigen::Affine3f& transform, CgModel& model);
    void importVertices (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                         CgModel& model);
    void importIndices (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                        CgModel& model);
    void importNormals (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                        CgModel& model);
    void importUVs (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                    CgModel& model);

    // Material import
    void importMaterial (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                         CgModel& model);
    void importCoreMaterialProperties (const fastgltf::Asset& asset,
                                       const fastgltf::Material& gltfMaterial,
                                       sabi::CgMaterial::CoreProperties& core);
    void importMetallicProperties (const fastgltf::Asset& asset,
                                   const fastgltf::Material& gltfMaterial,
                                   sabi::CgMaterial::MetallicProperties& metallic);
    void importSheenProperties (const fastgltf::Asset& asset,
                                const fastgltf::Material& gltfMaterial,
                                sabi::CgMaterial::SheenProperties& sheen);
    void importTranslucencyProperties (const fastgltf::Asset& asset,
                                       const fastgltf::Material& gltfMaterial,
                                       sabi::CgMaterial::TranslucencyProperties& translucency);
    void importSubsurfaceProperties (const fastgltf::Asset& asset,
                                     const fastgltf::Material& gltfMaterial,
                                     sabi::CgMaterial::SubsurfaceProperties& subsurface);
    void importEmissionProperties (const fastgltf::Asset& asset,
                                   const fastgltf::Material& gltfMaterial,
                                   sabi::CgMaterial::EmissionProperties& emission);
    void importClearcoatProperties (const fastgltf::Asset& asset,
                                    const fastgltf::Material& gltfMaterial,
                                    sabi::CgMaterial::ClearcoatProperties& clearcoat);
    void importTransparencyProperties (const fastgltf::Asset& asset,
                                       const fastgltf::Material& gltfMaterial,
                                       sabi::CgMaterial::TransparencyProperties& transparency);
    void importPackedTextures (const fastgltf::Asset& asset,
                               const fastgltf::Material& gltfMaterial,
                               sabi::CgMaterial::PackedTextureProperties& packedTextures);

    // Texture import
    void importImages (const fastgltf::Asset& asset, CgModel& model);
    void importCgImages (const fastgltf::Asset& asset, sabi::CgModel& model);
    void importTextures (const fastgltf::Asset& asset, CgModel& model);
    void importCgTextures (const fastgltf::Asset& asset, sabi::CgModel& model);
    void importSamplers (const fastgltf::Asset& asset, CgModel& model);
    void importCgSamplers (const fastgltf::Asset& asset, CgModel& model);
    sabi::TextureInfo importTextureInfo (const fastgltf::Asset& asset,
                                         const fastgltf::TextureInfo& textureInfo);
    sabi::CgTextureInfo importCgTextureInfo (const fastgltf::Asset& asset,
                                             const fastgltf::TextureInfo& textureInfo);
    // Image data handling
    void extractImageData (const fastgltf::Asset& asset, sabi::Image& sabiImage,
                           const fastgltf::Image& image, size_t imageIndex,
                           std::vector<unsigned char>& imageBytes, std::uint32_t& width,
                           std::uint32_t& height, std::uint32_t& channels,
                           std::string& imageIdentifier);
    void extractImageData (const fastgltf::Asset& asset, sabi::CgImage& sabiImage,
                           const fastgltf::Image& image, size_t imageIndex,
                           std::vector<unsigned char>& imageBytes, std::uint32_t& width,
                           std::uint32_t& height, std::uint32_t& channels,
                           std::string& imageIdentifier);

    // Helper functions
    std::string mimeTypeToString (fastgltf::MimeType mimeType);
    std::string getImageIdentifier (const fastgltf::Asset& asset, size_t imageIndex);
    std::string generateUniqueName (const std::string& baseName);
    Eigen::Affine3f computeLocalTransform (const fastgltf::Node& node);
    std::vector<Animation> importAnimations (const fastgltf::Asset& asset);
    CgModelPtr forgeIntoOne (const CgModelList& models);

};