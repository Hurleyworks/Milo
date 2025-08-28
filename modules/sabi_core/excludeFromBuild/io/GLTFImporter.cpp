
#include <stb_image/stb_image.h>

using fastgltf::Asset;
using fastgltf::Expected;
using fastgltf::GltfDataBuffer;
using fastgltf::Options;
using fastgltf::Parser;

using sabi::CgModelSurface;
using sabi::Material;

using Eigen::Vector3f;


std::pair<CgModelPtr, std::vector<Animation>> GLTFImporter::importModel (const std::string& filePath)
{
    try
    {
        if (!fs::exists (filePath) || !fs::is_regular_file (filePath))
        {
            throw std::runtime_error ("File does not exist!");
        }

        auto asset = loadGLTF (filePath);

        processScenes (asset);
        if (models.size() == 0) return {nullptr, {}};

        CgModelPtr cgModel = models.size() > 1 ? forgeIntoOne (models) : models[0];
        if (!cgModel) return {nullptr, {}};

        importImages (asset, *cgModel);
        importTextures (asset, *cgModel);
        importSamplers (asset, *cgModel);

        // CgMaterial support
        importCgImages (asset, *cgModel);
        importCgTextures (asset, *cgModel);
        importCgSamplers (asset, *cgModel);

        std::vector<Animation> animations = importAnimations (asset);

        return {cgModel, animations};
    }
    catch (const std::exception& e)
    {
        LOG (DBUG) << "Failed to load glTF file: " << filePath << ". Error: " << e.what();
        throw;
        //  return nullptr;
    }
}


// Asset loading and parsing implementation
fastgltf::Asset GLTFImporter::loadGLTF (const std::string& filePath)
{
    Expected<fastgltf::GltfDataBuffer> data = GltfDataBuffer::FromPath (filePath);
    Expected<Asset> asset (fastgltf::Error::None);

    // Set up parsing options and supported extensions
    auto options = fastgltf::Options::LoadExternalBuffers;
    auto extensions = fastgltf::Extensions::MSFT_packing_normalRoughnessMetallic |
                      fastgltf::Extensions::MSFT_packing_occlusionRoughnessMetallic;

    Parser parser (extensions);

    // Handle both .gltf and .glb formats
    switch (fastgltf::determineGltfFileType (data.get()))
    {
        case fastgltf::GltfType::glTF:
            asset = parser.loadGltf (data.get(), fs::path (filePath).parent_path(), options);
            break;

        case fastgltf::GltfType::GLB:
            asset = parser.loadGltfBinary (data.get(), fs::path (filePath).parent_path(), options);
            break;

        default:
            throw std::invalid_argument ("Error: Failed to determine glTF container type");
    }

    // Handle potential parsing errors
    if (asset.error() != fastgltf::Error::None)
    {
        std::string errorMsg = "Error loading glTF asset: ";
        switch (asset.error())
        {
            case fastgltf::Error::InvalidPath:
                errorMsg += "Invalid glTF directory path";
                break;
            case fastgltf::Error::MissingExtensions:
                errorMsg += "Required extensions not enabled in Parser";
                break;
            case fastgltf::Error::UnknownRequiredExtension:
                errorMsg += "Required extension not supported";
                break;
            case fastgltf::Error::InvalidJson:
                errorMsg += "JSON parsing error";
                break;
            case fastgltf::Error::InvalidGltf:
                errorMsg += "Missing or invalid data";
                break;
            case fastgltf::Error::InvalidGLB:
                errorMsg += "Invalid GLB container";
                break;
            case fastgltf::Error::MissingField:
                errorMsg += "Required field missing";
                break;
            case fastgltf::Error::InvalidFileData:
                errorMsg += "Invalid file data";
                break;
            default:
                errorMsg += "Unknown error occurred";
        }
        LOG (CRITICAL) << errorMsg;
        throw std::runtime_error (errorMsg);
    }

    return std::move (asset.get());
}

// Scene graph processing implementation
void GLTFImporter::processScenes (const fastgltf::Asset& asset)
{
    if (asset.scenes.empty())
    {
        LOG (CRITICAL) << "No scenes found in the glTF file";
        return;
    }

    // Use default scene or first scene if no default specified
    size_t sceneIndex = asset.defaultScene.value_or (0);
    const auto& scene = asset.scenes[sceneIndex];

    // Process each root node in the scene
    for (const auto& nodeIndex : scene.nodeIndices)
    {
        const auto& node = asset.nodes[nodeIndex];
        processNode (asset, node, Eigen::Affine3f::Identity());
    }
}

void GLTFImporter::processNode (const fastgltf::Asset& asset, const fastgltf::Node& node,
                                const Eigen::Affine3f& parentTransform)
{
    // Compute node's transform
    Eigen::Affine3f localTransform = computeLocalTransform (node);
    Eigen::Affine3f globalTransform = parentTransform * localTransform;

    // Process mesh if present
    if (node.meshIndex.has_value())
    {
        const auto& mesh = asset.meshes[node.meshIndex.value()];

        // Process each primitive in the mesh
        for (const auto& primitive : mesh.primitives)
        {
            CgModelPtr model = CgModel::create();
            importPrimitive (asset, mesh, primitive, globalTransform, *model);
            models.push_back (model);
        }
    }

    // Recursively process child nodes
    for (const auto& childIndex : node.children)
    {
        const auto& childNode = asset.nodes[childIndex];
        processNode (asset, childNode, globalTransform);
    }
}

// Geometry import implementations
void GLTFImporter::importPrimitive (const fastgltf::Asset& asset, const fastgltf::Mesh& mesh,
                                    const fastgltf::Primitive& primitive,
                                    const Eigen::Affine3f& transform, CgModel& model)
{
    // Import geometry data
    importVertices (asset, primitive, model);
    importIndices (asset, primitive, model);
    importNormals (asset, primitive, model);
    importUVs (asset, primitive, model);
    importMaterial (asset, primitive, model);

    // Apply transformation to vertices
    for (int i = 0; i < model.V.cols(); ++i)
    {
        Vector3f p = model.V.col (i);
        model.V.col (i) = transform * p;

        // Convert from right-handed to left-handed coordinate system
        Vector3f v = model.V.col (i);
        model.V.col (i) = Vector3f (v.x(), v.y(), v.z() * -1.0f);
    }

    // Transform normals if they exist
    if (model.N.cols() > 0)
    {
        Eigen::Matrix3f normalTransform = transform.linear().inverse().transpose();
        for (int i = 0; i < model.N.cols(); ++i)
        {
            model.N.col (i) = (normalTransform * model.N.col (i)).normalized();
        }
    }
}

void GLTFImporter::importVertices (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                                   CgModel& model)
{
    auto it = primitive.findAttribute ("POSITION");
    if (it == primitive.attributes.end())
    {
        return;
    }

    const auto& accessor = asset.accessors[it->accessorIndex];
    if (!accessor.bufferViewIndex.has_value())
    {
        LOG (CRITICAL) << "Position accessor missing buffer view";
        return;
    }

    // Allocate vertex buffer
    std::size_t vertexCount = accessor.count;
    model.V.resize (3, vertexCount);
    std::size_t i = 0;

    // Read vertex data
    fastgltf::iterateAccessor<fastgltf::math::fvec3> (
        asset, accessor,
        [&] (auto&& v3)
        {
            model.V.col (i++) = Eigen::Vector3f (v3[0], v3[1], v3[2]);
        });
}

void GLTFImporter::importIndices (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                                  CgModel& model)
{
    CgModelSurface surface;

    if (primitive.indicesAccessor.has_value())
    {
        const auto& accessor = asset.accessors[*primitive.indicesAccessor];
        std::size_t indexCount = accessor.count;
        surface.F.resize (3, indexCount / 3);

        // Read index data
        std::vector<uint32_t> indices (indexCount);
        fastgltf::iterateAccessorWithIndex<std::uint32_t> (
            asset, accessor,
            [&] (std::uint32_t index, size_t idx)
            {
                indices[idx] = index;
            });

        // Copy indices with winding order adjustment for LH coordinate system
        for (size_t i = 0; i < indexCount; i += 3)
        {
            surface.F.col (i / 3) << indices[i], indices[i + 2], indices[i + 1];
        }
    }

    model.S.push_back (std::move (surface));
}

void GLTFImporter::importNormals (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                                  CgModel& model)
{
    auto it = primitive.findAttribute ("NORMAL");
    if (it == primitive.attributes.end())
    {
        return;
    }

    const auto& accessor = asset.accessors[it->accessorIndex];
    if (!accessor.bufferViewIndex.has_value())
    {
        LOG (CRITICAL) << "Normal accessor missing buffer view";
        return;
    }

    // Allocate normal buffer
    std::size_t normalCount = accessor.count;
    model.N.resize (3, normalCount);
    std::size_t i = 0;

    // Read normal data
    fastgltf::iterateAccessor<fastgltf::math::fvec3> (
        asset, accessor,
        [&] (auto&& v3)
        {
            model.N.col (i++) = Eigen::Vector3f (v3[0], v3[1], v3[2]);
        });
}

void GLTFImporter::importUVs (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                              CgModel& model)
{
    auto it = primitive.findAttribute ("TEXCOORD_0");
    if (it == primitive.attributes.end())
    {
        return;
    }

    const auto& accessor = asset.accessors[it->accessorIndex];
    if (!accessor.bufferViewIndex.has_value())
    {
        LOG (CRITICAL) << "UV accessor missing buffer view";
        return;
    }

    // Allocate UV buffer
    std::size_t uvCount = accessor.count;
    model.UV0.resize (2, uvCount);
    std::size_t i = 0;

    // Read UV data
    fastgltf::iterateAccessor<fastgltf::math::fvec2> (
        asset, accessor,
        [&] (auto&& v2)
        {
            model.UV0.col (i++) = Eigen::Vector2f (v2[0], v2[1]);
        });
}
void GLTFImporter::importMaterial (const fastgltf::Asset& asset,
                                   const fastgltf::Primitive& primitive,
                                   sabi::CgModel& model)
{
    if (!primitive.materialIndex.has_value())
    {
        LOG (DBUG) << "Primitive does not have a material";
        if (!model.S.empty())
        {
            model.S.back().name = generateUniqueName ("Default_Surface");
        }
        return;
    }

    const auto& gltfMaterial = asset.materials[primitive.materialIndex.value()];
    sabi::Material material;

    // Set basic material properties
    std::string materialName = std::string (gltfMaterial.name);
    material.name = generateUniqueName (materialName.empty() ? "Material" : materialName);

    // Import CgMaterial properties
    auto& cgMaterial = model.S.back().cgMaterial;
    cgMaterial.name = material.name;
    cgMaterial.doubleSided = gltfMaterial.doubleSided;
    cgMaterial.flags.unlit = gltfMaterial.unlit;
    cgMaterial.flags.alphaMode = static_cast<sabi::AlphaMode> (gltfMaterial.alphaMode);
    cgMaterial.flags.alphaCutoff = gltfMaterial.alphaCutoff;

    // Import all material properties
    importCoreMaterialProperties (asset, gltfMaterial, cgMaterial.core);
    importMetallicProperties (asset, gltfMaterial, cgMaterial.metallic);
    importSheenProperties (asset, gltfMaterial, cgMaterial.sheen);
    importTranslucencyProperties (asset, gltfMaterial, cgMaterial.translucency);
    importSubsurfaceProperties (asset, gltfMaterial, cgMaterial.subsurface);
    importEmissionProperties (asset, gltfMaterial, cgMaterial.emission);
    importClearcoatProperties (asset, gltfMaterial, cgMaterial.clearcoat);
    importTransparencyProperties (asset, gltfMaterial, cgMaterial.transparency);
    importPackedTextures (asset, gltfMaterial, cgMaterial.packedTextures);

    // Handle normal texture
    if (gltfMaterial.normalTexture)
    {
        sabi::CgTextureInfo normalTextureInfo = importCgTextureInfo (asset, gltfMaterial.normalTexture.value());
        normalTextureInfo.scale = gltfMaterial.normalTexture->scale;
        cgMaterial.normalTexture = std::move (normalTextureInfo);
    }

    // Handle occlusion texture
    if (gltfMaterial.occlusionTexture)
    {
        sabi::CgTextureInfo occlusionTextureInfo = importCgTextureInfo (asset, gltfMaterial.occlusionTexture.value());
        occlusionTextureInfo.strength = gltfMaterial.occlusionTexture->strength;
        cgMaterial.occlusionTexture = std::move (occlusionTextureInfo);
    }

    // Apply material to surface
    if (!model.S.empty())
    {
        model.S.back().material = material;
        model.S.back().name = generateUniqueName (material.name + "_Surface");
    }
    else
    {
        LOG (CRITICAL) << "No surface available to assign material";
    }
}
#if 0
void GLTFImporter::importMaterial (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive,
                                   sabi::CgModel& model)
{
    if (!primitive.materialIndex.has_value())
    {
        LOG (DBUG) << "Primitive does not have a material";
        if (!model.S.empty())
        {
            model.S.back().name = generateUniqueName ("Default_Surface");
        }
        return;
    }

    const auto& gltfMaterial = asset.materials[primitive.materialIndex.value()];
    sabi::Material material;

    // Set basic material properties
    std::string materialName = std::string (gltfMaterial.name);
    material.name = generateUniqueName (materialName.empty() ? "Material" : materialName);

    // Import material properties
    auto& cgMaterial = model.S.back().cgMaterial;
    cgMaterial.name = material.name;
    cgMaterial.doubleSided = gltfMaterial.doubleSided;
    cgMaterial.flags.unlit = gltfMaterial.unlit;
    cgMaterial.flags.alphaMode = static_cast<sabi::AlphaMode> (gltfMaterial.alphaMode);
    cgMaterial.flags.alphaCutoff = gltfMaterial.alphaCutoff;

    // Import all material properties
    importCoreMaterialProperties (asset, gltfMaterial, cgMaterial.core);
    importMetallicProperties (asset, gltfMaterial, cgMaterial.metallic);
    importSheenProperties (asset, gltfMaterial, cgMaterial.sheen);
    importTranslucencyProperties (asset, gltfMaterial, cgMaterial.translucency);
    importSubsurfaceProperties (asset, gltfMaterial, cgMaterial.subsurface);
    importEmissionProperties (asset, gltfMaterial, cgMaterial.emission);
    importClearcoatProperties (asset, gltfMaterial, cgMaterial.clearcoat);
    importTransparencyProperties (asset, gltfMaterial, cgMaterial.transparency);
    importPackedTextures (asset, gltfMaterial, cgMaterial.packedTextures);

    // Apply material to surface
    if (!model.S.empty())
    {
        model.S.back().material = material;
        model.S.back().name = generateUniqueName (material.name + "_Surface");
    }
    else
    {
        LOG (CRITICAL) << "No surface available to assign material";
    }
}

#endif
void GLTFImporter::importCoreMaterialProperties (const fastgltf::Asset& asset,
                                                 const fastgltf::Material& gltfMaterial,
                                                 sabi::CgMaterial::CoreProperties& core)
{
    // Base color
    core.baseColor = {
        gltfMaterial.pbrData.baseColorFactor[0],
        gltfMaterial.pbrData.baseColorFactor[1],
        gltfMaterial.pbrData.baseColorFactor[2]};

    core.roughness = gltfMaterial.pbrData.roughnessFactor;
    core.specular = 0.5f;

    // Import base color texture
    if (gltfMaterial.pbrData.baseColorTexture)
    {
        core.baseColorTexture = importCgTextureInfo (asset, gltfMaterial.pbrData.baseColorTexture.value());
    }

    // Import roughness texture
    if (gltfMaterial.pbrData.metallicRoughnessTexture)
    {
        core.roughnessTexture = importCgTextureInfo (asset, gltfMaterial.pbrData.metallicRoughnessTexture.value());
    }

    
}

#if 0
void GLTFImporter::importCoreMaterialProperties (const fastgltf::Asset& asset,
                                                 const fastgltf::Material& gltfMaterial,
                                                 sabi::CgMaterial::CoreProperties& core)
{
    // Set base color
    core.baseColor = {
        gltfMaterial.pbrData.baseColorFactor[0],
        gltfMaterial.pbrData.baseColorFactor[1],
        gltfMaterial.pbrData.baseColorFactor[2]};

    // Set roughness and specular properties
    core.roughness = gltfMaterial.pbrData.roughnessFactor;
    core.specular = 0.5f; // Default value as glTF PBR doesn't have direct specular

    // Import textures
    if (gltfMaterial.pbrData.baseColorTexture)
    {
        core.baseColorTexture = importCgTextureInfo (asset, gltfMaterial.pbrData.baseColorTexture.value());
    }

    if (gltfMaterial.pbrData.metallicRoughnessTexture)
    {
        core.roughnessTexture = importCgTextureInfo (asset, gltfMaterial.pbrData.metallicRoughnessTexture.value());
    }
}

#endif

void GLTFImporter::importMetallicProperties (const fastgltf::Asset& asset,
                                             const fastgltf::Material& gltfMaterial,
                                             sabi::CgMaterial::MetallicProperties& metallic)
{
    metallic.metallic = gltfMaterial.pbrData.metallicFactor;

    // Handle anisotropic properties if present
    if (gltfMaterial.anisotropy)
    {
        metallic.anisotropic = gltfMaterial.anisotropy->anisotropyStrength;
        metallic.anisotropicRotation = gltfMaterial.anisotropy->anisotropyRotation;

        if (gltfMaterial.anisotropy->anisotropyTexture)
        {
            metallic.anisotropicTexture = importCgTextureInfo (asset,
                                                               gltfMaterial.anisotropy->anisotropyTexture.value());
        }
    }

    // Import metallic texture if present
    if (gltfMaterial.pbrData.metallicRoughnessTexture)
    {
        metallic.metallicTexture = importCgTextureInfo (asset,
                                                        gltfMaterial.pbrData.metallicRoughnessTexture.value());
    }
}

void GLTFImporter::importSheenProperties (const fastgltf::Asset& asset,
                                          const fastgltf::Material& gltfMaterial,
                                          sabi::CgMaterial::SheenProperties& sheen)
{
    if (gltfMaterial.sheen)
    {
        sheen.sheenColorFactor = Eigen::Vector3f (
            gltfMaterial.sheen->sheenColorFactor.x(),
            gltfMaterial.sheen->sheenColorFactor.y(),
            gltfMaterial.sheen->sheenColorFactor.z());
        sheen.sheenRoughnessFactor = gltfMaterial.sheen->sheenRoughnessFactor;

        if (gltfMaterial.sheen->sheenColorTexture)
        {
            sheen.sheenColorTexture = importCgTextureInfo (asset,
                                                           gltfMaterial.sheen->sheenColorTexture.value());
        }
        if (gltfMaterial.sheen->sheenRoughnessTexture)
        {
            sheen.sheenRoughnessTexture = importCgTextureInfo (asset,
                                                               gltfMaterial.sheen->sheenRoughnessTexture.value());
        }
    }
}

void GLTFImporter::importTranslucencyProperties (const fastgltf::Asset& asset,
                                                 const fastgltf::Material& gltfMaterial,
                                                 sabi::CgMaterial::TranslucencyProperties& translucency)
{
    // glTF doesn't have direct translucency properties
    // Set reasonable defaults or map from other properties if needed
    translucency.translucency = 0.0f;
    //translucency.scatterDistance = 0.0f;
}

void GLTFImporter::importSubsurfaceProperties (const fastgltf::Asset& asset,
                                               const fastgltf::Material& gltfMaterial,
                                               sabi::CgMaterial::SubsurfaceProperties& subsurface)
{
    if (gltfMaterial.volume)
    {
        subsurface.subsurface = gltfMaterial.volume->thicknessFactor;
        subsurface.subsurfaceColor = {
            gltfMaterial.volume->attenuationColor[0],
            gltfMaterial.volume->attenuationColor[1],
            gltfMaterial.volume->attenuationColor[2]};
        subsurface.subsurfaceDistance = gltfMaterial.volume->attenuationDistance;

        if (gltfMaterial.volume->thicknessTexture)
        {
            subsurface.subsurfaceColorTexture = importCgTextureInfo (asset,
                                                                     gltfMaterial.volume->thicknessTexture.value());
        }
    }

    subsurface.asymmetry = 0.0f; // Not directly supported in glTF
}

void GLTFImporter::importEmissionProperties (const fastgltf::Asset& asset,
                                             const fastgltf::Material& gltfMaterial,
                                             sabi::CgMaterial::EmissionProperties& emission)
{
    emission.luminous = gltfMaterial.emissiveStrength;
    emission.luminousColor = {
        gltfMaterial.emissiveFactor[0],
        gltfMaterial.emissiveFactor[1],
        gltfMaterial.emissiveFactor[2]};

    if (gltfMaterial.emissiveTexture)
    {
        emission.luminousTexture = importCgTextureInfo (asset,
                                                        gltfMaterial.emissiveTexture.value());
    }
}

void GLTFImporter::importClearcoatProperties (const fastgltf::Asset& asset,
                                              const fastgltf::Material& gltfMaterial,
                                              sabi::CgMaterial::ClearcoatProperties& clearcoat)
{
    if (gltfMaterial.clearcoat)
    {
        clearcoat.clearcoat = gltfMaterial.clearcoat->clearcoatFactor;
        clearcoat.clearcoatGloss = gltfMaterial.clearcoat->clearcoatRoughnessFactor;

        if (gltfMaterial.clearcoat->clearcoatTexture)
        {
            clearcoat.clearcoatTexture = importCgTextureInfo (asset,
                                                              gltfMaterial.clearcoat->clearcoatTexture.value());
        }

        if (gltfMaterial.clearcoat->clearcoatRoughnessTexture)
        {
            clearcoat.clearcoatRoughnessTexture = importCgTextureInfo (asset,
                                                                       gltfMaterial.clearcoat->clearcoatRoughnessTexture.value());
        }

        if (gltfMaterial.clearcoat->clearcoatNormalTexture)
        {
            clearcoat.clearcoatNormalTexture = importCgTextureInfo (asset,
                                                                    gltfMaterial.clearcoat->clearcoatNormalTexture.value());
        }
    }
}

void GLTFImporter::importTransparencyProperties (const fastgltf::Asset& asset,
                                                 const fastgltf::Material& gltfMaterial,
                                                 sabi::CgMaterial::TransparencyProperties& transparency)
{
    if (gltfMaterial.transmission)
    {
        transparency.transparency = gltfMaterial.transmission->transmissionFactor;

        if (gltfMaterial.transmission->transmissionTexture)
        {
            transparency.transparencyTexture = importCgTextureInfo (asset,
                                                                    gltfMaterial.transmission->transmissionTexture.value());
        }
    }

    transparency.refractionIndex = gltfMaterial.ior > 1.0f ? gltfMaterial.ior : 1.45f;
    transparency.thin = false;
    transparency.transmittance = {0.5f, 0.5f, 0.5f};
    transparency.transmittanceDistance = 1.0f;
}

void GLTFImporter::importPackedTextures (const fastgltf::Asset& asset,
                                         const fastgltf::Material& gltfMaterial,
                                         sabi::CgMaterial::PackedTextureProperties& packedTextures)
{
    if (gltfMaterial.packedOcclusionRoughnessMetallicTextures)
    {
        const auto& packedTexturesData = *gltfMaterial.packedOcclusionRoughnessMetallicTextures;

        // Handle ORM combined texture (Occlusion-Roughness-Metallic)
        if (packedTexturesData.occlusionRoughnessMetallicTexture)
        {
            packedTextures.occlusionRoughnessMetallicTexture = importCgTextureInfo (asset,
                                                                                    packedTexturesData.occlusionRoughnessMetallicTexture.value());
        }

        // Handle RMO combined texture (Roughness-Metallic-Occlusion)
        if (packedTexturesData.roughnessMetallicOcclusionTexture)
        {
            packedTextures.roughnessMetallicOcclusionTexture = importCgTextureInfo (asset,
                                                                                    packedTexturesData.roughnessMetallicOcclusionTexture.value());
        }

        // Handle normal texture from packed textures
        if (packedTexturesData.normalTexture)
        {
            packedTextures.normalRoughnessMetallicTexture = importCgTextureInfo (asset,
                                                                                 packedTexturesData.normalTexture.value());
        }
    }

    if (gltfMaterial.packedNormalMetallicRoughnessTexture)
    {
        packedTextures.normalRoughnessMetallicTexture = importCgTextureInfo (asset,
                                                                             gltfMaterial.packedNormalMetallicRoughnessTexture.value());
    }
}

void GLTFImporter::importImages (const fastgltf::Asset& asset, CgModel& model)
{
    model.images.reserve (asset.images.size());

    size_t index = 0;
    for (const auto& gltfImage : asset.images)
    {
        sabi::Image sabiImage;
        sabiImage.name = gltfImage.name;
        sabiImage.index = index;

        // Temporary storage for extracted image data
        std::vector<unsigned char> extractedImageBytes;
        std::uint32_t width, height, channels;
        std::string imageId;

        // Extract image data from GLTF source
        extractImageData (asset, sabiImage, gltfImage, index++,
                          extractedImageBytes, width, height, channels, imageId);

        // If we successfully extracted image data, store it in an ImageBuf
        if (extractedImageBytes.size())
        {
            OIIO::ImageSpec spec (width, height, channels, OIIO::TypeDesc::UINT8);
            OIIO::ImageBuf imageBuf (spec, extractedImageBytes.data());
            sabiImage.extractedImage = imageBuf;
        }

        model.images.push_back (std::move (sabiImage));
    }
}

void GLTFImporter::importCgImages (const fastgltf::Asset& asset, sabi::CgModel& model)
{
    model.cgImages.reserve (asset.images.size());

    size_t index = 0;
    for (const auto& gltfImage : asset.images)
    {
        sabi::CgImage cgImage;
        cgImage.name = gltfImage.name;
        cgImage.index = index;

        // Temporary storage for extracted image data
        std::vector<unsigned char> extractedImageBytes;
        std::uint32_t width, height, channels;
        std::string imageId;

        // Extract image data from GLTF source
        extractImageData (asset, cgImage, gltfImage, index++,
                          extractedImageBytes, width, height, channels, imageId);

        // If we successfully extracted image data, store it in an ImageBuf
        if (extractedImageBytes.size())
        {
            OIIO::ImageSpec spec (width, height, channels, OIIO::TypeDesc::UINT8);
            OIIO::ImageBuf imageBuf (spec, extractedImageBytes.data());
            cgImage.extractedImage = imageBuf;
        }

        model.cgImages.push_back (std::move (cgImage));
    }
}

void GLTFImporter::importTextures (const fastgltf::Asset& asset, CgModel& model)
{
    model.textures.reserve (asset.textures.size());

    for (const auto& gltfTexture : asset.textures)
    {
        sabi::Texture texture;

        // Copy basic properties
        texture.name = gltfTexture.name;
        texture.source = gltfTexture.imageIndex.value_or (-1);
        texture.sampler = gltfTexture.samplerIndex.value_or (-1);

        model.textures.push_back (std::move (texture));
    }

    LOG (DBUG) << "Imported " << model.textures.size() << " textures";
}

void GLTFImporter::importCgTextures (const fastgltf::Asset& asset, sabi::CgModel& model)
{
    model.cgTextures.reserve (asset.textures.size());

    for (const auto& gltfTexture : asset.textures)
    {
        sabi::CgTexture texture;

        // Copy basic properties
        texture.name = gltfTexture.name;
        texture.imageIndex = gltfTexture.imageIndex;
        texture.samplerIndex = gltfTexture.samplerIndex;

        // Copy extended format indices
        texture.basisuImageIndex = gltfTexture.basisuImageIndex;
        texture.ddsImageIndex = gltfTexture.ddsImageIndex;
        texture.webpImageIndex = gltfTexture.webpImageIndex;

        model.cgTextures.push_back (std::move (texture));
    }

    LOG (DBUG) << "Imported " << model.cgTextures.size() << " CG textures";
}

void GLTFImporter::importSamplers (const fastgltf::Asset& asset, CgModel& model)
{
    model.samplers.reserve (asset.samplers.size());

    for (const auto& gltfSampler : asset.samplers)
    {
        sabi::Sampler sampler;

        // Copy basic properties
        sampler.name = gltfSampler.name;

        // Handle optional magFilter and minFilter
        sampler.magFilter = gltfSampler.magFilter.has_value() ? static_cast<int> (gltfSampler.magFilter.value()) : INVALID_INDEX;

        sampler.minFilter = gltfSampler.minFilter.has_value() ? static_cast<int> (gltfSampler.minFilter.value()) : INVALID_INDEX;

        // Handle wrap modes
        sampler.wrapS = static_cast<int> (gltfSampler.wrapS);
        sampler.wrapT = static_cast<int> (gltfSampler.wrapT);

        model.samplers.push_back (std::move (sampler));
    }

    LOG (DBUG) << "Imported " << model.samplers.size() << " samplers";
}

void GLTFImporter::importCgSamplers (const fastgltf::Asset& asset, CgModel& model)
{
    model.cgSamplers.reserve (asset.samplers.size());

    for (const auto& gltfSampler : asset.samplers)
    {
        sabi::CgSampler sampler;

        // Copy basic properties
        sampler.name = gltfSampler.name;

        // Handle optional magFilter and minFilter with CgFilter types
        sampler.magFilter = gltfSampler.magFilter.has_value() ? static_cast<sabi::CgFilter> (gltfSampler.magFilter.value()) : sabi::CgFilter::Linear;

        sampler.minFilter = gltfSampler.minFilter.has_value() ? static_cast<sabi::CgFilter> (gltfSampler.minFilter.value()) : sabi::CgFilter::Linear;

        // Handle wrap modes with CgWrap types
        sampler.wrapS = static_cast<sabi::CgWrap> (gltfSampler.wrapS);
        sampler.wrapT = static_cast<sabi::CgWrap> (gltfSampler.wrapT);

        model.cgSamplers.push_back (std::move (sampler));
    }

    LOG (DBUG) << "Imported " << model.cgSamplers.size() << " CG samplers";
}

void GLTFImporter::extractImageData (const fastgltf::Asset& asset, sabi::Image& sabiImage,
                                     const fastgltf::Image& image, size_t imageIndex,
                                     std::vector<unsigned char>& imageBytes, std::uint32_t& width,
                                     std::uint32_t& height, std::uint32_t& channels,
                                     std::string& imageIdentifier)
{
    imageIdentifier = getImageIdentifier (asset, imageIndex);
    imageBytes.clear();
    width = height = channels = 0;

    // Handle different image source types
    std::visit ([&] (const auto& source)
                {
        using T = std::decay_t<decltype(source)>;
        
        if constexpr (std::is_same_v<T, fastgltf::sources::BufferView>) {
            const auto& bufferView = asset.bufferViews[source.bufferViewIndex];
            const auto& buffer = asset.buffers[bufferView.bufferIndex];

            std::visit([&](const auto& bufferSource) {
                using BufferT = std::decay_t<decltype(bufferSource)>;
                if constexpr (std::is_same_v<BufferT, fastgltf::sources::Array>) {
                    if (bufferView.byteOffset + bufferView.byteLength <= bufferSource.bytes.size()) {
                        imageBytes.resize(bufferView.byteLength);
                        std::transform(
                            bufferSource.bytes.begin() + bufferView.byteOffset,
                            bufferSource.bytes.begin() + bufferView.byteOffset + bufferView.byteLength,
                            imageBytes.begin(),
                            [](std::byte b) { return static_cast<unsigned char>(b); });
                    }
                }
            }, buffer.data);
        }
        else if constexpr (std::is_same_v<T, fastgltf::sources::URI>) {
            sabiImage.uri = source.uri.string();
            sabiImage.mimeType = fastgltf::getMimeTypeString(source.mimeType);
            LOG(DBUG) << "Stored image URI: " << sabiImage.uri;
        }
        else if constexpr (std::is_same_v<T, fastgltf::sources::Vector> || 
                          std::is_same_v<T, fastgltf::sources::Array>) {
            sabiImage.mimeType = fastgltf::getMimeTypeString(source.mimeType);
            auto start = reinterpret_cast<const unsigned char*>(source.bytes.data());
            imageBytes.assign(start, start + source.bytes.size());
        } },
                image.data);

    // Process image data if available
    if (!imageBytes.empty())
    {
        int w, h, c;
        unsigned char* decodedData = stbi_load_from_memory (
            imageBytes.data(),
            static_cast<int> (imageBytes.size()),
            &w, &h, &c, 0);

        if (decodedData)
        {
            width = static_cast<std::uint32_t> (w);
            height = static_cast<std::uint32_t> (h);
            channels = static_cast<std::uint32_t> (c);
            imageBytes.assign (decodedData, decodedData + (width * height * channels));
            stbi_image_free (decodedData);
        }
    }
}

void GLTFImporter::extractImageData (const fastgltf::Asset& asset, sabi::CgImage& sabiImage,
                                     const fastgltf::Image& image, size_t imageIndex,
                                     std::vector<unsigned char>& imageBytes, std::uint32_t& width,
                                     std::uint32_t& height, std::uint32_t& channels,
                                     std::string& imageIdentifier)
{
    // This implementation is identical to the above but operates on CgImage
    // Reuse the same implementation pattern but with CgImage specific features
    imageIdentifier = getImageIdentifier (asset, imageIndex);
    imageBytes.clear();
    width = height = channels = 0;

    std::visit ([&] (const auto& source)
                {
        using T = std::decay_t<decltype(source)>;
        
        if constexpr (std::is_same_v<T, fastgltf::sources::BufferView>) {
            // Same as above implementation
            const auto& bufferView = asset.bufferViews[source.bufferViewIndex];
            const auto& buffer = asset.buffers[bufferView.bufferIndex];

            std::visit([&](const auto& bufferSource) {
                using BufferT = std::decay_t<decltype(bufferSource)>;
                if constexpr (std::is_same_v<BufferT, fastgltf::sources::Array>) {
                    if (bufferView.byteOffset + bufferView.byteLength <= bufferSource.bytes.size()) {
                        imageBytes.resize(bufferView.byteLength);
                        std::transform(
                            bufferSource.bytes.begin() + bufferView.byteOffset,
                            bufferSource.bytes.begin() + bufferView.byteOffset + bufferView.byteLength,
                            imageBytes.begin(),
                            [](std::byte b) { return static_cast<unsigned char>(b); });
                    }
                }
            }, buffer.data);
        }
        else if constexpr (std::is_same_v<T, fastgltf::sources::URI>) {
            sabiImage.uri = source.uri.string();
            sabiImage.mimeType = fastgltf::getMimeTypeString(source.mimeType);
        }
        else if constexpr (std::is_same_v<T, fastgltf::sources::Vector> || 
                          std::is_same_v<T, fastgltf::sources::Array>) {
            sabiImage.mimeType = fastgltf::getMimeTypeString(source.mimeType);
            auto start = reinterpret_cast<const unsigned char*>(source.bytes.data());
            imageBytes.assign(start, start + source.bytes.size());
        } },
                image.data);

    if (!imageBytes.empty())
    {
        int w, h, c;
        unsigned char* decodedData = stbi_load_from_memory (
            imageBytes.data(),
            static_cast<int> (imageBytes.size()),
            &w, &h, &c, 0);

        if (decodedData)
        {
            width = static_cast<std::uint32_t> (w);
            height = static_cast<std::uint32_t> (h);
            channels = static_cast<std::uint32_t> (c);
            imageBytes.assign (decodedData, decodedData + (width * height * channels));
            stbi_image_free (decodedData);
        }
    }
}

std::string GLTFImporter::mimeTypeToString (fastgltf::MimeType mimeType)
{
    switch (mimeType)
    {
        case fastgltf::MimeType::JPEG:
            return "image/jpeg";
        case fastgltf::MimeType::PNG:
            return "image/png";
        case fastgltf::MimeType::KTX2:
            return "image/ktx2";
        case fastgltf::MimeType::DDS:
            return "image/vnd-ms.dds";
        case fastgltf::MimeType::GltfBuffer:
            return "application/gltf-buffer";
        case fastgltf::MimeType::OctetStream:
            return "application/octet-stream";
        default:
            return "unknown";
    }
}

std::string GLTFImporter::getImageIdentifier (const fastgltf::Asset& asset, size_t imageIndex)
{
    const auto& image = asset.images[imageIndex];
    std::string mimeTypeStr = "unknown";
    size_t byteLength = 0;

    std::visit ([&] (const auto& source)
                {
        using T = std::decay_t<decltype(source)>;
        if constexpr (std::is_same_v<T, fastgltf::sources::BufferView>) {
            const auto& bufferView = asset.bufferViews[source.bufferViewIndex];
            byteLength = bufferView.byteLength;
            mimeTypeStr = mimeTypeToString(source.mimeType);
        }
        else if constexpr (std::is_same_v<T, fastgltf::sources::URI>) {
            mimeTypeStr = mimeTypeToString(source.mimeType);
        } },
                image.data);

    return "Image_" + std::to_string (imageIndex) + "_" +
           mimeTypeStr + "_" + std::to_string (byteLength);
}

std::string GLTFImporter::generateUniqueName (const std::string& baseName)
{
    std::string uniqueName = baseName;
    int counter = 1;

    while (usedNames.find (uniqueName) != usedNames.end())
    {
        uniqueName = baseName + "_" + std::to_string (counter++);
    }
    usedNames.insert (uniqueName);
    return uniqueName;
}


Eigen::Affine3f GLTFImporter::computeLocalTransform (const fastgltf::Node& node)
{
    if (std::holds_alternative<fastgltf::TRS> (node.transform))
    {
        const auto& trs = std::get<fastgltf::TRS> (node.transform);
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();

        // Apply translation
        transform.translate (Eigen::Vector3f (trs.translation[0], trs.translation[1], trs.translation[2]));

        // Apply rotation
        Eigen::Quaternionf rotation (trs.rotation.w(), trs.rotation.x(), trs.rotation.y(), trs.rotation.z());
        transform.rotate (rotation);

        // Apply scale
        transform.scale (Eigen::Vector3f (trs.scale[0], trs.scale[1], trs.scale[2]));

        return transform;
    }
    else if (std::holds_alternative<fastgltf::math::fmat4x4> (node.transform))
    {
        const auto& matrix = std::get<fastgltf::math::fmat4x4> (node.transform);
        Eigen::Matrix4f eigenMatrix;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                eigenMatrix (i, j) = matrix[j][i]; // Note the swapped indices
            }
        }
        return Eigen::Affine3f (eigenMatrix);
    }

    return Eigen::Affine3f::Identity();
}

std::vector<Animation> GLTFImporter::importAnimations (const fastgltf::Asset& asset)
{
    std::vector<Animation> animations;
    animations.reserve (asset.animations.size());

    for (size_t i = 0; i < asset.animations.size(); ++i)
    {
        Animation animation;
        animation.name = asset.animations[i].name;
        animation.channels = animationExporter.exportAnimation (asset, i);
        animations.push_back (std::move (animation));
    }

    return animations;
}

CgModelPtr GLTFImporter::forgeIntoOne (const CgModelList& models)
{
    if (models.empty()) return nullptr;

    // Initialize counters for total vertices and triangles
    uint32_t totalVertices = 0;
    uint32_t totalTriangles = 0;
    bool anyModelHasUVs = false; // Flag to track if any model has UVs

    // Calculate vertex offsets
    std::vector<uint32_t> vertexOffsets;
    vertexOffsets.reserve (models.size());
    vertexOffsets.push_back (totalVertices);

    // Create output model
    auto flattenedModel = CgModel::create();

    // First pass: count totals and check for UV data
    for (const auto& m : models)
    {
        if (!m)
        {
            LOG (CRITICAL) << "Invalid null model in forgeIntoOne";
            continue;
        }

        totalVertices += m->V.cols();
        vertexOffsets.push_back (totalVertices);

        // Verify single surface per mesh constraint
        if (m->S.size() != 1)
        {
            LOG (CRITICAL) << "Multiple surfaces found in mesh during merge";
            return nullptr;
        }

        totalTriangles += m->S[0].triangleCount();

        // Check if this model has UV data
        if (m->UV0.cols() > 0)
        {
            anyModelHasUVs = true;
        }

        // Generate unique surface name
        auto& surface = m->S[0];
        surface.name = generateUniqueName (surface.name.empty() ? "Surface" : surface.name);

        flattenedModel->S.emplace_back (std::move (m->S[0]));
    }

    // Allocate vertex data
    flattenedModel->V.resize (3, totalVertices);

    // Only allocate UV data if any model has UVs
    if (anyModelHasUVs)
    {
        flattenedModel->UV0.resize (2, totalVertices);
    }

    // Copy vertex and UV data
    for (uint32_t index = 0; index < models.size(); ++index)
    {
        const auto& mesh = models[index];
        if (!mesh) continue;

        // Copy vertex data
        std::memcpy (flattenedModel->V.data() + vertexOffsets[index] * 3,
                     mesh->V.data(),
                     mesh->vertexCount() * 3 * sizeof (float));

        // Copy UV data if present and if UV buffer is allocated
        if (mesh->UV0.cols() > 0 && anyModelHasUVs)
        {
            std::memcpy (flattenedModel->UV0.data() + vertexOffsets[index] * 2,
                         mesh->UV0.data(),
                         mesh->vertexCount() * 2 * sizeof (float));
        }
    }

    // Update triangle indices
    for (uint32_t index = 0; index < flattenedModel->S.size(); ++index)
    {
        auto& s = flattenedModel->S[index];

        // Make sure index is valid
        if (index >= vertexOffsets.size() - 1)
        {
            LOG (CRITICAL) << "Vertex offset index out of bounds";
            continue;
        }

        uint32_t vertexOffset = vertexOffsets[index];
        auto& tris = s.indices();

        for (int i = 0; i < s.triangleCount(); i++)
        {
            Vector3u tri = tris.col (i);
            for (int j = 0; j < 3; j++)
            {
                tri[j] += vertexOffset;
            }
            tris.col (i) = tri;
        }
    }

    return flattenedModel;
}

 

sabi::TextureInfo GLTFImporter::importTextureInfo (const fastgltf::Asset& asset, const fastgltf::TextureInfo& textureInfo)
{
    sabi::TextureInfo result;
    result.textureIndex = textureInfo.textureIndex;
    result.texCoord = textureInfo.texCoordIndex;
    return result;
}

sabi::CgTextureInfo GLTFImporter::importCgTextureInfo (const fastgltf::Asset& asset, const fastgltf::TextureInfo& textureInfo)
{
    sabi::CgTextureInfo result;
    result.textureIndex = textureInfo.textureIndex;
    result.texCoordIndex = textureInfo.texCoordIndex;

    if (textureInfo.transform)
    {
        auto transform = std::make_unique<sabi::CgTextureTransform>();
        transform->rotation = textureInfo.transform->rotation;
        transform->uvOffset = Eigen::Vector2f (textureInfo.transform->uvOffset[0], textureInfo.transform->uvOffset[1]);
        transform->uvScale = Eigen::Vector2f (textureInfo.transform->uvScale[0], textureInfo.transform->uvScale[1]);
        transform->texCoordIndex = textureInfo.transform->texCoordIndex;
        result.transform = std::move (transform);
    }

    return result;
}

