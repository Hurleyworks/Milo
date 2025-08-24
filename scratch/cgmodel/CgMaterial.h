
#pragma once

#pragma once

#include <optional>
#include <memory>

struct CgTextureTransform
{
    float rotation = 0.0f;
    Eigen::Vector2f uvOffset = Eigen::Vector2f::Zero();
    Eigen::Vector2f uvScale = Eigen::Vector2f::Ones();
    std::optional<std::size_t> texCoordIndex;
};

struct CgTextureInfo
{
    std::size_t textureIndex;
    std::size_t texCoordIndex = 0;
    // Changed back to unique_ptr
    std::unique_ptr<CgTextureTransform> transform;

    // Special texture info variants
    float scale = 1.0f;    // For normal maps
    float strength = 1.0f; // For occlusion maps

    // Default Constructor
    CgTextureInfo() = default;

    // Copy Constructor (Deep Copy)
    CgTextureInfo (const CgTextureInfo& other) :
        textureIndex (other.textureIndex),
        texCoordIndex (other.texCoordIndex),
        scale (other.scale),
        strength (other.strength)
    {
        if (other.transform)
        {
            transform = std::make_unique<CgTextureTransform> (*other.transform);
        }
    }
    // Move Constructor
    CgTextureInfo (CgTextureInfo&& other) noexcept = default;

    // Copy Assignment Operator (Deep Copy)
    CgTextureInfo& operator= (const CgTextureInfo& other)
    {
        textureIndex = other.textureIndex;
        texCoordIndex = other.texCoordIndex;
        scale = other.scale;
        strength = other.strength;

        if (other.transform)
        {
            transform = std::make_unique<CgTextureTransform> (*other.transform);
        }
        else
        {
            transform.reset();
        }
        return *this;
    }
    // Move Assignment operator
    CgTextureInfo& operator= (CgTextureInfo&& other) noexcept = default;
};

struct CgTexture
{
    std::optional<std::size_t> samplerIndex;
    std::optional<std::size_t> imageIndex;

    // Extension texture indices
    std::optional<std::size_t> basisuImageIndex;
    std::optional<std::size_t> ddsImageIndex;
    std::optional<std::size_t> webpImageIndex;

    std::string name;
};

struct CgImage
{
    std::string uri;
    std::string mimeType;

    // For embedded/loaded images
    OIIO::ImageBuf extractedImage;

    std::size_t index = 0;
    std::string name;
};

// Using our own enum types to avoid dependency on fastgltf
enum class CgFilter : uint16_t
{
    Nearest = 9728,
    Linear = 9729,
    NearestMipMapNearest = 9984,
    LinearMipMapNearest = 9985,
    NearestMipMapLinear = 9986,
    LinearMipMapLinear = 9987
};

enum class CgWrap : uint16_t
{
    ClampToEdge = 33071,
    MirroredRepeat = 33648,
    Repeat = 10497
};

struct CgSampler
{
    std::optional<CgFilter> magFilter;
    std::optional<CgFilter> minFilter;
    CgWrap wrapS = CgWrap::Repeat;
    CgWrap wrapT = CgWrap::Repeat;

    std::string name;
};

enum class AlphaMode : std::uint8_t
{
    Opaque,
    Mask,
    Blend,
};

struct CgMaterial
{
    // Core PBR Properties aligned with UI organization
    struct CoreProperties
    {
        Eigen::Vector3f baseColor = {0.5f, 0.5f, 0.5f};
        float roughness = 0.0f;
        float specular = 0.5f;
        float specularTint = 0.0f;
        std::optional<CgTextureInfo> baseColorTexture;
        std::optional<CgTextureInfo> roughnessTexture;
    };

    struct SheenProperties
    {
        Eigen::Vector3f sheenColorFactor = {0.0f,0.0f, 0.0f};
        float sheenRoughnessFactor = 0.0f;
        std::optional<CgTextureInfo> sheenColorTexture;
        std::optional<CgTextureInfo> sheenRoughnessTexture;
    };

    struct TranslucencyProperties
    {
        float translucency = 0.0f;
        float flatness = 0.0f;
        std::optional<CgTextureInfo> translucencyTexture;
    };

    struct SubsurfaceProperties
    {
        float subsurface = 0.0f;
        Eigen::Vector3f subsurfaceColor = {1.0f, 1.0f, 1.0f};
        float subsurfaceDistance = 1.0f; // mm
        float asymmetry = 0.0f;
        std::optional<CgTextureInfo> subsurfaceColorTexture;
    };

    struct EmissionProperties
    {
        float luminous = 0.0f;
        Eigen::Vector3f luminousColor = {0.5f, 0.5f, 0.5f};
        std::optional<CgTextureInfo> luminousTexture;
    };

    struct MetallicProperties
    {
        float metallic = 0.0f;
        float anisotropic = 0.0f;
        float anisotropicRotation = 0.0f;
        std::optional<CgTextureInfo> metallicTexture;
        std::optional<CgTextureInfo> anisotropicTexture;
        std::optional<CgTextureInfo> anisotropicRotationTexture;
    };

    struct ClearcoatProperties
    {
        float clearcoat = 0.0f;
        float clearcoatGloss = 1.0f;
        std::optional<CgTextureInfo> clearcoatTexture;
        std::optional<CgTextureInfo> clearcoatRoughnessTexture;
        std::optional<CgTextureInfo> clearcoatNormalTexture;
    };

    struct TransparencyProperties
    {
        bool thin = false;
        float transparency = 0.0f;
        Eigen::Vector3f transmittance = {0.5f, 0.5f, 0.5f};
        float transmittanceDistance = 1.0f; // meters
        float refractionIndex = 1.5f;
        std::optional<CgTextureInfo> transparencyTexture;
        std::optional<CgTextureInfo> transmittanceTexture;
    };

    struct PackedTextureProperties
    {
        std::optional<CgTextureInfo> occlusionRoughnessMetallicTexture;
        std::optional<CgTextureInfo> normalRoughnessMetallicTexture;
        std::optional<CgTextureInfo> roughnessMetallicOcclusionTexture;
    };

    // Main material components
    CoreProperties core;
    SheenProperties sheen;
    TranslucencyProperties translucency;
    SubsurfaceProperties subsurface;
    EmissionProperties emission;
    MetallicProperties metallic;
    ClearcoatProperties clearcoat;
    TransparencyProperties transparency;
    PackedTextureProperties packedTextures;

    // Common material properties
    float bumpHeight = 1.0f;
    std::optional<CgTextureInfo> normalTexture;
    std::optional<CgTextureInfo> bumpTexture;
    std::optional<CgTextureInfo> occlusionTexture;
    bool doubleSided = false;

    // Material variant support
    std::vector<std::size_t> variantIndices;

    std::string name;

    // Add Material flags
    struct Flags
    {
        bool unlit = false;
        AlphaMode alphaMode = AlphaMode::Opaque;
        float alphaCutoff = 0.5f;
    } flags;
};