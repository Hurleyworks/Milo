
#pragma once




// Node connection information
struct NodeConnection
{
    std::string sourceNode;
    std::string targetNode;
    std::string outputName;
    std::string inputName;
};

// Image node information
struct ImageNodeInfo
{
    std::string nodeName;
    std::string imagePath;
    size_t nodeOffset;
    bool enabled = true;
    std::string uvMapName;
    std::vector<NodeConnection> connections;
};

struct ImageConnection
{
    ImageNodeInfo imageNode;
    std::string bsdfInputSocket; // Which BSDF input it connects to
};

struct StandardNodeInfo
{
    std::string nodeName;
    size_t nodeOffset;
    bool enabled = true;

    // Material properties
    Eigen::Vector3f color = Eigen::Vector3f (0.5f, 0.5f, 0.5f);
    float luminosity = 0.0f;
    float diffuse = 1.0f;
    float specular = 0.75f;
    float glossiness = 0.21f;
    float reflection = 0.0f;
    float transparency = 0.0f;
    float refractionIndex = 1.0f;
    float refractionBlur = 0.0f;
    float translucency = 0.0f;
    float colorHighlight = 0.0f;
    float colorFilter = 0.0f;
    float diffuseSharpness = 0.0f;
    float bumpHeight = 1.0f;

    std::vector<NodeConnection> connections;
};

struct PrincipledBSDFInfo
{
    std::string nodeName;
    size_t nodeOffset;
    bool enabled = true;

    // Base properties
    Eigen::Vector3f baseColor = Eigen::Vector3f (1.0f, 1.0f, 1.0f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float specular = 0.0f;
    float specularTint = 0.0f;

    // Sheen properties
    float sheen = 0.0f;
    float sheenTint = 0.0f;

    // Transparency & Transmission
    float transparency = 0.0f;
    float translucency = 0.0f;
   // float transmission = 0.0f;
    Eigen::Vector3f transmittance = Eigen::Vector3f (1.0f, 1.0f, 1.0f);
    float transmittanceDistance = 0.0f;

    // Subsurface properties
    float subsurface = 0.0f;
    float subsurfaceDistance = 0.0f;
    float flatness = 0.0f;
    Eigen::Vector3f subsurfaceColor = Eigen::Vector3f (1.0f, 1.0f, 1.0f);

    // Clearcoat
    float clearcoat = 0.0f;
    float clearcoatGloss = 0.0f;

    // Special effects
    float anisotropic = 0.0f;
    float anisotropicRotation = 0.0f;
    bool thinWalled = false;
    float asymmetry = 0.0f;

    // Emission
    float luminous = 0.0f;
    Eigen::Vector3f luminousColor = Eigen::Vector3f (0.0f, 0.0f, 0.0f);

    // Index of refraction
    float ior = 1.52f;

    std::vector<NodeConnection> connections;
};

struct NodeSocketDef
{
    std::string name;
    std::string type; // e.g., "Color", "Float", "Vector"
    bool isInput;     // true for input socket, false for output
    bool isConnected; // true if socket has a connection
};

struct NodeSocketInfo
{
    std::string nodeName;
    std::string nodeType;
    std::vector<NodeSocketDef> sockets;
};

// BSDFInput: Represents possible input sockets on a Principled BSDF node
enum class BSDFInput
{
    Color,
    Roughness,
    Normal,
    Metallic,
    Specular,
    SpecularTint,
    Sheen,
    SheenTint,
    Clearcoat,
    ClearcoatRoughness,
    Transparency,
    Translucency,
    Flatness,
    SubsurfaceDistance,
    SubsurfaceScattering,
    Distance,
    Asymmetry,
    Luminous,
    LuminousColor,
    Transmittance,
    TransmittanceDistance,
    Anisotropic,
    AnisotropicRotation,
    Projection
};

// Helper to convert BSDFInput enum to the actual input name used in LightWave
inline std::string getBSDFInputName (BSDFInput input)
{
    switch (input)
    {
        case BSDFInput::Color:
            return "Color";
        case BSDFInput::Roughness:
            return "Roughness";
        case BSDFInput::Normal:
            return "Normal";
        case BSDFInput::Metallic:
            return "Metallic";
        case BSDFInput::Specular:
            return "Specular";
        case BSDFInput::SpecularTint:
            return "Specular Tint";
        case BSDFInput::Sheen:
            return "Sheen";
        case BSDFInput::SheenTint:
            return "Sheen Tint";
        case BSDFInput::Clearcoat:
            return "Clearcoat";
        case BSDFInput::ClearcoatRoughness:
            return "Clearcoat Gloss";
        case BSDFInput::Transparency:
            return "Transparency";
        case BSDFInput::Translucency:
            return "Translucency";
        case BSDFInput::Flatness:
            return "Flatness";
        case BSDFInput::SubsurfaceDistance:
            return "Distance";
        case BSDFInput::SubsurfaceScattering:
            return "Subsurface Scattering";
        case BSDFInput::Distance:
            return "Distance";
        case BSDFInput::Asymmetry:
            return "Asymmetry";
        case BSDFInput::Luminous:
            return "Luminous";
        case BSDFInput::LuminousColor:
            return "Luminous Color";
        case BSDFInput::Transmittance:
            return "Transmittance";
        case BSDFInput::TransmittanceDistance:
            return "Transmittance Distance";
        case BSDFInput::Anisotropic:
            return "Anisotropic";
        case BSDFInput::AnisotropicRotation:
            return "Rotation";
        case BSDFInput::Projection:
            return "Projection";
        default:
            return "";
    }
}