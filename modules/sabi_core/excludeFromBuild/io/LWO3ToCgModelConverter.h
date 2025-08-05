#pragma once

// Converts LightWave Object (LWO3) layers to CgModel format with support for
// geometry, materials, textures and other attributes. Implements a flexible
// flag-based system for controlling which features are converted.

class LWO3ToCgModelConverter
{
 public:
    // Controls which features are included in the conversion
    enum class ConversionFlags
    {
        None = 0,
        Vertices = 1 << 0,  // Convert vertex positions
        Triangles = 1 << 1, // Convert triangle indices
        UVs = 1 << 2,       // Convert UV coordinates
        Materials = 1 << 3, // Convert materials including images/textures

        BasicGeometry = Vertices | Triangles,
        StandardGeometry = BasicGeometry | UVs,
        Complete = StandardGeometry | Materials
    };

    // Creates converter with specified feature flags
    explicit LWO3ToCgModelConverter (ConversionFlags flags = ConversionFlags::BasicGeometry);

    // Converts a single LWO3 layer to CgModel format
    CgModelPtr convert (const LWO3Layer* layer);

    // Gets the currently enabled conversion flags
    ConversionFlags getFlags() const { return flags_; }

    // Gets the last error message if conversion failed
    const std::string& getError() const { return errorMsg_; }

    // Sets the content directory for resolving relative paths
    void setContentDirectory (const fs::path& dir) { contentDir_ = dir; }
    const fs::path getContentDirectory() const { return contentDir_; }

    // Updates enabled conversion flags
    void setFlags (ConversionFlags flags) { flags_ = flags; }

    // Adds additional conversion flags while preserving existing ones
    void addFlags (ConversionFlags flags)
    {
        flags_ = static_cast<ConversionFlags> (
            static_cast<uint32_t> (flags_) | static_cast<uint32_t> (flags));
    }

 private:
    fs::path contentDir_;
    ConversionFlags flags_;
    std::string errorMsg_;

    std::unordered_map<std::string, size_t> imageTextureMap_;
    std::unordered_map<std::string, size_t> processedImages_;
    std::set<std::string> processedSurfaces_;

    // Validates layer data before conversion
    bool validateLayer (const LWO3Layer* layer);

    // Converts vertex positions to CgModel format
    bool convertVertices (const LWO3Layer* layer, CgModelPtr model);

    // Converts triangle indices to CgModel format
    bool convertTriangles (const LWO3Layer* layer, CgModelPtr model);

    // Converts UV coordinates to CgModel format
    bool convertUVs (const LWO3Layer* layer, CgModelPtr model);

    // Converts images and textures to CgModel format
    bool convertImages (const LWO3Layer* layer, CgModelPtr model);

    // Converts material properties and node graphs to CgModel format
    bool convertMaterials (const LWO3Layer* layer, CgModelPtr model);
};

// Enable bitwise operations on ConversionFlags
inline LWO3ToCgModelConverter::ConversionFlags operator| (
    LWO3ToCgModelConverter::ConversionFlags a,
    LWO3ToCgModelConverter::ConversionFlags b)
{
    return static_cast<LWO3ToCgModelConverter::ConversionFlags> (
        static_cast<uint32_t> (a) | static_cast<uint32_t> (b));
}

inline LWO3ToCgModelConverter::ConversionFlags operator& (
    LWO3ToCgModelConverter::ConversionFlags a,
    LWO3ToCgModelConverter::ConversionFlags b)
{
    return static_cast<LWO3ToCgModelConverter::ConversionFlags> (
        static_cast<uint32_t> (a) & static_cast<uint32_t> (b));
}