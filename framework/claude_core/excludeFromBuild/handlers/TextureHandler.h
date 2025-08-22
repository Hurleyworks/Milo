// TextureHandler is a resource management class for GPU textures in an OptiX-based renderer.
// It provides centralized texture management with the following key features:
//
// Resource Management:
// - Loads and manages textures from files and immediate values
// - Handles CPU to GPU texture data transfer
// - Manages CUDA arrays and texture objects
// - Implements automatic resource cleanup
//
// Caching System:
// - Maintains separate caches for file-based and immediate textures
// - Prevents duplicate texture loading and memory waste
// - Supports different texture formats (1D, 2D, 3D, 4D values)
// - Implements cache keys based on file paths and CUDA contexts
//
// Texture Processing:
// - Handles format conversion (e.g., RGB to RGBA)
// - Supports normalized (0-1) and unnormalized (HDR) values
// - Provides gamma correction handling
// - Manages different pixel formats (8-bit, 32-bit float)
//
// Memory Optimization:
// - Uses shared pointers for automatic resource management
// - Implements texture reuse through caching
// - Provides cleanup of GPU resources in destructor
// - Manages CUDA context-specific resources
//
// Integration:
// - Works with OptiX ray tracing system
// - Integrates with CUDA runtime
// - Supports material system requirements
// - Handles different texture types (color, normal maps, etc.)
//
// Thread Safety:
// - Not thread-safe by default
// - Requires external synchronization for multi-threaded access
// - Cache access should be synchronized across threads
//
// Usage:
// - Create via factory method TextureHandler::create()
// - Load textures using loadTexture()
// - Create immediate textures using createImmTexture()
// - Access cached textures through provided methods

// Immediate textures are GPU textures created directly from in-memory values rather than loaded from files.
// They are commonly used for:
// - Creating solid color textures (e.g., base colors for materials)
// - Setting default texture values
// - Generating simple patterns or gradients
// - Creating placeholder textures
// - Runtime-generated texture data
//
// The template parameter T supports several types:
// - float: Single channel (grayscale) textures
// - float2: Two channel textures
// - float3: RGB color textures (automatically expanded to RGBA)
// - float4: RGBA color textures
//
// The isNormalized parameter determines the storage format:
// - true: Values are clamped to [0,1] and stored as 8-bit unsigned integers (0-255)
// - false: Values are stored as 32-bit floats, allowing HDR values
//
// Caching:
// - Immediate textures are cached based on their values and CUDA context
// - Separate caches exist for each value type (float, float2, float3, float4)
// - Cached textures are automatically cleaned up when the TextureHandler is destroyed
//
// Example usages:
// - float gray = 0.5f;              // 50% gray
// - float3 red = {1.0f, 0, 0};     // Pure red
// - float4 blue = {0, 0, 1.0f, 1.0f}; // Pure blue with full opacity

#pragma once

#include "../RenderContext.h"

using TextureHandlerPtr = std::shared_ptr<class TextureHandler>;

// TextureHandler manages GPU textures for an OptiX-based renderer
// It provides caching, immediate texture creation, and texture loading capabilities
// The class optimizes memory usage by reusing textures and managing GPU resources
class TextureHandler
{
    // Defines various types of bump maps that can be handled
    enum class BumpMapTextureType
    {
        NormalMap = 0,
        NormalMap_BC,     // Block compressed normal map
        NormalMap_BC_2ch, // Block compressed 2-channel normal map
        HeightMap,
        HeightMap_BC, // Block compressed height map
    };

 public:
    // Factory method to create a shared TextureHandler instance
    static TextureHandlerPtr create (RenderContextPtr ctx)
    {
        return std::make_shared<TextureHandler> (ctx);
    }

    TextureHandler (RenderContextPtr ctx);
    ~TextureHandler();

    // Loads a texture from a file and returns it as a CUDA array
    // Handles RGBA conversion and implements caching for performance
    // Returns true if successful, false otherwise
    // needsDegamma indicates if the texture requires gamma correction
    // isHDR indicates if the texture contains HDR data
    bool loadTexture (const std::filesystem::path& filePath,
                      const cudau::Array** texture,
                      bool* needsDegamma,
                      bool* isHDR,
                      const std::string& requestedInput);

    // Creates an immediate texture from a value in memory
    // T can be float (grayscale), float2, float3 (RGB), or float4 (RGBA)
    // isNormalized determines if values are treated as 0-1 range (normalized)
    // or as unbounded values (unnormalized)
    template <typename T>
    void createImmTexture (
        const T& immValue,
        bool isNormalized,
        const cudau::Array** texture);

    bool textureHasAlpha (const cudau::Array* texture) const;

 private:
    RenderContextPtr ctx = nullptr;
    std::vector<std::shared_ptr<cudau::Array>> textures;

    bool analyzeAlphaChannel (const OIIO::ImageBuf& image);

    // Key structure for file-based texture cache
    // Combines filepath and CUDA context to uniquely identify a texture
    struct TextureCacheKey
    {
        std::filesystem::path filePath;
        CUcontext cuContext;
        std::string requestedInput; // Add this

        bool operator< (const TextureCacheKey& rKey) const
        {
            if (filePath < rKey.filePath) return true;
            if (filePath > rKey.filePath) return false;
            if (cuContext < rKey.cuContext) return true;
            if (cuContext > rKey.cuContext) return false;
            return requestedInput < rKey.requestedInput;
        }
    };

    // Key structure for immediate textures
    // Combines the immediate value and CUDA context
    template <typename T>
    struct ImmTextureCacheKey
    {
        T immValue;
        CUcontext cuContext;

        bool operator< (const ImmTextureCacheKey& rKey) const
        {
            if constexpr (std::is_same_v<T, float>)
            {
                if (immValue < rKey.immValue)
                    return true;
                else if (immValue > rKey.immValue)
                    return false;
            }
            else
            {
                if constexpr (std::is_same_v<T, float4>)
                {
                    if (immValue.w < rKey.immValue.w)
                        return true;
                    else if (immValue.w > rKey.immValue.w)
                        return false;
                }
                if constexpr (std::is_same_v<T, float4> || std::is_same_v<T, float3>)
                {
                    if (immValue.z < rKey.immValue.z)
                        return true;
                    else if (immValue.z > rKey.immValue.z)
                        return false;
                }
                if (immValue.y < rKey.immValue.y)
                    return true;
                else if (immValue.y > rKey.immValue.y)
                    return false;
                if (immValue.x < rKey.immValue.x)
                    return true;
                else if (immValue.x > rKey.immValue.x)
                    return false;
            }
            if (cuContext < rKey.cuContext)
                return true;
            else if (cuContext > rKey.cuContext)
                return false;
            return false;
        }
    };

    // Value structure for texture cache entries
    // Stores the actual texture data and metadata
    struct TextureCacheValue
    {
        cudau::Array texture;           // The CUDA array containing texture data
        bool needsDegamma;              // Indicates if gamma correction is needed
        bool isHDR;                     // Indicates if texture contains HDR data
        BumpMapTextureType bumpMapType; // Type of bump map if applicable
        bool hasAlpha = false;          // Flag to indicate meaningful alpha data
    };

    // Cache mappings for different texture types
    std::map<TextureCacheKey, TextureCacheValue> s_textureCache;                  // File-based textures
    std::map<ImmTextureCacheKey<float>, TextureCacheValue> s_Fx1ImmTextureCache;  // Grayscale immediate
    std::map<ImmTextureCacheKey<float2>, TextureCacheValue> s_Fx2ImmTextureCache; // 2-channel immediate
    std::map<ImmTextureCacheKey<float3>, TextureCacheValue> s_Fx3ImmTextureCache; // RGB immediate
    std::map<ImmTextureCacheKey<float4>, TextureCacheValue> s_Fx4ImmTextureCache; // RGBA immediate
};
