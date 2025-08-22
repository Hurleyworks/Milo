#include "TextureHandler.h"
#include "Handlers.h"

// TextureKind and TextureAnalyzer are from mace_core/basics/Util.h
// They are in the global namespace, included via mace_core.h

// Constructor initializes the texture handler with a render context
TextureHandler::TextureHandler (RenderContextPtr ctx) :
    ctx (ctx)
{
}

// Destructor ensures proper cleanup of all cached textures
TextureHandler::~TextureHandler()
{
    for (auto& it : s_textureCache)
    {
        it.second.texture.finalize();
    }
    for (auto& it : s_Fx1ImmTextureCache)
    {
        it.second.texture.finalize();
    }
    for (auto& it : s_Fx3ImmTextureCache)
    {
        it.second.texture.finalize();
    }
    for (auto& it : s_Fx4ImmTextureCache)
    {
        it.second.texture.finalize();
    }
}
bool TextureHandler::loadTexture (const std::filesystem::path& filePath,
                                  const cudau::Array** texture,
                                  bool* needsDegamma,
                                  bool* isHDR,
                                  const std::string& requestedInput)
{
    // Let TextureAnalyzer determine the format and channels
    TextureAnalyzer analyzer;
    TextureKind texKind = analyzer.analyzeTexture (filePath.string(), requestedInput);

    // Check cache with the requestedInput to differentiate cached versions
    TextureCacheKey cacheKey;
    cacheKey.filePath = filePath;
    cacheKey.cuContext = ctx->getCudaContext();
    cacheKey.requestedInput = requestedInput;

    if (s_textureCache.count (cacheKey))
    {
        const TextureCacheValue& value = s_textureCache.at (cacheKey);
        *texture = &value.texture;
        *needsDegamma = value.needsDegamma;
        if (isHDR)
            *isHDR = value.isHDR;
        return true;
    }

    bool success = true;
    TextureCacheValue cacheValue = {};

    // Load image using OIIO
    OIIO::ImageBuf image = ctx->getImageCache()->getCachedImage (filePath.generic_string(), false);
    OIIO::ImageSpec spec = image.spec();

    if (spec.format != OIIO::TypeDesc::UINT8) return false;

    // Track if this might have meaningful alpha (for analysis later)
    bool mightHaveAlpha = (spec.nchannels == 4 && requestedInput == "Color");

    OIIO::ImageBuf processedImage;
    // Handle packed formats where we need to extract specific channels
    if (texKind.format == TextureKind::Format::OcclusionRoughnessMetallic ||
        texKind.format == TextureKind::Format::RoughnessMetallic)
    {
        // Find the channel we need to extract
        std::string channelOutput;
        for (const auto& channel : texKind.channels)
        {
            if (channel.bsdfInput == requestedInput)
            {
                channelOutput = channel.channelOutput;
                break;
            }
        }

        if (!channelOutput.empty())
        {
            // Extract just the channel we need
            int channelIndex = -1;
            if (channelOutput == "Red")
                channelIndex = 0;
            else if (channelOutput == "Green")
                channelIndex = 1;
            else if (channelOutput == "Blue")
                channelIndex = 2;

            if (channelIndex >= 0)
            {
                int channelorder[] = {channelIndex};
                float channelvalues[] = {0};
                std::string channelnames[] = {requestedInput};
                processedImage = OIIO::ImageBufAlgo::channels (image, 1, channelorder,
                                                               channelvalues, channelnames);
                spec = processedImage.spec();
                mightHaveAlpha = false; // Single channel, no alpha
            }
        }
    }
    else if (spec.nchannels != 4)
    {
        // Standard RGBA conversion for non-packed textures
        float channelvalues[] = {0, 0, 0, 1.0f};
        int channelorder[] = {0, 1, 2, 3};
        std::string channelnames[] = {"R", "G", "B", "A"};
        processedImage = OIIO::ImageBufAlgo::channels (image, 4, channelorder,
                                                       channelvalues, channelnames);
        spec = processedImage.spec();
        mightHaveAlpha = false; // We created an all-opaque alpha channel
    }
    else
    {
        processedImage = image;
        // Keep the original 4-channel image which might have alpha
    }

    const uint8_t* const linearImageData =
        static_cast<uint8_t*> (processedImage.localpixels());

    // NOW we can analyze the alpha channel if needed
    bool hasNonTrivialAlpha = false;
    if (mightHaveAlpha && linearImageData)
    {
        hasNonTrivialAlpha = analyzeAlphaChannel (processedImage);
       // LOG (DBUG) << "Texture " << filePath.filename().string()
            //       << (hasNonTrivialAlpha ? " has non-trivial alpha" : " has trivial alpha");
    }

    // Create CUDA array with appropriate channel count
    if (linearImageData)
    {
        int channelCount = spec.nchannels; // Will be 1 for extracted channels, 4 for regular textures
        cacheValue.texture.initialize2D (
            ctx->getCudaContext(), cudau::ArrayElementType::UInt8, channelCount,
            cudau::ArraySurface::Disable,
            cudau::ArrayTextureGather::Disable,
            spec.width, spec.height, 1);
        cacheValue.texture.write<uint8_t> (linearImageData,
                                           spec.width * spec.height * channelCount);

        cacheValue.needsDegamma = true;
        cacheValue.isHDR = false;

        // Set the alpha flag based on our analysis
        cacheValue.hasAlpha = hasNonTrivialAlpha;
    }
    else
    {
        success = false;
    }

    // Cache and return
    if (success)
    {
        s_textureCache[cacheKey] = std::move (cacheValue);
        *texture = &s_textureCache.at (cacheKey).texture;
        *needsDegamma = s_textureCache.at (cacheKey).needsDegamma;
        if (isHDR)
            *isHDR = s_textureCache.at (cacheKey).isHDR;
    }

    return success;
}

bool TextureHandler::textureHasAlpha (const cudau::Array* texture) const
{
    // Find the texture in any of our caches
    for (const auto& entry : s_textureCache)
    {
        if (&entry.second.texture == texture)
        {
            return entry.second.hasAlpha;
        }
    }

    // Not found in cache, assume no alpha
    return false;
}

// Add this function in TextureHandler.cpp
bool TextureHandler::analyzeAlphaChannel (const OIIO::ImageBuf& image)
{
    const OIIO::ImageSpec& spec = image.spec();
    if (spec.nchannels < 4) return false;

    int alphaChannel = 3; // Alpha is typically the 4th channel (index 3)

    // Sample pixels to determine if alpha channel has varying values
    int sampleCount = 0;
    int nonOpaqueCount = 0;
    int nonTransparentCount = 0;

    // Number of samples to take (adjust as needed)
    const int maxSamples = std::min (100, spec.width * spec.height);

    // Sample in a grid pattern
    int xStep = std::max (1, spec.width / 10);
    int yStep = std::max (1, spec.height / 10);

    for (int y = 0; y < spec.height; y += yStep)
    {
        for (int x = 0; x < spec.width; x += xStep)
        {
            if (sampleCount >= maxSamples) break;

            float pixel[4];
            image.getpixel (x, y, pixel);

            float alpha = pixel[alphaChannel];
            if (alpha < 0.99f) nonOpaqueCount++;
            if (alpha > 0.01f) nonTransparentCount++;

            sampleCount++;
        }
    }

    // If we have both non-opaque and non-transparent pixels, or a significant number
    // of semi-transparent pixels, consider the alpha channel meaningful
    bool meaningful = (nonOpaqueCount > 0 && nonTransparentCount > 0) ||
                      (nonOpaqueCount > sampleCount * 0.05);

  //  LOG (DBUG) << "Alpha analysis: " << nonOpaqueCount << " semi-transparent pixels found in "
        //       << sampleCount << " samples";

    return meaningful;
}
// Template implementation for creating immediate textures
template <typename T>
void TextureHandler::createImmTexture (
    const T& immValue, bool isNormalized, const cudau::Array** texture)
{
    // Select appropriate cache based on value type
    std::map<ImmTextureCacheKey<T>, TextureCacheValue>* textureCache;
    uint32_t numComps = 0;
    if constexpr (std::is_same_v<T, float>)
    {
        textureCache = &s_Fx1ImmTextureCache;
        numComps = 1;
    }
    if constexpr (std::is_same_v<T, float2>)
    {
        textureCache = &s_Fx2ImmTextureCache;
        numComps = 2;
    }
    if constexpr (std::is_same_v<T, float3>)
    {
        textureCache = &s_Fx3ImmTextureCache;
        numComps = 4;
    }
    if constexpr (std::is_same_v<T, float4>)
    {
        textureCache = &s_Fx4ImmTextureCache;
        numComps = 4;
    }

    // Check if texture already exists in cache
    ImmTextureCacheKey<T> cacheKey;
    cacheKey.immValue = immValue;

    if (textureCache->count (cacheKey))
    {
        const TextureCacheValue& value = textureCache->at (cacheKey);
        *texture = &value.texture;
        return;
    }

    TextureCacheValue cacheValue;
    cacheValue.isHDR = !isNormalized;

    // Verify CUDA context is valid
    CUcontext cuContext = ctx->getCudaContext();
    if (!cuContext) {
        LOG(WARNING) << "No CUDA context available for texture creation";
        *texture = nullptr;
        return;
    }
    
    // Create normalized (8-bit) or unnormalized (float) texture
    if (isNormalized)
    {
        uint32_t data;
        if constexpr (std::is_same_v<T, float>)
        {
            data = std::min (static_cast<uint32_t> (255 * immValue), 255u);
        }
        if constexpr (std::is_same_v<T, float2>)
        {
            data = ((std::min (static_cast<uint32_t> (255 * immValue.x), 255u) << 0) |
                    (std::min (static_cast<uint32_t> (255 * immValue.y), 255u) << 8));
        }
        if constexpr (std::is_same_v<T, float3>)
        {
            data = ((std::min (static_cast<uint32_t> (255 * immValue.x), 255u) << 0) |
                    (std::min (static_cast<uint32_t> (255 * immValue.y), 255u) << 8) |
                    (std::min (static_cast<uint32_t> (255 * immValue.z), 255u) << 16) |
                    255 << 24);
        }
        if constexpr (std::is_same_v<T, float4>)
        {
            data = ((std::min (static_cast<uint32_t> (255 * immValue.x), 255u) << 0) |
                    (std::min (static_cast<uint32_t> (255 * immValue.y), 255u) << 8) |
                    (std::min (static_cast<uint32_t> (255 * immValue.z), 255u) << 16) |
                    (std::min (static_cast<uint32_t> (255 * immValue.w), 255u) << 24));
        }

        cacheValue.texture.initialize2D (
            ctx->getCudaContext(), cudau::ArrayElementType::UInt8, numComps,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write (reinterpret_cast<uint8_t*> (&data), numComps);
    }
    else
    {
        float data[4] = {0, 0, 0, 0};

        if constexpr (std::is_same_v<T, float>)
        {
            data[0] = immValue;
        }
        if constexpr (std::is_same_v<T, float2>)
        {
            data[0] = immValue.x;
            data[1] = immValue.y;
        }
        if constexpr (std::is_same_v<T, float3>)
        {
            data[0] = immValue.x;
            data[1] = immValue.y;
            data[2] = immValue.z;
            data[3] = 1.0f;
        }
        if constexpr (std::is_same_v<T, float4>)
        {
            data[0] = immValue.x;
            data[1] = immValue.y;
            data[2] = immValue.z;
            data[3] = immValue.w;
        }

        cacheValue.texture.initialize2D (
            ctx->getCudaContext(), cudau::ArrayElementType::Float32, numComps,
            cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
            1, 1, 1);
        cacheValue.texture.write (data, numComps);
    }

    // Cache the new texture
    (*textureCache)[cacheKey] = std::move (cacheValue);
    *texture = &textureCache->at (cacheKey).texture;
}

// Explicit template instantiations
template void TextureHandler::createImmTexture<float>(const float& immValue, bool isNormalized, const cudau::Array** texture);
template void TextureHandler::createImmTexture<float2>(const float2& immValue, bool isNormalized, const cudau::Array** texture);
template void TextureHandler::createImmTexture<float3>(const float3& immValue, bool isNormalized, const cudau::Array** texture);
template void TextureHandler::createImmTexture<float4>(const float4& immValue, bool isNormalized, const cudau::Array** texture);