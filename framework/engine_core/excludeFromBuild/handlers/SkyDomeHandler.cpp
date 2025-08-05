#include "SkyDomeHandler.h"
#include "../RenderContext.h"

// ctor
SkyDomeHandler::SkyDomeHandler(RenderContextPtr ctx) :
    ctx(ctx)
{
    LOG(DBUG) << _FN_;
}

// dtor
SkyDomeHandler::~SkyDomeHandler()
{
    LOG(DBUG) << _FN_;
    finalize();
}

void SkyDomeHandler::addSkyDomeImage(const OIIO::ImageBuf&& image)
{
    LOG(DBUG) << _FN_;
    
    if (envLightTexture)
        finalize();

    // need to add an alpha channel
    int channelorder[] = {0, 1, 2, 3};
    float channelvalues[] = {0 /*ignore*/, 0 /*ignore*/, 0 /*ignore*/, 1.0f};
    std::string channelnames[] = {"", "", "", "A"};
    OIIO::ImageBuf rgba = OIIO::ImageBufAlgo::channels(image, 4, channelorder, channelvalues, channelnames);

    const OIIO::ImageSpec& spec = rgba.spec();

    cudau::TextureSampler sampler_float;
    sampler_float.setXyFilterMode(cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode(0, cudau::TextureWrapMode::Clamp);
    sampler_float.setWrapMode(1, cudau::TextureWrapMode::Clamp);
    sampler_float.setMipMapFilterMode(cudau::TextureFilterMode::Point);
    sampler_float.setReadMode(cudau::TextureReadMode::ElementType);

    int32_t width = spec.width;
    int32_t height = spec.height;

    // don't delete ... owned by ImageBuf
    float* textureData = static_cast<float*>(rgba.localpixels());

    float* const importanceData = new float[width * height];
    for (int y = 0; y < height; ++y)
    {
        float theta = pi_v<float> * (y + 0.5f) / height;
        float sinTheta = std::sin(theta);
        for (int x = 0; x < width; ++x)
        {
            uint32_t idx = 4 * (y * width + x);
            textureData[idx + 0] = std::max(textureData[idx + 0], 0.0f);
            textureData[idx + 1] = std::max(textureData[idx + 1], 0.0f);
            textureData[idx + 2] = std::max(textureData[idx + 2], 0.0f);
            RGB value(textureData[idx + 0],
                       textureData[idx + 1],
                       textureData[idx + 2]);
            importanceData[y * width + x] = sRGB_calcLuminance(value) * sinTheta;
        }
    }

    envLightArray.initialize2D(
        ctx->getCudaContext(), cudau::ArrayElementType::Float32, 4,
        cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
        width, height, 1);

    envLightArray.write(textureData, width * height * 4);

    envLightImportanceMap.initialize(
        ctx->getCudaContext(), cudau::BufferType::Device, importanceData, width, height);

    delete[] importanceData;

    envLightTexture = sampler_float.createTextureObject(envLightArray);
    
    LOG(INFO) << "Sky dome environment texture loaded: " << width << "x" << height;
}

void SkyDomeHandler::finalize()
{
    if (ctx && envLightTexture) {
        envLightImportanceMap.finalize(ctx->getCudaContext());
        cuTexObjectDestroy(envLightTexture);
        envLightArray.finalize();
        envLightTexture = 0;
        LOG(DBUG) << "SkyDomeHandler resources finalized";
    }
}