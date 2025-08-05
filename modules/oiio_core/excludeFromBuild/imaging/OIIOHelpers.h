#pragma once

using namespace OIIO;

inline bool saveImageBuf (const OIIO::ImageBuf& img, const std::string& filename)
{
    LOG (DBUG) << "Saving image: " << filename
               << ", " << img.spec().width << "x" << img.spec().height
               << ", channels: " << img.spec().nchannels
               << ", format: " << img.spec().format;

    bool success = img.write (filename);
    if (!success)
    {
        LOG (CRITICAL) << "Failed to save image: " << OIIO::geterror();
    }
    return success;
}

inline OIIO::ImageBuf convertHDRtoLDR (const OIIO::ImageBuf& img)
{
    if (!img.initialized())
    {
        LOG (WARNING) << "Attempting to convert uninitialized image";
        return OIIO::ImageBuf();
    }

    OIIO::ImageSpec spec = img.spec();
    spec.format = OIIO::TypeDesc::UINT8;
    OIIO::ImageBuf converted (spec);

    bool success = OIIO::ImageBufAlgo::colorconvert (converted, img, "linear", "sRGB");
    if (!success)
    {
        LOG (CRITICAL) << "Error converting image: " << OIIO::geterror();
        return OIIO::ImageBuf();
    }

    return converted;
}

#if 0
inline OIIO::ImageBuf createAdjustedFrame(const OIIO::ImageBuf& sourceFrame, uint32_t viewportWidth, uint32_t viewportHeight)
{
    const OIIO::ImageSpec& sourceSpec = sourceFrame.spec();

    // If sizes match, return original frame
    if (sourceSpec.width == viewportWidth && sourceSpec.height == viewportHeight)
    {
        return sourceFrame;
    }

    // Create black background buffer matching viewport
    OIIO::ImageBuf background (OIIO::ImageSpec (viewportWidth, viewportHeight, sourceSpec.nchannels, sourceSpec.format));
    OIIO::ImageBufAlgo::zero (background);

    // Calculate scaling to maintain aspect ratio
    float sourceAspect = static_cast<float> (sourceSpec.width) / sourceSpec.height;
    float viewportAspect = static_cast<float> (viewportWidth) / viewportHeight;

    int targetWidth = viewportWidth;
    int targetHeight = viewportHeight;
    int xOffset = 0;
    int yOffset = 0;

    if (sourceAspect > viewportAspect)
    {
        targetHeight = static_cast<int> (viewportWidth / sourceAspect);
        yOffset = (viewportHeight - targetHeight) / 2;
    }
    else
    {
        targetWidth = static_cast<int> (viewportHeight * sourceAspect);
        xOffset = (viewportWidth - targetWidth) / 2;
    }

    // Resize source frame
    OIIO::ROI resizeRoi (0, targetWidth, 0, targetHeight, 0, 1, 0, sourceSpec.nchannels);
    OIIO::ImageBuf resized = OIIO::ImageBufAlgo::resize (sourceFrame, "", 0, resizeRoi);

    // Paste onto background
    OIIO::ImageBufAlgo::paste (background, xOffset, yOffset, 0, 0, resized);

    return background;
}
#endif