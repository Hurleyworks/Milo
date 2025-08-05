#pragma once

// The CameraSensor class represents a digital camera sensor, managing both HDR (High Dynamic Range)
// and LDR (Low Dynamic Range) image data. It utilizes a double-buffering technique to ensure
// thread-safe operations between the rendering thread and the front-end thread.
//
// Key features:
// 1. Double buffering: The class maintains two sets of image buffers (HDR) to allow
//    simultaneous reading and writing operations.
// 2. Thread safety: Uses atomic operations and memory ordering to ensure safe concurrent access.
// 3.
// 4. Flexible resolution: Allows dynamic changes to sensor resolution.
// 5. Performance optimization: Uses OIIO (OpenImageIO) for efficient image processing.
//
// Double buffering implementation:
// - Two image buffers are maintained: images[0] and images[1]
// - currentReadBuffer atomic variable indicates which buffer is currently safe for reading.
// - updateImage() writes to the non-reading buffer, then atomically switches the currentReadBuffer.
// - getHDRImage() always read from the current read buffer.
//
// This approach allows the rendering thread to update the image data while the front-end thread
// can safely read the most recent complete image, preventing data races and ensuring consistency.
//
// The class is designed for high-performance scenarios, such as real-time rendering in a LightWave3D
// plugin, where efficient and thread-safe image handling is crucial.

using Eigen::Vector2i;

class CameraSensor
{
 public:
    CameraSensor() :
        currentReadBuffer (0)
    {
        // Start with default resolution
        setPixelResolution (DEFAULT_DESKTOP_WINDOW_WIDTH, DEFAULT_DESKTOP_WINDOW_HEIGHT);
    }

    void setPixelResolution (uint32_t w, uint32_t h)
    {
        LOG (DBUG) << _FN_ << "   " << w << ", " << h;
        // Create new buffers at new resolution
        OIIO::ImageSpec newSpec (w, h, 4, OIIO::TypeDesc::FLOAT);

        // Initialize both buffers before switching
        OIIO::ImageBuf newBuffers[2];
        newBuffers[0].reset (newSpec, OIIO::InitializePixels::Yes);
        newBuffers[1].reset (newSpec, OIIO::InitializePixels::Yes);

        // Verify both buffers initialized correctly
        if (!newBuffers[0].initialized() || !newBuffers[1].initialized())
        {
            return;
        }

        // Update dimensions
        width.store (w, std::memory_order_release);
        height.store (h, std::memory_order_release);

        // Only swap buffers if both new ones are valid
        images[0].swap (newBuffers[0]);
        images[1].swap (newBuffers[1]);

        aspect = static_cast<float> (w) / static_cast<float> (h);
        invPixelResolution = Eigen::Vector2f (1.0f / static_cast<float> (w),
                                              1.0f / static_cast<float> (h));
    }

    bool getHDRImageCopy (OIIO::ImageBuf& outBuffer) const
    {
        const int readBuffer = currentReadBuffer.load (std::memory_order_acquire);
        const OIIO::ImageBuf& srcBuf = images[readBuffer];

        if (!srcBuf.initialized()) return false;

        // Create new buffer with same specs
        const OIIO::ImageSpec& spec = srcBuf.spec();
        outBuffer.reset (spec);

        // Copy the pixels
        return OIIO::ImageBufAlgo::copy (outBuffer, srcBuf);
    }

    bool updateImage (const void* renderedPixels, Vector2i renderSize, bool previewMode, uint32_t renderScale, bool flipVertical = false)
    {
        // Early validation of input data
        if (!renderedPixels)
        {
            LOG (WARNING) << "Invalid pixel data pointer";
            return false;
        }

        uint32_t renderWidth = renderSize.x();
        uint32_t renderHeight = renderSize.y();

        const int writeBuffer = 1 - currentReadBuffer.load (std::memory_order_acquire);
        const uint32_t viewportWidth = width.load (std::memory_order_acquire);
        const uint32_t viewportHeight = height.load (std::memory_order_acquire);

        // Validate viewport dimensions
        if (viewportWidth == 0 || viewportHeight == 0)
        {
            LOG (WARNING) << "Invalid viewport dimensions: " << viewportWidth << "x" << viewportHeight;
            return false;
        }

        // Validate render dimensions
        if (renderWidth == 0 || renderHeight == 0)
        {
            LOG (WARNING) << "Invalid render dimensions: " << renderWidth << "x" << renderHeight;
            return false;
        }

        /*  LOG (DBUG) << "UpdateImage: viewport=" << viewportWidth << "x" << viewportHeight
                     << " render=" << renderWidth << "x" << renderHeight
                     << " preview=" << (previewMode ? "true" : "false")
                     << " scale=" << renderScale;*/

        if (previewMode && renderScale > 1)
        {
            // Create temp buffer at viewport dimensions
            OIIO::ImageSpec spec (viewportWidth, viewportHeight, 4, OIIO::TypeDesc::FLOAT);
            OIIO::ImageBuf tempBuf (spec);

            // Create source buffer at render dimensions
            OIIO::ImageSpec srcSpec (renderWidth, renderHeight, 4, OIIO::TypeDesc::FLOAT);
            OIIO::ImageBuf srcBuf (srcSpec);

            // Validate source buffer creation
            if (!srcBuf.initialized())
            {
                LOG (WARNING) << "Failed to initialize source buffer";
                return false;
            }

            // Copy rendered pixels to source buffer
            void* srcPixels = srcBuf.localpixels();
            if (!srcPixels)
            {
                LOG (WARNING) << "Failed to get source buffer pixels";
                return false;
            }

            const size_t srcByteSize = renderWidth * renderHeight * 4 * sizeof (float);
            std::memcpy (srcPixels, renderedPixels, srcByteSize);

            // Resize from render dimensions to viewport dimensions
            if (!OIIO::ImageBufAlgo::resize (tempBuf, srcBuf))
            {
                LOG (WARNING) << "Resize failed: " << tempBuf.geterror();
                return false;
            }

            // Initialize write buffer at viewport dimensions
            images[writeBuffer].reset (spec, OIIO::InitializePixels::Yes);
            if (!images[writeBuffer].initialized())
            {
                LOG (WARNING) << "Failed to initialize write buffer";
                return false;
            }

            // Copy resized result
            if (!OIIO::ImageBufAlgo::copy (images[writeBuffer], tempBuf))
            {
                LOG (WARNING) << "Failed to copy to write buffer";
                return false;
            }
        }
        else
        {
            // Create buffers - source at render dimensions, destination at viewport dimensions
            OIIO::ImageSpec srcSpec (renderWidth, renderHeight, 4, OIIO::TypeDesc::FLOAT);
            OIIO::ImageBuf srcBuf (srcSpec);

            void* srcPixels = srcBuf.localpixels();
            if (!srcPixels)
            {
                LOG (WARNING) << "Failed to get source buffer pixels";
                return false;
            }

            const size_t srcByteSize = renderWidth * renderHeight * 4 * sizeof (float);
            std::memcpy (srcPixels, renderedPixels, srcByteSize);

            // If dimensions don't match, resize
            if (renderWidth != viewportWidth || renderHeight != viewportHeight)
            {
                OIIO::ImageSpec destSpec (viewportWidth, viewportHeight, 4, OIIO::TypeDesc::FLOAT);
                OIIO::ImageBuf destBuf (destSpec);

                if (!OIIO::ImageBufAlgo::resize (destBuf, srcBuf))
                {
                    LOG (WARNING) << "Failed to resize rendered image: " << destBuf.geterror();
                    return false;
                }

                images[writeBuffer].reset (destSpec, OIIO::InitializePixels::Yes);
                if (!OIIO::ImageBufAlgo::copy (images[writeBuffer], destBuf))
                {
                    LOG (WARNING) << "Failed to copy resized image";
                    return false;
                }
            }
            else
            {
                // Dimensions match, direct copy
                images[writeBuffer].reset (srcSpec, OIIO::InitializePixels::Yes);
                if (!OIIO::ImageBufAlgo::copy (images[writeBuffer], srcBuf))
                {
                    LOG (WARNING) << "Failed to copy to write buffer";
                    return false;
                }
            }
        }

        if (flipVertical)
        {
            // Flip the result vertically
            OIIO::ImageBuf flipped;
            if (!OIIO::ImageBufAlgo::flip (flipped, images[writeBuffer]))
            {
                LOG (WARNING) << "Failed to flip buffer: " << flipped.geterror();
                return false;
            }
            images[writeBuffer].copy (flipped);
        }
        

        currentReadBuffer.store (writeBuffer, std::memory_order_release);
        return true;
    }
#if 0
    bool updateImage (const void* renderedPixels, Vector2i renderSize, bool previewMode, uint32_t renderScale)
    {
        if (!renderedPixels) return false;

        const int writeBuffer = 1 - currentReadBuffer.load (std::memory_order_acquire);
        const uint32_t viewportWidth = width.load (std::memory_order_acquire);
        const uint32_t viewportHeight = height.load (std::memory_order_acquire);

        if (previewMode && renderScale > 1)
        {
            // Create temp buffer at viewport dimensions (not render dimensions)
            OIIO::ImageSpec spec (viewportWidth, viewportHeight, 4, OIIO::TypeDesc::FLOAT);
            OIIO::ImageBuf tempBuf (spec);

            // Create source buffer at render dimensions
            OIIO::ImageSpec srcSpec (viewportWidth * renderScale, viewportHeight * renderScale, 4, OIIO::TypeDesc::FLOAT);
            OIIO::ImageBuf srcBuf (srcSpec);

            // Copy rendered pixels to source buffer
            void* srcPixels = srcBuf.localpixels();
            if (!srcPixels) return false;

            const size_t srcByteSize = srcSpec.width * srcSpec.height * 4 * sizeof (float);
            std::memcpy (srcPixels, renderedPixels, srcByteSize);

            // Resize from render dimensions to viewport dimensions
            if (!OIIO::ImageBufAlgo::resize (tempBuf, srcBuf))
            {
                LOG (WARNING) << "Resize failed: " << tempBuf.geterror();
                return false;
            }

            // Initialize write buffer at viewport dimensions
            images[writeBuffer].reset (spec, OIIO::InitializePixels::Yes);
            if (!images[writeBuffer].initialized())
            {
                LOG (WARNING) << "Failed to initialize write buffer";
                return false;
            }

            // Copy resized result
            if (!OIIO::ImageBufAlgo::copy (images[writeBuffer], tempBuf))
            {
                LOG (WARNING) << "Failed to copy to write buffer";
                return false;
            }
        }
        else
        {
            // Handle non-preview mode (same as original)
            OIIO::ImageSpec spec (viewportWidth, viewportHeight, 4, OIIO::TypeDesc::FLOAT);
            OIIO::ImageBuf tempBuf (spec);

            void* destPixels = tempBuf.localpixels();
            if (!destPixels) return false;

            const size_t byteSize = viewportWidth * viewportHeight * 4 * sizeof (float);
            std::memcpy (destPixels, renderedPixels, byteSize);

            images[writeBuffer].reset (spec, OIIO::InitializePixels::Yes);
            if (!OIIO::ImageBufAlgo::copy (images[writeBuffer], tempBuf))
            {
                return false;
            }
        }

        // Flip the result vertically
        OIIO::ImageBuf flipped;
        if (!OIIO::ImageBufAlgo::flip (flipped, images[writeBuffer]))
        {
            LOG (WARNING) << "Failed to flip buffer: " << flipped.geterror();
            return false;
        }
        images[writeBuffer].copy (flipped);

        currentReadBuffer.store (writeBuffer, std::memory_order_release);
        return true;
    }
#endif

    bool isValid() const
    {
        const int readBuffer = currentReadBuffer.load (std::memory_order_acquire);
        return images[readBuffer].initialized();
    }

    const OIIO::ImageBuf& getHDRImage() const
    {
        const int readBuffer = currentReadBuffer.load (std::memory_order_acquire);
        return images[readBuffer];
    }

    Eigen::Vector2i getPixelResolution() const
    {
        return Eigen::Vector2i (
            width.load (std::memory_order_acquire),
            height.load (std::memory_order_acquire));
    }
    // Returns the size of one pixel (1 / pixel resolution)
    const Eigen::Vector2f& pixelSize() const
    {
        return invPixelResolution;
    }

    float getPixelAspectRatio() const { return aspect; }
    void setPixelAspectRatio (float aspect) { this->aspect = aspect; }

    const Eigen::Vector2f& getSensorSize() const
    {
        return sensorSize;
    }

 private:
    float aspect;
    Eigen::Vector2f invPixelResolution;
    Eigen::Vector2f sensorSize = Eigen::Vector2f (0.036f, 0.024f);

    std::array<OIIO::ImageBuf, 2> images;
    std::atomic<int> currentReadBuffer;
    std::atomic<uint32_t> width;
    std::atomic<uint32_t> height;
};