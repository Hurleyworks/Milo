#pragma once

// SkyDomeHandler manages HDR environment lighting and sky dome images for OptiX rendering.
// Creates importance sampling maps for efficient IBL (Image-Based Lighting).
// Handles texture creation, mipmap generation, and probability distribution for environment sampling.

#include "../common/common_host.h"
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

class RenderContext;
using RenderContextPtr = std::shared_ptr<RenderContext>;
using SkyDomeHandlerPtr = std::shared_ptr<class SkyDomeHandler>;

class SkyDomeHandler
{
public:
    // Factory method to create a new SkyDomeHandler instance
    static SkyDomeHandlerPtr create(RenderContextPtr ctx) { return std::make_shared<SkyDomeHandler>(ctx); }
    
public:
    // Constructor initializes the handler with a render context
    SkyDomeHandler(RenderContextPtr ctx);
    
    // Destructor cleans up texture and distribution resources
    ~SkyDomeHandler();

    // Adds a new sky dome image to be used as environment lighting
    // Takes ownership of the provided image
    void addSkyDomeImage(const OIIO::ImageBuf&& image);
     
    // Returns the CUDA texture object for the environment lighting
    // Used by shaders to sample the environment during rendering
    CUtexObject getEnvironmentTexture() { return envLightTexture; }
    
    // Returns the importance sampling distribution for the environment map
    // Used to efficiently sample the environment based on light contribution
    RegularConstantContinuousDistribution2D& getImportanceMap() { return envLightImportanceMap; }

    // Check if environment texture is loaded
    bool hasEnvironmentTexture() const { return envLightTexture != 0; }

    // Finalizes and cleans up resources before shutdown or reinitialization
    // Releases CUDA resources and texture objects
    void finalize();

private:
    RenderContextPtr ctx;                // Render context for CUDA operations

    cudau::Array envLightArray;         // CUDA array storing the environment map texture
    CUtexObject envLightTexture = 0;    // CUDA texture object for the environment map
    
    // Importance sampling distribution for the environment map
    // Used to sample directions based on light contribution for efficient rendering
    RegularConstantContinuousDistribution2D envLightImportanceMap;

}; // end class SkyDomeHandler