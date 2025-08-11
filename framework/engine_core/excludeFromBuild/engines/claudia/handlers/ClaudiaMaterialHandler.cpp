#include "ClaudiaMaterialHandler.h"
#include "../../../handlers/Handlers.h"

using sabi::Image;
// using sabi::Texture;  // Commented out to avoid conflict
using sabi::TextureInfo;

using sabi::CgImage;
using sabi::CgTexture;
using sabi::CgTextureInfo;

// Constructor initializes texture samplers with appropriate filtering and wrapping modes
ClaudiaMaterialHandler::ClaudiaMaterialHandler (RenderContextPtr ctx) :
    ctx (ctx)
{
    LOG (DBUG) << _FN_;

    // Setup sRGB sampler for color textures
    sampler_sRGB.setXyFilterMode (cudau::TextureFilterMode::Linear);
    sampler_sRGB.setWrapMode (0, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setWrapMode (1, cudau::TextureWrapMode::Repeat);
    sampler_sRGB.setMipMapFilterMode (cudau::TextureFilterMode::Linear);
    sampler_sRGB.setReadMode (cudau::TextureReadMode::NormalizedFloat_sRGB);

    // Setup float sampler for HDR textures
    sampler_float.setXyFilterMode (cudau::TextureFilterMode::Linear);
    sampler_float.setWrapMode (0, cudau::TextureWrapMode::Repeat);
    sampler_float.setWrapMode (1, cudau::TextureWrapMode::Repeat);
    sampler_float.setMipMapFilterMode (cudau::TextureFilterMode::Linear);
    sampler_float.setReadMode (cudau::TextureReadMode::ElementType);

    // Setup normalized float sampler for technical textures
    sampler_normFloat.setXyFilterMode (cudau::TextureFilterMode::Linear);
    sampler_normFloat.setWrapMode (0, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setWrapMode (1, cudau::TextureWrapMode::Repeat);
    sampler_normFloat.setMipMapFilterMode (cudau::TextureFilterMode::Linear);
    sampler_normFloat.setReadMode (cudau::TextureReadMode::NormalizedFloat);
}

// Destructor logs cleanup and materials are cleaned up through smart pointers
ClaudiaMaterialHandler::~ClaudiaMaterialHandler()
{
    LOG (DBUG) << _FN_;
    finalize();
}

// Initialize the material handler with buffer allocation
void ClaudiaMaterialHandler::initialize()
{
    if (isInitialized_)
    {
        LOG(WARNING) << "ClaudiaMaterialHandler already initialized";
        return;
    }
    
    LOG(INFO) << "Initializing ClaudiaMaterialHandler";
    
    if (!ctx)
    {
        LOG(WARNING) << "RenderContext is null, cannot initialize";
        return;
    }
    
    CUcontext cuContext = ctx->getCudaContext();
    if (!cuContext)
    {
        LOG(WARNING) << "CUDA context not available";
        return;
    }
    
    // Initialize slot finder for materials
    materialSlotFinder_.initialize(maxNumMaterials);
    LOG(INFO) << "Initialized material slot finder with capacity: " << maxNumMaterials;
    
    // Initialize material data buffer
    materialDataBuffer_.initialize(cuContext, cudau::BufferType::Device, maxNumMaterials);
    LOG(INFO) << "Initialized material data buffer";
    
    isInitialized_ = true;
    LOG(INFO) << "ClaudiaMaterialHandler initialized successfully";
}

// Safely releases all texture and material resources
void ClaudiaMaterialHandler::finalize()
{
    if (!isInitialized_)
        return;
        
    LOG(INFO) << "Finalizing ClaudiaMaterialHandler";
    
    try
    {
        // First sync CUDA operations
        if (ctx && ctx->getCudaStream())
        {
            CUDADRV_CHECK (cuStreamSynchronize (ctx->getCudaStream()));
        }

        // Clear material references which will destroy OptiX/CUDA resources
        materials.clear();

        // Clean up material data buffer
        if (materialDataBuffer_.isInitialized())
            materialDataBuffer_.finalize();
        
        // Clean up slot finder
        materialSlotFinder_.finalize();
        
        isInitialized_ = false;
        LOG (INFO) << "ClaudiaMaterialHandler finalized";
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error during material handler finalization: " << e.what();
    }
}
// Calculates texture dimension information including power-of-two and BC compression status
shared::TexDimInfo ClaudiaMaterialHandler::calcDimInfo (
    const cudau::Array* cuArray,
    bool isLeftHanded)
{
    shared::TexDimInfo dimInfo = {};

    // Handle null array case
    if (!cuArray)
    {
        // LOG (DBUG) << "Creating default 1x1 dimension info (null array)";
        dimInfo.dimX = 1;
        dimInfo.dimY = 1;
        dimInfo.isNonPowerOfTwo = 0;
        dimInfo.isBCTexture = 0;
        dimInfo.isLeftHanded = isLeftHanded;
        return dimInfo;
    }

    try
    {
        // Get dimensions safely
        uint32_t w = static_cast<uint32_t> (cuArray->getWidth());
        uint32_t h = static_cast<uint32_t> (cuArray->getHeight());

        // LOG (DBUG) << "Processing dimensions: " << w << "x" << h;

        // Validate dimensions
        if (w == 0 || h == 0)
        {
            LOG (WARNING) << "Invalid texture dimensions: " << w << "x" << h << ", using 1x1";
            w = 1;
            h = 1;
        }

        bool wIsPowerOfTwo = (w & (w - 1)) == 0;
        bool hIsPowerOfTwo = (h & (h - 1)) == 0;

        dimInfo.dimX = w;
        dimInfo.dimY = h;
        dimInfo.isNonPowerOfTwo = !wIsPowerOfTwo || !hIsPowerOfTwo;
        dimInfo.isBCTexture = cuArray->isBCTexture();
        dimInfo.isLeftHanded = isLeftHanded;

        #if 0
        LOG (DBUG) << "Created dimension info: " << w << "x" << h
                   << " POT: " << (!dimInfo.isNonPowerOfTwo)
                   << " BC: " << dimInfo.isBCTexture;
        #endif
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Exception in calcDimInfo: " << e.what();
        // Return safe defaults
        dimInfo.dimX = 1;
        dimInfo.dimY = 1;
        dimInfo.isNonPowerOfTwo = 0;
        dimInfo.isBCTexture = 0;
        dimInfo.isLeftHanded = isLeftHanded;
    }

    return dimInfo;
}

// Creates base OptiX material and configures hit groups for ray types
optixu::Material ClaudiaMaterialHandler::createOptixMaterial()
{
    optixu::Material mat = ctx->getOptiXContext().createMaterial();

    // NOTE: Hit groups will be set by ClaudiaEngine after pipelines are created
    // This differs from production code where PipelineHandler is accessible via Handlers
    
    return mat;
}

optixu::Material ClaudiaMaterialHandler::createDisneyMaterial (const CgMaterial& material,
                                                        const std::filesystem::path& materialFolder, CgModelPtr model)
{
    // Create the OptiX material first
    optixu::Material mat = createOptixMaterial();
    
    if (!isInitialized_)
    {
        LOG(WARNING) << "ClaudiaMaterialHandler not initialized - falling back to legacy mode";
        // Fall back to the old way without buffer
        auto hostDisney = std::make_unique<DisneyMaterial>();
        // ... process textures and set user data as before
        shared::DisneyData disneyData = {};
        mat.setUserData(disneyData);
        materials.push_back(std::move(hostDisney));
        return mat;
    }
    
    CUcontext cuContext = ctx->getCudaContext();
    if (!cuContext)
    {
        LOG(WARNING) << "No CUDA context available for material creation";
        return mat;
    }
    
    // Allocate a material slot
    uint32_t materialSlot = materialSlotFinder_.getFirstAvailableSlot();
    if (materialSlot == InvalidSlotIndex)
    {
        LOG(WARNING) << "No available material slots";
        return mat;
    }
    materialSlotFinder_.setInUse(materialSlot);
    
    // Map the material data buffer for CPU access
    shared::DisneyData* matDataOnHost = materialDataBuffer_.map();
    if (!matDataOnHost)
    {
        LOG(WARNING) << "Failed to map material data buffer";
        materialSlotFinder_.setNotInUse(materialSlot);
        return mat;
    }
    
    shared::DisneyData& disneyData = matDataOnHost[materialSlot];
    
    // Create host-side material for texture management
    auto hostDisney = std::make_unique<DisneyMaterial>();

    // Process core properties
    processTextureInfo (material.core.baseColorTexture, hostDisney.get(),
                        &hostDisney->baseColor, &hostDisney->texBaseColor, materialFolder, model,
                        material.core.baseColor);

    processTextureInfo (material.core.roughnessTexture, hostDisney.get(),
                        &hostDisney->roughness, &hostDisney->texRoughness, materialFolder, model,
                        Eigen::Vector3f::Constant (material.core.roughness));

    processTextureInfo (material.metallic.metallicTexture, hostDisney.get(),
                        &hostDisney->metallic, &hostDisney->texMetallic, materialFolder, model,
                        Eigen::Vector3f::Constant (material.metallic.metallic));

    processTextureInfo (std::nullopt, hostDisney.get(),
                        &hostDisney->transparency, &hostDisney->texTransparency, materialFolder, model,
                        Eigen::Vector3f::Constant (material.transparency.transparency));

    processTextureInfo (material.transparency.transmittanceTexture, hostDisney.get(),
                        &hostDisney->transmittance, &hostDisney->texTransmittance, materialFolder, model,
                        material.transparency.transmittance);

    processTextureInfo (std::nullopt, hostDisney.get(),
                        &hostDisney->ior, &hostDisney->texIOR, materialFolder, model,
                        Eigen::Vector3f::Constant (material.transparency.refractionIndex));

    processTextureInfo (std::nullopt, hostDisney.get(),
                        &hostDisney->transmittanceDistance, &hostDisney->texTransmittanceDistance, materialFolder, model,
                        Eigen::Vector3f::Constant (material.transparency.transmittanceDistance));

    processTextureInfo (material.sheen.sheenColorTexture, hostDisney.get(),
                        &hostDisney->sheenColor, &hostDisney->texSheenColor, materialFolder, model,
                        material.sheen.sheenColorFactor);

    processTextureInfo (material.clearcoat.clearcoatTexture, hostDisney.get(),
                        &hostDisney->clearcoat, &hostDisney->texClearcoat, materialFolder, model,
                        Eigen::Vector3f::Constant (material.clearcoat.clearcoat));

    processTextureInfo (material.normalTexture, hostDisney.get(),
                        &hostDisney->normal, &hostDisney->texNormal, materialFolder, model);

    processTextureInfo (material.emission.luminousTexture, hostDisney.get(),
                        &hostDisney->emissive, &hostDisney->texEmissive,
                        materialFolder, model,
                        material.emission.luminousColor * material.emission.luminous);

    processTextureInfo (std::nullopt, hostDisney.get(),
                        &hostDisney->emissiveStrength, &hostDisney->texEmissiveStrength,
                        materialFolder, model,
                        Eigen::Vector3f::Constant (material.emission.luminous)); // Process emissive properties

    // Fill in the DisneyData structure directly in the buffer
    // Core properties
    disneyData.baseColor = hostDisney->texBaseColor;
    disneyData.metallic = hostDisney->texMetallic;
    disneyData.roughness = hostDisney->texRoughness;

    // Transparency properties
    disneyData.transparency = hostDisney->texTransparency;
    disneyData.transmittance = hostDisney->texTransmittance;
    disneyData.ior = hostDisney->texIOR;
    disneyData.transmittanceDistance = hostDisney->texTransmittanceDistance;
    disneyData.thinWalled = material.transparency.thin ? 1 : 0;

    // Add emissive properties
    disneyData.emissive = hostDisney->texEmissive;
    disneyData.emissiveStrength = hostDisney->texEmissiveStrength;

    // Sheen properties
    disneyData.sheenColor = hostDisney->texSheenColor;

    // Clearcoat properties
    disneyData.clearcoat = hostDisney->texClearcoat;

    // Normal mapping
    disneyData.normal = hostDisney->texNormal;

    // Set dimension info for all textures
    disneyData.baseColor_dimInfo = calcDimInfo (hostDisney->baseColor);
    disneyData.metallic_dimInfo = calcDimInfo (hostDisney->metallic);
    disneyData.roughness_dimInfo = calcDimInfo (hostDisney->roughness);
    disneyData.transparency_dimInfo = calcDimInfo (hostDisney->transparency);
    disneyData.translucency_dimInfo = calcDimInfo (hostDisney->translucency);
    disneyData.transmittance_dimInfo = calcDimInfo (hostDisney->transmittance);
    disneyData.transmittanceDistance_dimInfo = calcDimInfo (hostDisney->transmittanceDistance);
    disneyData.ior_dimInfo = calcDimInfo (hostDisney->ior);
    disneyData.emissive_dimInfo = calcDimInfo (hostDisney->emissive);
    disneyData.emissiveStrength_dimInfo = calcDimInfo (hostDisney->emissiveStrength);
    disneyData.sheenColor_dimInfo = calcDimInfo (hostDisney->sheenColor);
    disneyData.clearcoat_dimInfo = calcDimInfo (hostDisney->clearcoat);
    disneyData.normal_dimInfo = calcDimInfo (hostDisney->normal);
    disneyData.useAlphaForTransparency = hostDisney->useAlphaForTransparency ? 1 : 0;

    // Unmap the buffer after we're done writing
    materialDataBuffer_.unmap();
    
    // Store the host material for texture lifetime management
    materials.push_back (std::move (hostDisney));
    
    // Set the material slot index in the OptiX material's user data
    // This allows the GPU code to find the material data in the buffer
    mat.setUserData(materialSlot);
    
    LOG(DBUG) << "Created Disney material in slot " << materialSlot;

    return mat;
}

// Process texture information to load and configure a texture
void ClaudiaMaterialHandler::processTextureInfo (
    const std::optional<CgTextureInfo>& texInfo,
    DisneyMaterial* hostDisney,
    const cudau::Array** targetArray,
    CUtexObject* targetTexObject,
    const std::filesystem::path& materialFolder,
    CgModelPtr model,
    const Eigen::Vector3f& defaultVector)
{
    bool needsDegamma = false;
    bool textureLoaded = false;
    bool isHDR = false;

    // Determine which texture we're processing based on the targetArray
    std::string requestedInput;
    if (targetArray == &hostDisney->baseColor)
    {
        requestedInput = "Color";
    }
    else if (targetArray == &hostDisney->roughness)
    {
        requestedInput = "Roughness";
    }
    else if (targetArray == &hostDisney->metallic)
    {
        requestedInput = "Metallic";
    }
    else if (targetArray == &hostDisney->sheenColor)
    {
        requestedInput = "Sheen";
    }
    else if (targetArray == &hostDisney->normal)
    {
        requestedInput = "Normal";
    }
    else if (targetArray == &hostDisney->clearcoat)
    {
        requestedInput = "Clearcoat";
    }
    else if (targetArray == &hostDisney->subsurface)
    {
        requestedInput = "Subsurface";
    }
    else if (targetArray == &hostDisney->emissive)
    {
        requestedInput = "Luminous";
    }
    else if (targetArray == &hostDisney->transmittance)
    {
        requestedInput = "Transmittance";
    }
    else
    {
        requestedInput = "Color"; // Default fallback
    }

    if (requestedInput == "Luminous")
    {
       // LOG (DBUG) << "Processing luminous texture with default value: "
         //          << defaultVector.transpose();

        if (texInfo)
        {
           // LOG (DBUG) << "Has texture info - index: "
             //          << texInfo->textureIndex;
        }
        else
        {
          //  LOG (DBUG) << "Using immediate luminous value";
        }
    }

    if (texInfo)
    {
        const auto& texture = model->cgTextures[texInfo->textureIndex];
        if (texture.imageIndex)
        {
            const auto& image = model->cgImages[*texture.imageIndex];

            //   LOG (DBUG) << "Attempting to load texture: " << image.uri;

            std::filesystem::path texturePath (image.uri);
            if (texturePath.is_absolute())
            {
                // LOG (DBUG) << "Using absolute path: " << texturePath;
                textureLoaded = ctx->getHandlers().textureHandler->loadTexture (
                    texturePath,
                    targetArray,
                    &needsDegamma,
                    &isHDR,
                    requestedInput);
            }
            else
            {
                auto fullPath = FileServices::findFileInFolder (
                    materialFolder,
                    texturePath.filename().string());
                if (fullPath)
                {
                    // LOG (DBUG) << "Found texture at: " << fullPath.value();
                    textureLoaded = ctx->getHandlers().textureHandler->loadTexture (
                        fullPath.value(),
                        targetArray,
                        &needsDegamma,
                        &isHDR,
                        requestedInput);
                }
                else
                {
                    LOG (WARNING) << "Could not find texture: " << texturePath.filename().string()
                                  << " in folder: " << materialFolder;
                }
            }
        }
    }

    if (!textureLoaded)
    {
      //  mace::vecStr3f (defaultVector, DBUG, "IMMEDIATE VALUE");
        float4 immValue = make_float4 (
            defaultVector.x(),
            defaultVector.y(),
            defaultVector.z(),
            1.0f);

        ctx->getHandlers().textureHandler->createImmTexture (immValue, true, targetArray);
        needsDegamma = true;
    }

    if (!*targetArray)
    {
        LOG (WARNING) << "Failed to create texture array";
        return;
    }

    *targetTexObject = needsDegamma ? sampler_sRGB.createTextureObject (**targetArray) : sampler_normFloat.createTextureObject (**targetArray);

    // *** NEW CODE: Check for alpha in base color texture ***
    if (targetArray == &hostDisney->baseColor && requestedInput == "Color")
    {
        bool hasAlpha = ctx->getHandlers().textureHandler->textureHasAlpha (*targetArray);

        if (hasAlpha)
        {
           // LOG (DBUG) << "Texture has meaningful alpha, adjusting transparency";

            // Only apply alpha to transparency if the material isn't already transparent
            if (hostDisney->texTransparency == 0)
            {
                // Use alpha as a transparency source with a 0.5 blend weight for compatibility
                // We'll create a transparency texture, but allow the material system to decide
                // how much to rely on it versus the existing transparency parameter

                // Flag that this material should use alpha-driven transparency
                hostDisney->useAlphaForTransparency = true;

                // The shader will read alpha from the baseColor texture directly
                // so we don't need to create a separate texture
            }
        }
    }
}

// Updates an existing material with new properties
void ClaudiaMaterialHandler::updateMaterial (
    const Material& sabiMaterial,
    optixu::Material& material,
    const std::filesystem::path& materialFolder,
    CgModelPtr model)
{
#if 0
    // Get existing host material if available
    HostMaterial* hostMat = nullptr;
    if (materials.size())
    {
        hostMat = materials[0];
    }
    else
    {
        hostMat = new HostMaterial();
    }

    // Initialize as SimplePBR type
    hostMat->initializeAs (HostMaterial::Type::SimplePBR);

    bool needsDegamma = false;
    bool textureLoaded = false;

    // Process base color texture if present
    if (sabiMaterial.pbrMetallicRoughness.baseColorTexture)
    {
        TextureInfo info = sabiMaterial.pbrMetallicRoughness.baseColorTexture.value();
        Texture texture = model->textures[info.textureIndex];
        Image image = model->images[texture.source];
        std::string textureName = image.uri;

        std::filesystem::path texturePath (textureName);
        if (texturePath.is_absolute())
        {
            textureLoaded = ctx->getHandlers().textureHandler->loadTexture (
                texturePath,
                &hostMat->body.asSimplePBR.baseColor_opacity,
                &needsDegamma);
        }
        else
        {
            std::filesystem::path p (textureName);
            auto fullPath = FileServices::findFileInFolder (materialFolder, p.filename().string());
            if (fullPath)
            {
                textureLoaded = ctx->getHandlers().textureHandler->loadTexture (
                    fullPath.value(),
                    &hostMat->body.asSimplePBR.baseColor_opacity,
                    &needsDegamma);
            }
        }
    }

    // Create immediate base color if no texture
    if (!hostMat->body.asSimplePBR.baseColor_opacity)
    {
        float4 immBaseColor = make_float4 (
            sabiMaterial.pbrMetallicRoughness.baseColorFactor[0],
            sabiMaterial.pbrMetallicRoughness.baseColorFactor[1],
            sabiMaterial.pbrMetallicRoughness.baseColorFactor[2],
            1.0f);
        ctx->getHandlers().textureHandler->createImmTexture (
            immBaseColor,
            true,
            &hostMat->body.asSimplePBR.baseColor_opacity);
        needsDegamma = true;
    }

    // Create texture object with appropriate sampler
    if (needsDegamma)
        hostMat->body.asSimplePBR.texBaseColor_opacity =
            sampler_sRGB.createTextureObject (*hostMat->body.asSimplePBR.baseColor_opacity);
    else
        hostMat->body.asSimplePBR.texBaseColor_opacity =
            sampler_normFloat.createTextureObject (*hostMat->body.asSimplePBR.baseColor_opacity);

    // Process metallic-roughness texture if present
    if (sabiMaterial.pbrMetallicRoughness.metallicRoughnessTexture)
    {
        TextureInfo info = sabiMaterial.pbrMetallicRoughness.metallicRoughnessTexture.value();
        Texture texture = model->textures[info.textureIndex];
        Image image = model->images[texture.source];
        std::string textureName = image.uri;

        std::filesystem::path texturePath (textureName);
        bool metallicNeedsDegamma = false;
        if (texturePath.is_absolute())
        {
            textureLoaded = ctx->getHandlers().textureHandler->loadTexture (
                texturePath,
                &hostMat->body.asSimplePBR.occlusion_roughness_metallic,
                &metallicNeedsDegamma);
        }
        else
        {
            std::filesystem::path p (textureName);
            auto fullPath = FileServices::findFileInFolder (materialFolder, p.filename().string());
            if (fullPath)
            {
                textureLoaded = ctx->getHandlers().textureHandler->loadTexture (
                    fullPath.value(),
                    &hostMat->body.asSimplePBR.occlusion_roughness_metallic,
                    &metallicNeedsDegamma);
            }
        }
    }

    // Create immediate metallic value if no texture
    if (!hostMat->body.asSimplePBR.occlusion_roughness_metallic)
    {
        float metallic = sabiMaterial.pbrMetallicRoughness.metallicFactor;
        ctx->getHandlers().textureHandler->createImmTexture (
            metallic,
            true,
            &hostMat->body.asSimplePBR.occlusion_roughness_metallic);
    }

    // Create texture object for metallic-roughness
    hostMat->body.asSimplePBR.texOcclusion_roughness_metallic =
        sampler_normFloat.createTextureObject (*hostMat->body.asSimplePBR.occlusion_roughness_metallic);

    // Setup material data for GPU
    shared::MaterialData matData = {};
    matData.asSimplePBR.baseColor_opacity = hostMat->body.asSimplePBR.texBaseColor_opacity;
    matData.asSimplePBR.occlusion_roughness_metallic = hostMat->body.asSimplePBR.texOcclusion_roughness_metallic;
    matData.asSimplePBR.baseColor_opacity_dimInfo =
        calcDimInfo (*hostMat->body.asSimplePBR.baseColor_opacity);
    matData.asSimplePBR.occlusion_roughness_metallic_dimInfo =
        calcDimInfo (*hostMat->body.asSimplePBR.occlusion_roughness_metallic);

    // Update the OptiX material with new data
    material.setUserData (matData);

#endif
}