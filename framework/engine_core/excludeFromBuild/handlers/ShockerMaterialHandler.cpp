// ShockerMaterialHandler.cpp
// Implementation of material handling for ShockerModels

#include "ShockerMaterialHandler.h"
#include "../model/ShockerModel.h"
#include "TextureHandler.h"
#include "Handlers.h"
#include <mace_core/mace_core.h>

ShockerMaterialHandler::ShockerMaterialHandler()
{
}

ShockerMaterialHandler::~ShockerMaterialHandler()
{
    clear();
}

void ShockerMaterialHandler::initialize(RenderContext* ctx)
{
    ctx_ = ctx;
    
    // Initialize material data buffer for GPU if we have a context
    if (ctx_ && ctx_->getCudaContext()) {
        // Initialize CUDA buffer for material data
        materialDataBuffer_.initialize(
            ctx_->getCudaContext(),
            cudau::BufferType::Device,
            MaxNumMaterials
        );
    }
    
    // Initialize slot finder
    materialSlotFinder_.initialize(MaxNumMaterials);
    
    // Create a default material
    createDefaultMaterial();
    
    isInitialized_ = true;
    
    LOG(INFO) << "ShockerMaterialHandler initialized with capacity: " << MaxNumMaterials;
}

void ShockerMaterialHandler::clear()
{
    materials_.clear();
    materialSlotFinder_.reset();
    
    if (materialDataBuffer_.isInitialized()) {
        materialDataBuffer_.finalize();
    }
    
    isInitialized_ = false;
    
    LOG(INFO) << "ShockerMaterialHandler cleared";
}

void ShockerMaterialHandler::processMaterialsForModel(
    ShockerModel* model,
    const CgModelPtr& cgModel,
    const std::filesystem::path& materialFolder)
{
    if (!model) {
        LOG(WARNING) << "processMaterialsForModel: null model provided";
        return;
    }
    
    if (!cgModel) {
        LOG(WARNING) << "processMaterialsForModel: null CgModel provided";
        // Assign default material to all geometry instances
        DisneyMaterial* defaultMat = getMaterial(0); // Default is always at index 0
        for (const auto& geomInst : model->getGeometryInstances()) {
            assignMaterialToGeometryInstance(geomInst.get(), defaultMat);
        }
        return;
    }
    
    const auto& geomInstances = model->getGeometryInstances();
    const auto& surfaces = cgModel->S;
    
    // Each surface should have a corresponding geometry instance
    if (geomInstances.size() != surfaces.size()) {
        LOG(WARNING) << "Mismatch: " << geomInstances.size() 
                     << " geometry instances but " << surfaces.size() << " surfaces";
    }
    
    // Process each surface's material
    for (size_t i = 0; i < std::min(geomInstances.size(), surfaces.size()); ++i) {
        GeometryInstance* geomInst = geomInstances[i].get();
        const auto& surface = surfaces[i];
        
        // Get material from surface directly
        const CgMaterial& cgMat = surface.cgMaterial;
        
        // Create material for this surface (TextureHandler already caches textures)
        DisneyMaterial* material = createMaterialFromCg(cgMat, cgModel, materialFolder);
        
        // If no material was created, use default
        if (!material) {
            material = getMaterial(0); // Default material
            LOG(DBUG) << "Using default material for surface " << i;
        }
        
        // Assign material to geometry instance
        assignMaterialToGeometryInstance(geomInst, material);
    }
    
    // Materials processed for model
}

DisneyMaterial* ShockerMaterialHandler::createMaterialFromCg(
    const CgMaterial& cgMaterial,
    const CgModelPtr& model,
    const std::filesystem::path& materialFolder)
{
    // Allocate a new slot
    uint32_t slot = materialSlotFinder_.getFirstAvailableSlot();
    if (slot == InvalidSlotIndex) {
        LOG(WARNING) << "No available material slots";
        return nullptr;
    }
    materialSlotFinder_.setInUse(slot);
    
    // Create new Disney material
    auto disney = std::make_unique<DisneyMaterial>();
    
    // Convert properties from CgMaterial to DisneyMaterial
    convertBaseProperties(disney.get(), cgMaterial.core);
    convertMetallicProperties(disney.get(), cgMaterial.metallic);
    convertSheenProperties(disney.get(), cgMaterial.sheen);
    convertClearcoatProperties(disney.get(), cgMaterial.clearcoat);
    convertSubsurfaceProperties(disney.get(), cgMaterial.subsurface);
    convertTransparencyProperties(disney.get(), cgMaterial.transparency);
    convertEmissionProperties(disney.get(), cgMaterial.emission);
    
    // Process texture information for all material properties
    if (model) {
        // Base color texture
        processTextureInfo(
            cgMaterial.core.baseColorTexture,
            &disney->baseColor,
            &disney->texBaseColor,
            materialFolder,
            model,
            "Color",
            cgMaterial.core.baseColor);
        
        // Roughness texture
        processTextureInfo(
            cgMaterial.core.roughnessTexture,
            &disney->roughness,
            &disney->texRoughness,
            materialFolder,
            model,
            "Roughness",
            Eigen::Vector3f(cgMaterial.core.roughness, cgMaterial.core.roughness, cgMaterial.core.roughness));
        
        // Metallic texture
        processTextureInfo(
            cgMaterial.metallic.metallicTexture,
            &disney->metallic,
            &disney->texMetallic,
            materialFolder,
            model,
            "Metallic",
            Eigen::Vector3f(cgMaterial.metallic.metallic, cgMaterial.metallic.metallic, cgMaterial.metallic.metallic));
        
        // Normal texture
        processTextureInfo(
            cgMaterial.normalTexture,
            &disney->normal,
            &disney->texNormal,
            materialFolder,
            model,
            "Normal",
            Eigen::Vector3f(0.5f, 0.5f, 1.0f));
        
        // Emissive texture
        processTextureInfo(
            cgMaterial.emission.luminousTexture,
            &disney->emissive,
            &disney->texEmissive,
            materialFolder,
            model,
            "Luminous",
            cgMaterial.emission.luminousColor);
    }
    
    // Handle material flags
    disney->useAlphaForTransparency = 
        (cgMaterial.flags.alphaMode == sabi::AlphaMode::Blend);
    
    // Store and return
    DisneyMaterial* result = disney.get();
    materials_.push_back(std::move(disney));
    
    return result;
}

void ShockerMaterialHandler::assignMaterialToGeometryInstance(
    GeometryInstance* geomInst,
    DisneyMaterial* material)
{
    if (!geomInst) {
        LOG(WARNING) << "assignMaterialToGeometryInstance: null geometry instance";
        return;
    }
    
    // GeometryInstance expects a DisneyMaterial pointer
    geomInst->mat = reinterpret_cast<Material*>(material);
    
    // Material assigned to geometry instance
}

void ShockerMaterialHandler::uploadMaterialsToGPU()
{
    if (!ctx_ || !materialDataBuffer_.isInitialized()) {
        LOG(WARNING) << "Cannot upload materials: context or buffer not initialized";
        return;
    }
    
    // Map the buffer for writing
    shared::DisneyData* matDataOnHost = materialDataBuffer_.map();
    if (!matDataOnHost) {
        LOG(WARNING) << "Failed to map material data buffer";
        return;
    }
    
    // Convert all materials to device format and copy to mapped buffer
    size_t materialCount = 0;
    for (const auto& material : materials_) {
        if (materialCount >= MaxNumMaterials) {
            LOG(WARNING) << "Reached maximum material limit: " << MaxNumMaterials;
            break;
        }
        matDataOnHost[materialCount] = convertToDeviceData(material.get());
        materialCount++;
    }
    
    // Unmap the buffer to upload to GPU
    materialDataBuffer_.unmap();
    
    LOG(INFO) << "Uploaded " << materialCount << " materials to GPU";
}

void ShockerMaterialHandler::convertBaseProperties(
    DisneyMaterial* disney,
    const CgMaterial::CoreProperties& core)
{
    // Properties are now handled by processTextureInfo in createMaterialFromCg
    // This function is kept for potential future non-texture property conversion
}

void ShockerMaterialHandler::convertMetallicProperties(
    DisneyMaterial* disney,
    const CgMaterial::MetallicProperties& metallic)
{
    disney->metallic = nullptr;
    disney->texMetallic = 0;
    
    disney->anisotropic = nullptr;
    disney->texAnisotropic = 0;
    
    disney->anisotropicRotation = nullptr;
    disney->texAnisotropicRotation = 0;
}

void ShockerMaterialHandler::convertSheenProperties(
    DisneyMaterial* disney,
    const CgMaterial::SheenProperties& sheen)
{
    disney->sheenColor = nullptr;
    disney->texSheenColor = 0;
    
    disney->sheenRoughness = nullptr;
    disney->texSheenRoughness = 0;
}

void ShockerMaterialHandler::convertClearcoatProperties(
    DisneyMaterial* disney,
    const CgMaterial::ClearcoatProperties& clearcoat)
{
    disney->clearcoat = nullptr;
    disney->texClearcoat = 0;
    
    disney->clearcoatGloss = nullptr;
    disney->texClearcoatGloss = 0;
    
    disney->clearcoatNormal = nullptr;
    disney->texClearcoatNormal = 0;
}

void ShockerMaterialHandler::convertSubsurfaceProperties(
    DisneyMaterial* disney,
    const CgMaterial::SubsurfaceProperties& subsurface)
{
    disney->subsurface = nullptr;
    disney->texSubsurface = 0;
    
    disney->subsurfaceColor = nullptr;
    disney->texSubsurfaceColor = 0;
    
    disney->subsurfaceRadius = nullptr;
    disney->texSubsurfaceRadius = 0;
}

void ShockerMaterialHandler::convertTransparencyProperties(
    DisneyMaterial* disney,
    const CgMaterial::TransparencyProperties& transparency)
{
    disney->transparency = nullptr;
    disney->texTransparency = 0;
    
    disney->transmittance = nullptr;
    disney->texTransmittance = 0;
    
    disney->transmittanceDistance = nullptr;
    disney->texTransmittanceDistance = 0;
    
    disney->ior = nullptr;
    disney->texIOR = 0;
}

void ShockerMaterialHandler::convertEmissionProperties(
    DisneyMaterial* disney,
    const CgMaterial::EmissionProperties& emission)
{
    disney->emissive = nullptr;
    disney->texEmissive = 0;
    
    disney->emissiveStrength = nullptr;
    disney->texEmissiveStrength = 0;
}

void ShockerMaterialHandler::processTextureInfo(
    const std::optional<CgTextureInfo>& texInfo,
    const cudau::Array** targetArray,
    CUtexObject* targetTexObject,
    const std::filesystem::path& materialFolder,
    const CgModelPtr& model,
    const std::string& requestedInput,
    const Eigen::Vector3f& defaultValue)
{
    if (!ctx_ || !ctx_->getHandlers().textureHandler) {
        // No texture handler available - this is expected in tests
        return;
    }
    
    auto texHandler = ctx_->getHandlers().textureHandler;
    bool textureLoaded = false;
    bool needsDegamma = false;
    bool isHDR = false;
    
    // Try to load texture if specified
    if (texInfo && model && !model->cgTextures.empty()) {
        if (texInfo->textureIndex < model->cgTextures.size()) {
            const auto& texture = model->cgTextures[texInfo->textureIndex];
            if (texture.imageIndex && *texture.imageIndex < model->cgImages.size()) {
                const auto& image = model->cgImages[*texture.imageIndex];
                
                std::filesystem::path texturePath(image.uri);
                if (texturePath.is_absolute()) {
                    textureLoaded = texHandler->loadTexture(
                        texturePath,
                        targetArray,
                        &needsDegamma,
                        &isHDR,
                        requestedInput);
                } else {
                    auto fullPath = FileServices::findFileInFolder(
                        materialFolder,
                        texturePath.filename().string());
                    if (fullPath) {
                        textureLoaded = texHandler->loadTexture(
                            fullPath.value(),
                            targetArray,
                            &needsDegamma,
                            &isHDR,
                            requestedInput);
                    } else {
                        LOG(DBUG) << "Could not find texture: " << texturePath.filename().string()
                                  << " in folder: " << materialFolder;
                    }
                }
            }
        }
    }
    
    // Create immediate texture with default value if loading failed
    if (!textureLoaded) {
        float4 immValue = make_float4(
            defaultValue.x(),
            defaultValue.y(),
            defaultValue.z(),
            1.0f);
        texHandler->createImmTexture(immValue, true, targetArray);
        needsDegamma = (requestedInput == "Color" || requestedInput == "Luminous");
    }
    
    // Create texture object if array was created
    if (*targetArray) {
        cudau::TextureSampler sampler;
        sampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
        sampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
        sampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
        sampler.setMipMapFilterMode(cudau::TextureFilterMode::Linear);
        
        if (needsDegamma) {
            sampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);
        } else {
            sampler.setReadMode(cudau::TextureReadMode::NormalizedFloat);
        }
        
        *targetTexObject = sampler.createTextureObject(**targetArray);
        
        // Note: Alpha channel handling would need the DisneyMaterial pointer
        // which isn't available in this context. This could be handled
        // in the calling function if needed.
    }
}

DisneyMaterial* ShockerMaterialHandler::createDefaultMaterial()
{
    // Allocate slot 0 for default material
    uint32_t slot = materialSlotFinder_.getFirstAvailableSlot();
    if (slot != 0) {
        LOG(WARNING) << "Default material not allocated to slot 0, got slot " << slot;
    }
    materialSlotFinder_.setInUse(slot);
    
    // Create default material with neutral gray appearance
    auto disney = std::make_unique<DisneyMaterial>();
    
    // All properties are already initialized to nullptr/0 by the constructor
    // This represents a simple gray diffuse material
    
    DisneyMaterial* result = disney.get();
    materials_.push_back(std::move(disney));
    
    LOG(INFO) << "Created default material at slot " << slot;
    
    return result;
}

shared::TexDimInfo ShockerMaterialHandler::calcDimInfo(
    const cudau::Array* cuArray,
    bool isLeftHanded)
{
    shared::TexDimInfo dimInfo = {};
    
    // Handle null array case
    if (!cuArray) {
        dimInfo.dimX = 1;
        dimInfo.dimY = 1;
        dimInfo.isNonPowerOfTwo = 0;
        dimInfo.isBCTexture = 0;
        dimInfo.isLeftHanded = isLeftHanded;
        return dimInfo;
    }
    
    try {
        // Get dimensions safely
        uint32_t w = static_cast<uint32_t>(cuArray->getWidth());
        uint32_t h = static_cast<uint32_t>(cuArray->getHeight());
        
        // Validate dimensions
        if (w == 0 || h == 0) {
            LOG(WARNING) << "Invalid texture dimensions: " << w << "x" << h << ", using 1x1";
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
    }
    catch (const std::exception& e) {
        LOG(WARNING) << "Exception in calcDimInfo: " << e.what();
        // Return safe defaults
        dimInfo.dimX = 1;
        dimInfo.dimY = 1;
        dimInfo.isNonPowerOfTwo = 0;
        dimInfo.isBCTexture = 0;
        dimInfo.isLeftHanded = isLeftHanded;
    }
    
    return dimInfo;
}

shared::DisneyData ShockerMaterialHandler::convertToDeviceData(const DisneyMaterial* hostMaterial)
{
    shared::DisneyData deviceData;
    
    // Copy texture object handles
    deviceData.baseColor = hostMaterial->texBaseColor;
    deviceData.roughness = hostMaterial->texRoughness;
    deviceData.metallic = hostMaterial->texMetallic;
    deviceData.specular = hostMaterial->texSpecular;
    deviceData.anisotropic = hostMaterial->texAnisotropic;
    deviceData.anisotropicRotation = hostMaterial->texAnisotropicRotation;
    deviceData.sheenColor = hostMaterial->texSheenColor;
    deviceData.sheenRoughness = hostMaterial->texSheenRoughness;
    deviceData.clearcoat = hostMaterial->texClearcoat;
    deviceData.clearcoatGloss = hostMaterial->texClearcoatGloss;
    deviceData.clearcoatNormal = hostMaterial->texClearcoatNormal;
    deviceData.subsurface = hostMaterial->texSubsurface;
    deviceData.subsurfaceColor = hostMaterial->texSubsurfaceColor;
    deviceData.subsurfaceRadius = hostMaterial->texSubsurfaceRadius;
    deviceData.translucency = hostMaterial->texTranslucency;
    deviceData.transparency = hostMaterial->texTransparency;
    deviceData.transmittance = hostMaterial->texTransmittance;
    deviceData.transmittanceDistance = hostMaterial->texTransmittanceDistance;
    deviceData.ior = hostMaterial->texIOR;
    deviceData.normal = hostMaterial->texNormal;
    deviceData.emissive = hostMaterial->texEmissive;
    deviceData.emissiveStrength = hostMaterial->texEmissiveStrength;
    
    // Calculate actual dimension info from textures
    deviceData.baseColor_dimInfo = calcDimInfo(hostMaterial->baseColor);
    deviceData.roughness_dimInfo = calcDimInfo(hostMaterial->roughness);
    deviceData.metallic_dimInfo = calcDimInfo(hostMaterial->metallic);
    deviceData.specular_dimInfo = calcDimInfo(hostMaterial->specular);
    deviceData.anisotropic_dimInfo = calcDimInfo(hostMaterial->anisotropic);
    deviceData.anisotropicRotation_dimInfo = calcDimInfo(hostMaterial->anisotropicRotation);
    deviceData.sheenColor_dimInfo = calcDimInfo(hostMaterial->sheenColor);
    deviceData.sheenRoughness_dimInfo = calcDimInfo(hostMaterial->sheenRoughness);
    deviceData.clearcoat_dimInfo = calcDimInfo(hostMaterial->clearcoat);
    deviceData.clearcoatGloss_dimInfo = calcDimInfo(hostMaterial->clearcoatGloss);
    deviceData.clearcoatNormal_dimInfo = calcDimInfo(hostMaterial->clearcoatNormal, false); // Normal maps are right-handed
    deviceData.subsurface_dimInfo = calcDimInfo(hostMaterial->subsurface);
    deviceData.subsurfaceColor_dimInfo = calcDimInfo(hostMaterial->subsurfaceColor);
    deviceData.subsurfaceRadius_dimInfo = calcDimInfo(hostMaterial->subsurfaceRadius);
    deviceData.translucency_dimInfo = calcDimInfo(hostMaterial->translucency);
    deviceData.transparency_dimInfo = calcDimInfo(hostMaterial->transparency);
    deviceData.transmittance_dimInfo = calcDimInfo(hostMaterial->transmittance);
    deviceData.transmittanceDistance_dimInfo = calcDimInfo(hostMaterial->transmittanceDistance);
    deviceData.ior_dimInfo = calcDimInfo(hostMaterial->ior);
    deviceData.normal_dimInfo = calcDimInfo(hostMaterial->normal, false); // Normal maps are right-handed
    deviceData.emissive_dimInfo = calcDimInfo(hostMaterial->emissive);
    deviceData.emissiveStrength_dimInfo = calcDimInfo(hostMaterial->emissiveStrength);
    
    // Set flags
    deviceData.thinWalled = 0;
    deviceData.doubleSided = 0;
    deviceData.useAlphaForTransparency = hostMaterial->useAlphaForTransparency ? 1 : 0;
    
    return deviceData;
}