#pragma once

// ShockerMaterialHandler.h
// Manages material creation and assignment for ShockerModels
// Converts CgMaterials to DisneyMaterials and assigns them to GeometryInstances

#include "../RenderContext.h"
#include "../common/common_host.h"
#include "../material/HostDisneyMaterial.h"
#include "../material/DeviceDisneyMaterial.h"
#include "../model/ShockerModel.h"
#include "../model/ShockerCore.h"

#include <sabi_core/sabi_core.h>

using sabi::CgModelPtr;
using sabi::CgMaterial;
using sabi::CgTextureInfo;

class ShockerMaterialHandler;
class AreaLightHandler;
using ShockerMaterialHandlerPtr = std::shared_ptr<ShockerMaterialHandler>;
using AreaLightHandlerPtr = std::shared_ptr<AreaLightHandler>;

class ShockerMaterialHandler
{
public:
    // Factory method
    static ShockerMaterialHandlerPtr create()
    {
        return std::make_shared<ShockerMaterialHandler>();
    }
    
    ShockerMaterialHandler();
    ~ShockerMaterialHandler();
    
    // Initialize the material handler
    void initialize(RenderContext* ctx);
    
    // Set the area light handler for notifications
    void setAreaLightHandler(AreaLightHandlerPtr areaLightHandler) { 
        areaLightHandler_ = areaLightHandler; 
    }
    
    // Clear all materials
    void clear();
    
    // Process materials for a ShockerModel
    // Creates DisneyMaterials for each surface and assigns them to ShockerSurfaces
    void processMaterialsForModel(
        ShockerModel* model,
        const CgModelPtr& cgModel,
        const std::filesystem::path& materialFolder = {});
    
    // Create a DisneyMaterial from a CgMaterial
    DisneyMaterial* createMaterialFromCg(
        const CgMaterial& cgMaterial,
        const CgModelPtr& model = nullptr,
        const std::filesystem::path& materialFolder = {});
    
    // Assign a material to a ShockerSurface (new type-safe version)
    void assignMaterialToSurface(
        shocker::ShockerSurface* surface,
        DisneyMaterial* material);
    
   
    // Get all created materials
    const std::vector<std::unique_ptr<DisneyMaterial>>& getAllMaterials() const { 
        return materials_; 
    }
    
    // Get material by index
    DisneyMaterial* getMaterial(size_t index) {
        return index < materials_.size() ? materials_[index].get() : nullptr;
    }
    
    // Get material data buffer for GPU
    cudau::TypedBuffer<shared::DisneyData>* getMaterialDataBuffer() { 
        return &materialDataBuffer_; 
    }
    
    // Upload all material data to GPU
    void uploadMaterialsToGPU();
    
    // Material capacity
    static constexpr uint32_t MaxNumMaterials = 1024;
    static constexpr uint32_t InvalidSlotIndex = SlotFinder::InvalidSlotIndex;
    
private:
    // Convert specific material properties
    void convertBaseProperties(
        DisneyMaterial* disney,
        const CgMaterial::CoreProperties& core);
    
    void convertMetallicProperties(
        DisneyMaterial* disney,
        const CgMaterial::MetallicProperties& metallic);
    
    void convertSheenProperties(
        DisneyMaterial* disney,
        const CgMaterial::SheenProperties& sheen);
    
    void convertClearcoatProperties(
        DisneyMaterial* disney,
        const CgMaterial::ClearcoatProperties& clearcoat);
    
    void convertSubsurfaceProperties(
        DisneyMaterial* disney,
        const CgMaterial::SubsurfaceProperties& subsurface);
    
    void convertTransparencyProperties(
        DisneyMaterial* disney,
        const CgMaterial::TransparencyProperties& transparency);
    
    void convertEmissionProperties(
        DisneyMaterial* disney,
        const CgMaterial::EmissionProperties& emission);
    
    // Process texture information
    void processTextureInfo(
        const std::optional<CgTextureInfo>& texInfo,
        const cudau::Array** targetArray,
        CUtexObject* targetTexObject,
        const std::filesystem::path& materialFolder,
        const CgModelPtr& model,
        const std::string& requestedInput = "Color",
        const Eigen::Vector3f& defaultValue = Eigen::Vector3f::Zero());
    
    // Create default material
    DisneyMaterial* createDefaultMaterial();
    
    // Convert device data for GPU upload
    shared::DisneyData convertToDeviceData(const DisneyMaterial* hostMaterial);
    
    // Calculate texture dimension information
    shared::TexDimInfo calcDimInfo(
        const cudau::Array* cuArray,
        bool isLeftHanded = true);
    
private:
    // Render context (may be null for testing)
    RenderContext* ctx_ = nullptr;
    
    // Area light handler for notifications
    AreaLightHandlerPtr areaLightHandler_ = nullptr;
    
    // Collection of all created materials
    std::vector<std::unique_ptr<DisneyMaterial>> materials_;
    
    // Slot management for materials
    SlotFinder materialSlotFinder_;
    
    // Material data buffer for GPU
    cudau::TypedBuffer<shared::DisneyData> materialDataBuffer_;
    
    // Track initialization state
    bool isInitialized_ = false;
};