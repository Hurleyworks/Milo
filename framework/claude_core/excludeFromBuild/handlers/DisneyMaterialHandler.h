#pragma once

// DisneyMaterialHandler.h
// Manages material creation, updating, and texture processing for the OptiX rendering system
// Implements physically-based materials and specialized visualization materials

#include "../RenderContext.h"
#include "../common/common_host.h"
#include "../material/HostDisneyMaterial.h"
#include "../material/DeviceDisneyMaterial.h"

using DisneyMaterialHandlerPtr = std::shared_ptr<class DisneyMaterialHandler>;
using sabi::CgModelPtr;
using sabi::CgTextureInfo;

class DisneyMaterialHandler
{
 public:
    // Factory method to create a new DisneyMaterialHandler instance
    static DisneyMaterialHandlerPtr create (RenderContextPtr ctx)
    {
        return std::make_shared<DisneyMaterialHandler> (ctx);
    }

    // Constructor initializes the handler with texture samplers and context
    DisneyMaterialHandler (RenderContextPtr ctx);

    // Destructor cleans up materials and texture resources
    ~DisneyMaterialHandler();

    // Initialize the material handler with buffer allocation
    void initialize();

    // Finalizes material setup before rendering begins
    void finalize();

    // Creates a Disney BRDF material with physically-based rendering properties
    // Implements the Disney Principled BRDF model with comprehensive parameter set
    // Returns a tuple of (Material, materialSlot) via std::tie
    std::tuple<optixu::Material, uint32_t> createDisneyMaterial (
        const CgMaterial& material,              // Source material data
        const std::filesystem::path& materialFolder, // Folder containing texture files
        CgModelPtr model);                       // Model this material is associated with

    // Updates an existing material with new properties
    // Allows dynamic modification of material parameters during rendering
    void updateMaterial (
        const Material& sabiMaterial,            // Updated material data
        optixu::Material& material,              // OptiX material to update
        const std::filesystem::path& materialFolder, // Folder containing texture files
        CgModelPtr model);                       // Model this material is associated with

    // Get the material data buffer for launch parameters
    cudau::TypedBuffer<shared::DisneyData>* getMaterialDataBuffer() { return &materialDataBuffer_; }

    // Material capacity constant
    static constexpr uint32_t maxNumMaterials = 1024;
    static constexpr uint32_t InvalidSlotIndex = SlotFinder::InvalidSlotIndex;

 private:
    RenderContextPtr ctx = nullptr;   // Render context for OptiX operations
  
    // Collection of all created materials for lifecycle management
    std::vector<std::unique_ptr<DisneyMaterial>> materials;

    // Slot management for materials
    SlotFinder materialSlotFinder_;
    
    // Material data buffer
    cudau::TypedBuffer<shared::DisneyData> materialDataBuffer_;

    // Track if initialized
    bool isInitialized_ = false;

    // Texture samplers for different color spaces and formats
    cudau::TextureSampler sampler_sRGB;      // For sRGB textures requiring gamma correction
    cudau::TextureSampler sampler_float;     // For raw float textures
    cudau::TextureSampler sampler_normFloat; // For normalized float textures

    // Calculates texture dimension information including power-of-two status
    shared::TexDimInfo calcDimInfo (
        const cudau::Array* cuArray,     // CUDA array containing texture data
        bool isLeftHanded = true);       // Coordinate system handedness

    // Creates base OptiX material with hit groups and programs configured
    optixu::Material createOptixMaterial();
    
    // Process texture information for a given texture type (albedo, normal, roughness, etc.)
    // Handles texture loading, CUDA texture object creation, and material parameter updates
    void processTextureInfo (
        const std::optional<CgTextureInfo>& texInfo,  // Texture information from content
        DisneyMaterial* hostDisney,                   // Host-side material to update
        const cudau::Array** targetArray,             // Target array to load texture into
        CUtexObject* targetTexObject,                 // Target texture object to create
        const std::filesystem::path& materialFolder,  // Folder containing texture files
        CgModelPtr model,                             // Model this texture is for
        const Eigen::Vector3f& defaultVector = Eigen::Vector3f::Zero()); // Default value if no texture
};