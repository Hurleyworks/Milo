// mostly taken from Shocker GfxExp
// https://github.com/shocker-0x15/GfxExp

#pragma once
#include "../common/basic_types.h"

namespace shared
{
    // Device-side Disney material data for OptiX rendering that consolidates
    // all Disney BRDF parameters and texture handles
    struct DisneyData
    {
        // Core properties
        CUtexObject baseColor;
        CUtexObject roughness;
        CUtexObject metallic;

        // Specular properties
        CUtexObject specular;
        CUtexObject anisotropic;
        CUtexObject anisotropicRotation;

        // Sheen properties
        CUtexObject sheenColor;
        CUtexObject sheenRoughness;

        // Clearcoat properties
        CUtexObject clearcoat;
        CUtexObject clearcoatGloss;
        CUtexObject clearcoatNormal;

        // Subsurface properties
        CUtexObject subsurface;
        CUtexObject subsurfaceColor;
        CUtexObject subsurfaceRadius;

        // Transmission properties
        CUtexObject translucency;
        CUtexObject transparency;
        CUtexObject transmittance;
        CUtexObject transmittanceDistance;
        CUtexObject ior;

        // Normal mapping
        CUtexObject normal;

          // Emissive properties
        CUtexObject emissive;
        CUtexObject emissiveStrength;

        // Texture dimension info for all textures
        TexDimInfo baseColor_dimInfo;
        TexDimInfo roughness_dimInfo;
        TexDimInfo metallic_dimInfo;
        TexDimInfo specular_dimInfo;
        TexDimInfo anisotropic_dimInfo;
        TexDimInfo anisotropicRotation_dimInfo;
        TexDimInfo sheenColor_dimInfo;
        TexDimInfo sheenRoughness_dimInfo;
        TexDimInfo clearcoat_dimInfo;
        TexDimInfo clearcoatGloss_dimInfo;
        TexDimInfo clearcoatNormal_dimInfo;
        TexDimInfo subsurface_dimInfo;
        TexDimInfo subsurfaceColor_dimInfo;
        TexDimInfo subsurfaceRadius_dimInfo;
        TexDimInfo translucency_dimInfo;
        TexDimInfo transparency_dimInfo;
        TexDimInfo transmittance_dimInfo;
        TexDimInfo transmittanceDistance_dimInfo;
        TexDimInfo ior_dimInfo;
        TexDimInfo normal_dimInfo;
        TexDimInfo emissive_dimInfo;
        TexDimInfo emissiveStrength_dimInfo;

        // Material flags
        uint32_t thinWalled : 1;
        uint32_t doubleSided : 1;
        // Flag for using alpha channel for transparency
        int useAlphaForTransparency;
    };

} // namespace shared
