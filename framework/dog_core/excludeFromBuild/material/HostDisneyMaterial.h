
#pragma once

#include "../common/basic_types.h"

// Type Summary:
// cudau::Array* - CUDA array pointer type for GPU-accessible data storage. Represents
//                 arrays that can be accessed from both host and device code. These are
//                 typically used to store parameter values or lookup tables for materials.
// CUtexObject   - CUDA texture object handle (unsigned long long). References texture
//                 data on the GPU for efficient sampling in kernels. A value of 0 
//                 indicates no texture is bound. Used for mapping images to material
//                 properties with hardware-accelerated filtering and caching.

// Host-side Disney material representation for OptiX rendering that consolidates
// all Disney BRDF parameters and texture data into a single streamlined structure
struct DisneyMaterial
{
    // Core properties
    const cudau::Array* baseColor;
    CUtexObject texBaseColor;
    const cudau::Array* roughness;
    CUtexObject texRoughness;
    const cudau::Array* metallic;
    CUtexObject texMetallic;

    // Specular properties
    const cudau::Array* specular;
    CUtexObject texSpecular;
    const cudau::Array* anisotropic;
    CUtexObject texAnisotropic;
    const cudau::Array* anisotropicRotation;
    CUtexObject texAnisotropicRotation;

    // Sheen properties
    const cudau::Array* sheenColor;
    CUtexObject texSheenColor;
    const cudau::Array* sheenRoughness;
    CUtexObject texSheenRoughness;

    // Clearcoat properties
    const cudau::Array* clearcoat;
    CUtexObject texClearcoat;
    const cudau::Array* clearcoatGloss;
    CUtexObject texClearcoatGloss;
    const cudau::Array* clearcoatNormal;
    CUtexObject texClearcoatNormal;

    // Subsurface properties
    const cudau::Array* subsurface;
    CUtexObject texSubsurface;
    const cudau::Array* subsurfaceColor;
    CUtexObject texSubsurfaceColor;
    const cudau::Array* subsurfaceRadius;
    CUtexObject texSubsurfaceRadius;

    // Transmission properties
    const cudau::Array* translucency;
    CUtexObject texTranslucency;
    const cudau::Array* transparency;
    CUtexObject texTransparency;
    const cudau::Array* transmittance;
    CUtexObject texTransmittance;
    const cudau::Array* transmittanceDistance;
    CUtexObject texTransmittanceDistance;
    const cudau::Array* ior;
    CUtexObject texIOR;

     // Emissive properties
    const cudau::Array* emissive;
    CUtexObject texEmissive;
    const cudau::Array* emissiveStrength;
    CUtexObject texEmissiveStrength;

    // Normal mapping
    const cudau::Array* normal;
    CUtexObject texNormal;

       bool useAlphaForTransparency = false;

    // Default constructor initializes all pointers and handles to null/zero
    DisneyMaterial() :
        baseColor (nullptr), texBaseColor (0),
        roughness (nullptr), texRoughness (0),
        metallic (nullptr), texMetallic (0),
        specular (nullptr), texSpecular (0),
        anisotropic (nullptr), texAnisotropic (0),
        anisotropicRotation (nullptr), texAnisotropicRotation (0),
        sheenColor (nullptr), texSheenColor (0),
        sheenRoughness (nullptr), texSheenRoughness (0),
        clearcoat (nullptr), texClearcoat (0),
        clearcoatGloss (nullptr), texClearcoatGloss (0),
        clearcoatNormal (nullptr), texClearcoatNormal (0),
        subsurface (nullptr), texSubsurface (0),
        subsurfaceColor (nullptr), texSubsurfaceColor (0),
        subsurfaceRadius (nullptr), texSubsurfaceRadius (0),
        translucency (nullptr), texTranslucency (0),
        transparency (nullptr), texTransparency (0),
        transmittance (nullptr), texTransmittance (0),
        transmittanceDistance (nullptr), texTransmittanceDistance (0),
        ior (nullptr), texIOR (0),
        normal (nullptr), texNormal (0),
        emissive (nullptr), texEmissive (0),
        emissiveStrength (nullptr), texEmissiveStrength (0) {}
};
