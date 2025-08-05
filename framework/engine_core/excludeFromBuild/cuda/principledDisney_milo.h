#pragma once

#include "mx_core_milo.h"

class DisneyPrincipled
{
 private:
    RGB baseColor;
    float metallic;
    float roughness;
    float specularTint;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    float subsurface;
    float ior;
    Vector2D alpha;
    float transparency;
    float transmittance;
    float transmittanceDistance;
    bool thinWalled;
    RGB emissiveColor;
    float emissiveStrength;

 public:
    CUDA_DEVICE_FUNCTION static DisneyPrincipled create (
        const shared::DisneyData& matData,
        const Point2D& texCoord,
        float mipLevel,
        int allGlass,
        float globalIOR,
        float transmittanceDist,
        int globalGlasstype,
        shared::BSDFFlags flags = shared::BSDFFlags::None)
    {
        float4 baseColorValue = tex2DLod<float4> (matData.baseColor, texCoord.x, texCoord.y, mipLevel);
        float metallic = tex2DLod<float> (matData.metallic, texCoord.x, texCoord.y, mipLevel);
        float roughness = tex2DLod<float> (matData.roughness, texCoord.x, texCoord.y, mipLevel);
        float2 specAniso = tex2DLod<float2> (matData.anisotropic, texCoord.x, texCoord.y, mipLevel);
        float2 sheenValues = tex2DLod<float2> (matData.sheenColor, texCoord.x, texCoord.y, mipLevel);
        float2 clearcoatValues = tex2DLod<float2> (matData.clearcoatGloss, texCoord.x, texCoord.y, mipLevel);
        float2 subsurfaceSpecTrans = tex2DLod<float2> (matData.subsurfaceRadius, texCoord.x, texCoord.y, mipLevel);
        float iorValue = globalIOR;
        float transparency = tex2DLod<float> (matData.transparency, texCoord.x, texCoord.y, mipLevel);
        float transmittance = tex2DLod<float> (matData.transmittance, texCoord.x, texCoord.y, mipLevel);
        float transmittanceDistance = tex2DLod<float> (matData.transmittanceDistance, texCoord.x, texCoord.y, mipLevel);
        float4 emissiveValue = tex2DLod<float4> (matData.emissive, texCoord.x, texCoord.y, mipLevel);
        float emissiveStrength = tex2DLod<float> (matData.emissiveStrength, texCoord.x, texCoord.y, mipLevel);

        bool thinWalled = globalGlasstype ? 0 : 1;

        #if 0
        //  If this material uses alpha for transparency, factor it in
        if (matData.useAlphaForTransparency)
        {
            // Get alpha value from base color texture
            float alpha = baseColorValue.w;

            // Blend between the explicit transparency value and the alpha-driven transparency
            // Use a conservative approach to maintain compatibility
            float alphaWeight = 0.7f; // Can be adjusted as needed
            float originalTransparency = transparency;
            transparency = mix (originalTransparency, 1.0f - alpha, alphaWeight);
        }
        #endif
        // For binary alpha, a threshold approach works better than blending
        if (matData.useAlphaForTransparency)
        {
            // Get alpha value from base color texture
            float alpha = baseColorValue.w;

            // For images with binary transparency, use a threshold approach
            // If alpha is below threshold, treat as fully transparent
            float alphaThreshold = 0.5f;
            if (alpha < alphaThreshold)
            {
                transparency = 1.0f; // Fully transparent
            }
            else if (transparency < 0.1f)
            {
                // Only override if the material isn't already intentionally transparent
                transparency = 0.0f; // Fully opaque
            }
        }

        if (allGlass)
        {
            transparency = 1.0f;
            transmittance = 1.0f;
            transmittanceDistance = transmittanceDist;
            metallic = 0.0f;
            roughness = 0.001f;
        }

        return DisneyPrincipled (
            RGB (baseColorValue.x, baseColorValue.y, baseColorValue.z),
            metallic,
            roughness,
            specAniso.x,
            specAniso.y,
            sheenValues.x,
            sheenValues.y,
            clearcoatValues.x,
            clearcoatValues.y,
            subsurfaceSpecTrans.x,
            subsurfaceSpecTrans.y,
            iorValue,
            transparency,
            transmittance,
            transmittanceDistance,
            thinWalled,
            RGB (emissiveValue.x, emissiveValue.y, emissiveValue.z),
            emissiveStrength);
    }

    CUDA_DEVICE_FUNCTION DisneyPrincipled (
        const RGB& baseColor,
        float metallic,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearcoat,
        float clearcoatGloss,
        float subsurface,
        float specTrans,
        float ior,
        float transparency,
        float transmittance,
        float transmittanceDistance,
        bool thinWalled,
        const RGB& emissiveColor,
        float emissiveStrength) :
        baseColor (baseColor),
        metallic (metallic), roughness (clamp (roughness, 0.001f, 1.0f)), specularTint (specularTint), anisotropic (anisotropic), sheen (sheen), sheenTint (sheenTint), clearcoat (clearcoat), clearcoatGloss (clearcoatGloss), subsurface (subsurface), ior (ior), transparency (transparency), transmittance (transmittance), transmittanceDistance (transmittanceDistance), thinWalled (thinWalled), emissiveColor (emissiveColor), emissiveStrength (emissiveStrength)
    {
        mx_roughness_anisotropy (roughness, anisotropic, &alpha);
    }

    // Evaluates metallic BRDF for conductive surfaces
    CUDA_DEVICE_FUNCTION RGB evaluateMetallic (
        const Vector3D& vGiven,
        float uDir0,
        float uDir1,
        Vector3D* vSampled,
        float* dirPDensity) const
    {
        // Sample GGX VNDF for metals
        Vector3D H = mx_ggx_importance_sample_VNDF (
            Vector2D (uDir0, uDir1),
            vGiven,
            alpha);

        *vSampled = mx_reflect (-vGiven, H);

        // Skip if invalid reflection
        if (vSampled->z <= 0.0f)
        {
            *dirPDensity = 0.0f;
            return RGB (0.0f);
        }

        // For pure metals, use the base color directly as F0
        RGB F0 = baseColor;

        float NdotV = max (vGiven.z, 0.0f);
        float NdotL = max (vSampled->z, 0.0f);
        float HdotL = max (dot (H, *vSampled), 0.0f);
        float HdotV = max (dot (H, vGiven), 0.0f);

        // Enhanced Fresnel term for metals
        RGB F = mx_fresnel_schlick (HdotV, F0);

        // Strengthen edge reflections for metals
        float edgeFalloff = pow (1.0f - HdotV, 5.0f);
        F = F * (1.0f + edgeFalloff * 0.5f);

        // Calculate remaining GGX terms
        float avgAlpha = mx_average_alpha (alpha);
        float D = mx_ggx_NDF (H, alpha);
        float G = mx_ggx_smith_G2 (NdotL, NdotV, avgAlpha);

        // Energy compensation for multiple scattering
        RGB energyCompensation = mx_ggx_energy_compensation (NdotV, avgAlpha, F0);

        // Compute final BRDF value with boosted reflections
        RGB brdf = (F * D * G / (4.0f * NdotV * NdotL)) * energyCompensation;

        // Handle PDF
        float G1V = mx_ggx_smith_G1 (NdotV, avgAlpha);
        *dirPDensity = D * G1V / (4.0f * NdotV);

        return brdf;
    }

    CUDA_DEVICE_FUNCTION RGB evaluate (const Vector3D& vGiven, const Vector3D& vSampled) const
    {
        // Keep existing early exit for glass/transparent
        if (roughness < 0.002f && transparency > 0.9f)
        {
            return RGB (0.0f);
        }

        // Keep existing glass handling
        if (transparency > 0.0f && metallic <= 0.0f)
        {
            float NdotV = abs (vGiven.z);
            float NdotL = abs (vSampled.z);

            FresnelData fd = mx_init_fresnel_dielectric (
                ior,
                0.0f,
                1.0f);
            fd.refraction = true;

            if (thinWalled)
            {
                RGB T = baseColor;
                float F = mx_fresnel_dielectric (NdotV, ior);
                return T * RGB (1.0f - F) * transparency * transmittance;
            }
            else
            {
                bool isReflection = dot (vGiven, vSampled) > 0.0f;
                if (isReflection)
                {
                    Vector3D H = normalize (vGiven + vSampled);
                    float VdotH = max (dot (vGiven, H), 0.0f);
                    RGB F = mx_compute_fresnel (VdotH, fd);
                    float avgAlpha = mx_average_alpha (alpha);
                    float D = mx_ggx_NDF (H, alpha);
                    float G = mx_ggx_smith_G2 (NdotL, NdotV, avgAlpha);
                    return F * D * G * (1.0f - transparency) / (4.0f * NdotV * NdotL);
                }
                else
                {
                    float eta = 1.0f / ior;
                    Vector3D H = normalize (vGiven + vSampled * eta);
                    if (!isfinite (H.x)) return RGB (0.0f);

                    RGB T;
                    T.r = exp (-baseColor.r * transmittanceDistance);
                    T.g = exp (-baseColor.g * transmittanceDistance);
                    T.b = exp (-baseColor.b * transmittanceDistance);

                    RGB F = mx_compute_fresnel (abs (dot (vGiven, H)), fd);
                    float avgAlpha = mx_average_alpha (alpha);
                    float D = mx_ggx_NDF (H, alpha);
                    float G = mx_ggx_smith_G2 (NdotL, NdotV, avgAlpha);
                    return T * (RGB (1.0f) - F) * transparency * transmittance *
                           D * G / (4.0f * NdotV * NdotL * eta * eta);
                }
            }
        }

        Vector3D H = normalize (vGiven + vSampled);
        float NdotL = max (vSampled.z, 0.0f);
        float NdotV = max (vGiven.z, 0.0f);
        float VdotH = max (dot (vGiven, H), 0.0f);
        float NdotH = max (H.z, 0.0f);
        float LdotH = max (dot (vSampled, H), 0.0f);

        if (NdotL <= 0.0f || NdotV <= 0.0f)
            return RGB (0.0f);

        RGB F0;
        if (metallic > 0.0f)
        {
            // Keep existing metal F0 calculation
            F0 = mix (RGB (0.08f), baseColor, metallic);
        }
        else
        {
            // Physical dielectric F0 based on IOR
            float F0_dielectric = mx_ior_to_f0 (ior);
            F0_dielectric = max (F0_dielectric, 0.04f);
            F0 = RGB (F0_dielectric);
        }

        RGB F = mx_fresnel_schlick (VdotH, F0);

        float avgAlpha = mx_average_alpha (alpha);
        float D = mx_ggx_NDF (H, alpha);
        float G = mx_ggx_smith_G2 (NdotL, NdotV, avgAlpha);

        // Specular term
        RGB Fr = F * D * G / (4.0f * NdotV * NdotL);

        // Diffuse term
        RGB Fd = RGB (0.0f);
        if (metallic < 1.0f)
        {
            RGB diffuseAlbedo = baseColor * (1.0f - metallic);
            RGB diffuseAttenuation = RGB (1.0f) - F;

            float FL = mx_pow5 (1.0f - NdotL);
            float FV = mx_pow5 (1.0f - NdotV);
            float FD90 = 0.5f + 2.0f * LdotH * LdotH * avgAlpha;
            float FD = mix (1.0f, FD90, FL) * mix (1.0f, FD90, FV);

            Fd = diffuseAlbedo * FD * diffuseAttenuation * M_PI_INV;
        }

        return (Fd + Fr) * NdotL;
    }

    CUDA_DEVICE_FUNCTION RGB sampleThroughput (
        const Vector3D& vGiven,
        float uDir0,
        float uDir1,
        Vector3D* vSampled,
        float* dirPDensity) const
    {
        // Keep existing metal handling - it works well
        if (metallic > 0.0f)
        {
            return evaluateMetallic (vGiven, uDir0, uDir1, vSampled, dirPDensity);
        }

        if (transparency > 0.0f)
        {
            bool entering = vGiven.z >= 0.0f;
            const float eta = ior;
            const float eEnter = entering ? 1.0f : eta;
            const float eExit = entering ? eta : 1.0f;
            Vector3D dirV = entering ? vGiven : -vGiven;

            // Calculate Fresnel for front surface
            float cosTheta = std::abs (dirV.z);
            float F1 = fresnel (eEnter, eExit, cosTheta);

            if (thinWalled)
            {
                // Increase edge visibility by properly accumulating all reflection/transmission paths

                // First surface Fresnel
                float cosTheta = std::abs (dirV.z);
                float F1 = fresnel (eEnter, eExit, cosTheta);

                // Internal reflection calculation
                float sinTheta2 = 1.0f - cosTheta * cosTheta;
                float recRelIOR = eEnter / eExit;
                float sinInternal2 = recRelIOR * recRelIOR * sinTheta2;
                float cosInternal = std::sqrt (std::fmax (0.0f, 1.0f - sinInternal2));

                // Second surface Fresnel
                float F2 = fresnel (eExit, eEnter, cosInternal);

                // Account for multiple internal reflections
                float T12 = (1.0f - F1) * (1.0f - F2);
                float R12 = F1 + (T12 * F2); // Total reflection including internal bounce

                if (uDir0 < R12)
                { // Increased probability of reflection
                    // Perfect mirror reflection
                    *vSampled = Vector3D (-dirV.x, -dirV.y, dirV.z);
                    *dirPDensity = R12;

                    // Enhanced edge reflectivity
                    return RGB (R12 / std::fabs (cosTheta));
                }
                else
                {
                    *vSampled = -vGiven;
                    float T = T12 / (1.0f - R12); // Normalized transmission
                    *dirPDensity = 1.0f - R12;

                    RGB throughput = RGB (T) * transparency;
                    // Add subtle internal scattering for more realistic edges
                    if (cosTheta < 0.2f)
                    {                                    // At grazing angles
                        throughput *= (cosTheta * 5.0f); // Gradual falloff
                    }
                    return throughput;
                }
            }
            else
            {
                // Original volumetric glass handling...
                if (uDir0 < F1)
                {
                    // Handle reflection
                    if (dirV.z == 0.0f)
                    {
                        *dirPDensity = 0.0f;
                        return RGB (0.0f);
                    }
                    *vSampled = Vector3D (-dirV.x, -dirV.y, dirV.z);
                    *dirPDensity = F1;
                    return RGB (F1 * (1.0f / std::fabs (dirV.z)));
                }
                else
                {
                    // Handle refraction with Beer's law
                    float sinEnter2 = 1.0f - dirV.z * dirV.z;
                    float recRelIOR = eEnter / eExit;
                    float sinExit2 = recRelIOR * recRelIOR * sinEnter2;

                    // Check for total internal reflection
                    if (sinExit2 >= 1.0f)
                    {
                        *dirPDensity = 0.0f;
                        return RGB (0.0f);
                    }

                    float cosExit = std::sqrt (std::fmax (0.0f, 1.0f - sinExit2));
                    *vSampled = Vector3D (recRelIOR * -dirV.x, recRelIOR * -dirV.y, -cosExit);
                    *vSampled = entering ? *vSampled : -*vSampled;

                    float T = 1.0f - F1;
                    *dirPDensity = T;
                    RGB ret = RGB (T) * transparency;

                    // Apply Beer's law absorption for thick glass
                    float dist = transmittanceDistance;
                    ret.r *= exp (-baseColor.r * dist);
                    ret.g *= exp (-baseColor.g * dist);
                    ret.b *= exp (-baseColor.b * dist);

                    float squeezeFactor = pow2 (eEnter / eExit);
                    ret *= squeezeFactor / std::fabs (cosExit);
                    *dirPDensity *= squeezeFactor;

                    return ret;
                }
            }
        }

        // Choose between specular and diffuse sampling based on metalness
        float specularProb = lerp (0.5f, 1.0f, metallic);

        Vector3D H;
        if (uDir0 < specularProb)
        {
            // Adjust random number for conditional sampling
            float uAdjusted = uDir0 / specularProb;
            // Importance sample GGX
            H = mx_ggx_importance_sample_VNDF (
                Vector2D (uAdjusted, uDir1),
                vGiven,
                alpha);
            *vSampled = mx_reflect (-vGiven, H);
        }
        else
        {
            // Cosine weighted diffuse sampling
            float uAdjusted = (uDir0 - specularProb) / (1.0f - specularProb);
            *vSampled = mx_cosine_sample_hemisphere (uAdjusted, uDir1);
        }

        float NdotV = max (vGiven.z, 0.0f);
        float G1V = mx_ggx_smith_G1 (NdotV, mx_average_alpha (alpha));
        *dirPDensity = mx_ggx_NDF (H, alpha) * G1V / (4.0f * NdotV);

        return evaluate (vGiven, *vSampled);
    }

    CUDA_DEVICE_FUNCTION float evaluatePDF (
        const Vector3D& vGiven,
        const Vector3D& vSampled) const
    {
        if (vSampled.z <= 0.0f)
            return 0.0f;

        Vector3D H = normalize (vGiven + vSampled);
        float NdotV = max (vGiven.z, 0.0f);
        float G1V = mx_ggx_smith_G1 (NdotV, mx_average_alpha (alpha));

        return mx_ggx_NDF (H, alpha) * G1V / (4.0f * NdotV);
    }

    // Evaluates the emissive contribution of the material
    CUDA_DEVICE_FUNCTION RGB evaluateEmission() const
    {
        return emissiveColor * emissiveStrength;
    }

    CUDA_DEVICE_FUNCTION RGB evaluateDHReflectanceEstimate (
        const Vector3D& vGiven) const
    {
        float NdotV = max (vGiven.z, 0.0f);

        // Calculate base F0 value
        RGB F0;
        if (metallic > 0.0f)
        {
            F0 = mix (RGB (0.08f), baseColor, metallic);
        }
        else
        {
            float F0_dielectric = mx_ior_to_f0 (ior);
            F0 = RGB (F0_dielectric);
        }

        // Calculate fresnel
        RGB F = mx_fresnel_schlick (NdotV, F0);
        RGB one (1.0f);

        // Return mix between diffuse and specular based on metallic parameter
        return mix (
            baseColor * (one - F), // Diffuse component
            F,                              // Specular component
            metallic               // Mix factor
        );
    }
};

