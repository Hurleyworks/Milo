#pragma once

// This header contains core mathematical functions and utilities for material and lighting calculations
// Based on MaterialX specifications from the Academy Software Foundation
// The functions here form the foundation for physically-based rendering calculations

#include "../ripr_shared.h"
#include "../../../material/DeviceDisneyMaterial.h"
#include "../../../common/deviceCommon.h"

using namespace ripr_shared;
using namespace shared;

// Mathematical constants used throughout the rendering calculations
#define M_PI 3.1415926535897932f // Pi - ratio of circle's circumference to diameter
#define M_PI_INV (1.0f / M_PI)   // Inverse of Pi - used for normalizing values
#define M_FLOAT_EPS 1e-5f        // Small epsilon value to prevent division by zero

// Different models for calculating Fresnel effects (how light reflects off surfaces)
#define FRESNEL_MODEL_DIELECTRIC 0 // For non-metals (glass, water, etc.)
#define FRESNEL_MODEL_CONDUCTOR 1  // For metals (gold, silver, etc.)
#define FRESNEL_MODEL_SCHLICK 2    // Schlick's approximation - faster but less accurate

// Stores data needed for Fresnel calculations
struct FresnelData
{
    int model; // Which Fresnel model to use
    bool airy; // Whether to use Airy model for thin film interference

    RGB ior;        // Index of refraction - how much light bends in material
    RGB extinction; // How quickly light is absorbed in the material

    RGB F0;    // Fresnel reflectance at normal incidence (0 degrees)
    RGB F82;   // Fresnel reflectance at 82 degrees
    RGB F90;   // Fresnel reflectance at 90 degrees (grazing angle)
    float exponent; // Custom exponent for Fresnel calculations

    float tf_thickness; // Thickness of thin film layer (if present)
    float tf_ior;       // Index of refraction of thin film layer

    bool refraction; // Whether to calculate refraction
};

// Basic math utility functions
// Returns the square of a number
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_square (float x)
{
    return x * x;
}

// Returns x raised to the 5th power - used in Fresnel calculations
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_pow5 (float x)
{
    float x2 = mx_square (x);
    return x2 * x2 * x;
}

// Returns x raised to the 6th power
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_pow6 (float x)
{
    float x2 = mx_square (x);
    return x2 * x2 * x2;
}

// Core helper functions

// Linearly interpolates between x and y based on a
// When a is 0, returns x. When a is 1, returns y
CUDA_DEVICE_FUNCTION CUDA_INLINE float mix (float x, float y, float a)
{
    return x * (1.0f - a) + y * a;
}

// Vector version of mix function for RGB colors
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB mix (const RGB& x, const RGB& y, float a)
{
    return x * (1.0f - a) + y * a;
}

// Clamps a value between minimum and maximum bounds
CUDA_DEVICE_FUNCTION CUDA_INLINE float clamp (float x, float minVal, float maxVal)
{
    return min (max (x, minVal), maxVal);
}

// Calculates perfect reflection direction using incident vector I and surface normal N
// Used for mirror-like reflections
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D mx_reflect (const Vector3D& I, const Vector3D& N)
{
    return I - 2.0f * dot (N, I) * N;
}

// Calculates refraction direction using Snell's law
// I is incident vector, N is surface normal, eta is ratio of refractive indices
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D refract (const Vector3D& I, const Vector3D& N, float eta)
{
    float dotNI = dot (N, I);
    float k = 1.0f - eta * eta * (1.0f - dotNI * dotNI);
    if (k < 0.0f)
        return Vector3D (0.0f); // Total internal reflection case
    return eta * I - (eta * dotNI + sqrt (k)) * N;
}

// Schlick's approximation for Fresnel reflectance
// Simpler and faster than full Fresnel equations
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_fresnel_schlick (float cosTheta, float F0)
{
    float x = clamp (1.0f - cosTheta, 0.0f, 1.0f);
    float x5 = mx_pow5 (x);
    return F0 + (1.0f - F0) * x5;
}

// RGB color version of Schlick's Fresnel approximation
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB mx_fresnel_schlick (float cosTheta, const RGB& F0)
{
    float x = clamp (1.0f - cosTheta, 0.0f, 1.0f);
    float x5 = mx_pow5 (x);
    return F0 + (RGB (1.0f) - F0) * x5;
}

// Extended Schlick's Fresnel with custom F90 value
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_fresnel_schlick (float cosTheta, float F0, float F90)
{
    float x = clamp (1.0f - cosTheta, 0.0f, 1.0f);
    float x5 = mx_pow5 (x);
    return mix (F0, F90, x5);
}

// RGB color version of extended Schlick's Fresnel
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB mx_fresnel_schlick (float cosTheta, const RGB& F0, const RGB& F90)
{
    float x = clamp (1.0f - cosTheta, 0.0f, 1.0f);
    float x5 = mx_pow5 (x);
    return mix (F0, F90, x5);
}

// Custom exponent version of Schlick's Fresnel
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_fresnel_schlick (float cosTheta, float F0, float F90, float exponent)
{
    float x = clamp (1.0f - cosTheta, 0.0f, 1.0f);
    return mix (F0, F90, pow (x, exponent));
}

// Calculates refraction through a solid sphere
// Used for special effects like glass balls
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D mx_refraction_solid_sphere (
    const Vector3D& R,
    const Vector3D& N,
    float ior)
{
    // First refraction when entering sphere
    Vector3D R1 = refract (R, N, 1.0f / ior);

    // Calculate exit normal and second refraction when leaving sphere
    Vector3D N1 = normalize (R1 * dot (R1, N) - N * 0.5f);
    return refract (R1, N1, ior);
}

// Base refraction function using Snell's law
// I is incident vector, N is surface normal, eta is ratio of refractive indices
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D mx_refract (
    const Vector3D& I,
    const Vector3D& N,
    float eta)
{
    float cosi = dot (-I, N);                                       // Cosine of incident angle
    float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);          // Cosine^2 of transmitted angle
    Vector3D t = eta * I + ((eta * cosi - sqrt (abs (cost2))) * N); // Refracted direction
    return cost2 > 0.0f ? t : Vector3D (0.0f);                      // Return 0 if total internal reflection
}

// Samples a point on a disk using concentric mapping
// This creates a more uniform distribution compared to naive mapping
// Used for lens sampling and other effects requiring circular distributions
CUDA_DEVICE_FUNCTION CUDA_INLINE void mx_concentric_sample_disk (float u1, float u2, float* dx, float* dy)
{
    float r, theta;
    // Transform from [0,1]^2 to [-1,1]^2
    float sx = 2.0f * u1 - 1.0f;
    float sy = 2.0f * u2 - 1.0f;

    // Handle degenerate case at origin
    if (sx == 0.0f && sy == 0.0f)
    {
        *dx = 0.0f;
        *dy = 0.0f;
        return;
    }

    // Map square to circle using concentric mapping
    if (sx >= -sy)
    {
        if (sx > sy)
        {
            r = sx;
            theta = (sy / sx);
        }
        else
        {
            r = sy;
            theta = 2.0f - (sx / sy);
        }
    }
    else
    {
        if (sx <= sy)
        {
            r = -sx;
            theta = 4.0f + (sy / sx);
        }
        else
        {
            r = -sy;
            theta = 6.0f - (sx / sy);
        }
    }

    // Convert to polar coordinates
    theta *= M_PI / 4.0f;
    *dx = r * cosf (theta);
    *dy = r * sinf (theta);
}

// Generates points on a hemisphere with cosine-weighted distribution
// This importance sampling matches the cosine term in diffuse reflection
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D mx_cosine_sample_hemisphere (float u1, float u2)
{
    float x, y;
    mx_concentric_sample_disk (u1, u2, &x, &y);
    float z = sqrtf (max (0.0f, 1.0f - x * x - y * y));
    return Vector3D (x, y, z);
}

// Calculates anisotropic roughness values from isotropic roughness and anisotropy
// Used in GGX/Cook-Torrance specular reflection models
CUDA_DEVICE_FUNCTION CUDA_INLINE void mx_roughness_anisotropy (
    float roughness, float anisotropy, Vector2D* outAlpha)
{
    float roughness_sqr = clamp (roughness * roughness, M_FLOAT_EPS, 1.0f);
    if (anisotropy > 0.0f)
    {
        // Calculate different roughness values for X and Y directions
        float aspect = sqrt (1.0f - clamp (anisotropy, 0.0f, 0.98f));
        outAlpha->x = min (roughness_sqr / aspect, 1.0f);
        outAlpha->y = roughness_sqr * aspect;
    }
    else
    {
        // Isotropic case - same roughness in all directions
        outAlpha->x = roughness_sqr;
        outAlpha->y = roughness_sqr;
    }
}

// Implements dielectric Fresnel equations for non-metals
// Returns reflectance at a given angle for a specific index of refraction
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_fresnel_dielectric (float cosTheta, float ior)
{
    float c = cosTheta;
    float g2 = ior * ior + c * c - 1.0f;
    if (g2 < 0.0f)
        return 1.0f; // Total internal reflection

    float g = sqrt (g2);
    // Average polarization states for unpolarized light
    return 0.5f * mx_square ((g - c) / (g + c)) *
           (1.0f + mx_square (((g + c) * c - 1.0f) / ((g - c) * c + 1.0f)));
}

// GGX Normal Distribution Function (NDF)
// Models the statistical distribution of microfacet normals
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_ggx_NDF (const Vector3D& H, const Vector2D& alpha)
{
    float alphax2 = alpha.x * alpha.x;
    float alphay2 = alpha.y * alpha.y;
    float denom = H.x * H.x / alphax2 + H.y * H.y / alphay2 + H.z * H.z;

    // Regularization for very smooth surfaces
    float eps = 1e-7f;
    denom = max (denom, eps);

    return 1.0f / (M_PI * alpha.x * alpha.y * denom * denom);
}

// Smith G1 term for GGX - models shadowing/masking for a single direction
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_ggx_smith_G1 (float cosTheta, float alpha)
{
    float cosTheta2 = mx_square (cosTheta);
    float tanTheta2 = (1.0f - cosTheta2) / cosTheta2;
    return 2.0f / (1.0f + sqrt (1.0f + mx_square (alpha) * tanTheta2));
}

// Complete Smith term for GGX - combines shadowing/masking for both view and light
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_ggx_smith_G2 (float NdotL, float NdotV, float alpha)
{
    float alpha2 = mx_square (alpha);
    float lambdaL = sqrt (alpha2 + (1.0f - alpha2) * mx_square (NdotL));
    float lambdaV = sqrt (alpha2 + (1.0f - alpha2) * mx_square (NdotV));
    return 2.0f / (lambdaL / NdotL + lambdaV / NdotV);
}

// Calculates average alpha value for isotropic approximations
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_average_alpha (const Vector2D& alpha)
{
    return sqrt (alpha.x * alpha.y);
}

// Converts Index of Refraction (IOR) to F0 (base reflectivity at normal incidence)
CUDA_DEVICE_FUNCTION CUDA_INLINE float mx_ior_to_f0 (float ior)
{
    return mx_square ((ior - 1.0f) / (ior + 1.0f));
}

// Samples GGX Distribution using Visible Normal Distribution Function (VNDF)
// This produces less noisy results compared to regular NDF sampling
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D mx_ggx_importance_sample_VNDF (
    const Vector2D& Xi,
    const Vector3D& V,
    const Vector2D& alpha)
{
    // Transform view direction to match roughness
    Vector3D Vh = normalize (Vector3D (alpha.x * V.x, alpha.y * V.y, V.z));

    // Construct orthonormal basis
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    Vector3D T1 = lensq > 0.0f ? Vector3D (-Vh.y, Vh.x, 0.0f) * rsqrt (lensq) : Vector3D (1.0f, 0.0f, 0.0f);
    Vector3D T2 = cross (Vh, T1);

    // Sample point with polar coordinates (r, phi)
    float r = sqrt (Xi.x);
    float phi = 2.0f * M_PI * Xi.y;
    float t1 = r * cos (phi);
    float t2 = r * sin (phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt (1.0f - t1 * t1) + s * t2;

    // Reproject onto hemisphere
    Vector3D Nh = t1 * T1 + t2 * T2 + sqrt (max (0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // Transform back to original space
    return normalize (Vector3D (alpha.x * Nh.x, alpha.y * Nh.y, max (0.0f, Nh.z)));
}

// Calculates directional albedo for GGX BRDF
// Used for energy conservation in specular reflections
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB mx_ggx_dir_albedo (
    float NdotV,
    float alpha,
    const RGB& F0,
    const RGB& F90)
{
    float x = NdotV;
    float y = alpha;
    float x2 = mx_square (x);
    float y2 = mx_square (y);

    // Fitted polynomial approximation
    float rx = 0.1003f + (-0.6303f * x) + (9.748f * y) +
               (-2.038f * x * y) + (29.34f * x2) + (-8.245f * y2);

    float ry = 0.9345f + (-2.323f * x) + (2.229f * y) +
               (-3.748f * x * y) + (1.424f * x2) + (-0.7684f * y2);

    return F0 * rx + F90 * ry;
}

// Calculates energy compensation factor for multiple scattering in GGX BRDF
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB mx_ggx_energy_compensation (
    float NdotV,
    float alpha,
    const RGB& F0)
{
    float Eavg = 0.0f;
    if (alpha > 0.0f)
    {
        float E = min (exp2 (-10.23f * NdotV * alpha) - exp2 (-10.23f * alpha), 1.0f);
        Eavg = 1.0f - E;
    }
    return RGB (1.0f) + F0 * (1.0f / Eavg - 1.0f);
}

// Computes Fresnel factor for a given angle and material properties
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB mx_compute_fresnel (
    float cosTheta,
    const FresnelData& fd)
{
    if (fd.model == FRESNEL_MODEL_DIELECTRIC)
    {
        // For non-metals (dielectrics), use scalar Fresnel
        float F = mx_fresnel_dielectric (cosTheta, fd.ior.r);
        return RGB (F);
    }

    // For other models, use Schlick's approximation
    float x = 1.0f - cosTheta;
    float x5 = mx_pow5 (x);
    return mix (fd.F0, fd.F90, x5);
}

// Initialize FresnelData for dielectric materials (non-metals)
CUDA_DEVICE_FUNCTION CUDA_INLINE FresnelData mx_init_fresnel_dielectric (
    float ior,
    float tf_thickness,
    float tf_ior)
{
    FresnelData fd;
    fd.model = FRESNEL_MODEL_DIELECTRIC;
    fd.airy = tf_thickness > 0.0f;
    fd.ior = RGB (ior);
    fd.extinction = RGB (0.0f);
    fd.F0 = RGB (0.0f);
    fd.F82 = RGB (0.0f);
    fd.F90 = RGB (0.0f);
    fd.exponent = 0.0f;
    fd.tf_thickness = tf_thickness;
    fd.tf_ior = tf_ior;
    fd.refraction = false;
    return fd;
}

// Initialize FresnelData for custom Schlick approximation
CUDA_DEVICE_FUNCTION CUDA_INLINE FresnelData mx_init_fresnel_schlick (
    const RGB& F0,
    const RGB& F82,
    const RGB& F90,
    float exponent,
    float tf_thickness,
    float tf_ior)
{
    FresnelData fd;
    fd.model = FRESNEL_MODEL_SCHLICK;
    fd.airy = tf_thickness > 0.0f;
    fd.ior = RGB (0.0f);
    fd.extinction = RGB (0.0f);
    fd.F0 = F0;
    fd.F82 = F82;
    fd.F90 = F90;
    fd.exponent = exponent;
    fd.tf_thickness = tf_thickness;
    fd.tf_ior = tf_ior;
    fd.refraction = false;
    return fd;
}

// not materialX code
// Basic vector operations
CUDA_DEVICE_FUNCTION CUDA_INLINE float3 operator- (const float3& a)
{
    return make_float3 (-a.x, -a.y, -a.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 operator+ (const float3& a, const float3& b)
{
    return make_float3 (a.x + b.x, a.y + b.y, a.z + b.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 operator- (const float3& a, const float3& b)
{
    return make_float3 (a.x - b.x, a.y - b.y, a.z - b.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 operator* (const float3& a, const float3& b)
{
    return make_float3 (a.x * b.x, a.y * b.y, a.z * b.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 operator* (const float3& a, float b)
{
    return make_float3 (a.x * b, a.y * b, a.z * b);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 operator* (float a, const float3& b)
{
    return b * a;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 schlickFresnel (const float3& f0Reflectance, float cos)
{
    return f0Reflectance + (make_float3 (1.0f) - f0Reflectance) * pow5 (1 - cos);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float fresnel (float etaEnter, float etaExit, float cosEnter)
{
    float sinExit = etaEnter / etaExit * std::sqrt (std::fmax (0.0f, 1.0f - cosEnter * cosEnter));
    if (sinExit >= 1.0f)
    {
        return 1.0f;
    }
    else
    {
        float cosExit = std::sqrt (std::fmax (0.0f, 1.0f - sinExit * sinExit));
        float Rparl = ((etaExit * cosEnter) - (etaEnter * cosExit)) / ((etaExit * cosEnter) + (etaEnter * cosExit));
        float Rperp = ((etaEnter * cosEnter) - (etaExit * cosExit)) / ((etaEnter * cosEnter) + (etaExit * cosExit));
        return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
    }
}