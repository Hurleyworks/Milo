#pragma once

#include "common_shared.h"

// Mathematical constants
static constexpr float Pi = 3.14159265358979323846f; // Pi value for circular/spherical calculations
static constexpr float RayEpsilon = 1e-4;            // Small value to prevent self-intersection of rays




// Helpful macros for vector printing
#define V2FMT "%g, %g"
#define V3FMT "%g, %g, %g"
#ifndef V4FMT
#define V4FMT "%g, %g, %g"
#endif
#define v2print(v) (v).x, (v).y
#define v3print(v) (v).x, (v).y, (v).z
#define v4print(v) (v).x, (v).y, (v).z, (v).w

// Converts spherical coordinates (phi, theta) to Cartesian (x,y,z)
// phi: angle in x-z plane (0 to 2π)
// theta: angle from y axis (0 to π)
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D fromPolarYUp (float phi, float theta)
{
    float sinPhi, cosPhi;
    float sinTheta, cosTheta;
    sincosf (phi, &sinPhi, &cosPhi);
    sincosf (theta, &sinTheta, &cosTheta);
    return Vector3D (-sinPhi * sinTheta, cosTheta, cosPhi * sinTheta);
}

// Converts Cartesian coordinates (x,y,z) to spherical (phi, theta)
CUDA_DEVICE_FUNCTION CUDA_INLINE void toPolarYUp (const Vector3D& v, float* phi, float* theta)
{
    *theta = std::acos (min (max (v.y, -1.0f), 1.0f));
    *phi = std::fmod (std::atan2 (-v.x, v.z) + 2 * Pi, 2 * Pi);
}

// Calculates the halfway vector between two vectors (used in specular reflection)
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D halfVector (const Vector3D& a, const Vector3D& b)
{
    return normalize (a + b);
}

// Encoding/decoding functions for compact storage
CUDA_DEVICE_FUNCTION CUDA_INLINE uint16_t encodeBarycentric(float bc) {
    return static_cast<uint16_t>(min(static_cast<uint32_t>(bc * 65535u), 65535u));
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float decodeBarycentric(uint16_t qbc) {
    return qbc / 65535.0f;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t encodeVector(const Vector3D &v) {
    float phi, theta;
    toPolarYUp(v, &phi, &theta);
    const uint32_t qPhi = min(static_cast<uint32_t>((phi / (2 * Pi)) * 65535u), 65535u);
    const uint32_t qTheta = min(static_cast<uint32_t>((theta / Pi) * 65535u), 65535u);
    return (qTheta << 16) | qPhi;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D decodeVector(uint32_t qv) {
    const uint32_t qPhi = qv & 0xFFFF;
    const uint32_t qTheta = qv >> 16;
    const float phi = 2 * Pi * (qPhi / 65535.0f);
    const float theta = Pi * (qTheta / 65535.0f);
    return fromPolarYUp(phi, theta);
}

// Returns absolute value of dot product between two vectors
template <bool isNormalA, bool isNormalB>
CUDA_DEVICE_FUNCTION CUDA_INLINE float absDot (
    const Vector3D_T<float, isNormalA>& a, const Vector3D_T<float, isNormalB>& b)
{
    return std::fabs (dot (a, b));
}

// Creates a coordinate system (tangent space) from a surface normal
CUDA_DEVICE_FUNCTION CUDA_INLINE void makeCoordinateSystem (
    const Normal3D& normal, Vector3D* tangent, Vector3D* bitangent)
{
    float sign = normal.z >= 0 ? 1 : -1;
    const float a = -1 / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    *tangent = Vector3D (1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    *bitangent = Vector3D (b, sign + normal.y * normal.y * a, -normal.y);
}

// Simple ray origin offset to avoid self-intersection
CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D offsetRayOriginNaive (
    const Point3D& p, const Normal3D& geometricNormal)
{
    return p + RayEpsilon * geometricNormal;
}

// Advanced ray origin offset to avoid self-intersection
// Uses both floating-point and integer math for better precision
CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D offsetRayOrigin (
    const Point3D& p, const Normal3D& geometricNormal)
{
    constexpr float kOrigin = 1.0f / 32.0f;        // Threshold for near-origin handling
    constexpr float kFloatScale = 1.0f / 65536.0f; // Scale for floating-point offset
    constexpr float kIntScale = 256.0f;            // Scale for integer offset

    // Convert normal to integer offset
    int32_t offsetInInt[] = {
        static_cast<int32_t> (kIntScale * geometricNormal.x),
        static_cast<int32_t> (kIntScale * geometricNormal.y),
        static_cast<int32_t> (kIntScale * geometricNormal.z)};

    // Apply offset using integer math for points far from origin
    Point3D newP1 (__int_as_float (__float_as_int (p.x) + (p.x < 0 ? -1 : 1) * offsetInInt[0]),
                   __int_as_float (__float_as_int (p.y) + (p.y < 0 ? -1 : 1) * offsetInInt[1]),
                   __int_as_float (__float_as_int (p.z) + (p.z < 0 ? -1 : 1) * offsetInInt[2]));

    // Apply offset using floating-point math for points near origin
    Point3D newP2 = p + kFloatScale * geometricNormal;

    // Choose appropriate offset based on distance from origin
    return Point3D (std::fabs (p.x) < kOrigin ? newP2.x : newP1.x,
                    std::fabs (p.y) < kOrigin ? newP2.y : newP1.y,
                    std::fabs (p.z) < kOrigin ? newP2.z : newP1.z);
}

// Reference frame for surface shading calculations
struct ReferenceFrame
{
    Vector3D tangent;   // Tangent vector (along surface)
    Vector3D bitangent; // Bitangent vector (perpendicular to tangent and normal)
    Normal3D normal;    // Surface normal

    CUDA_DEVICE_FUNCTION ReferenceFrame() {}

    // Create frame from complete orthonormal basis
    CUDA_DEVICE_FUNCTION ReferenceFrame (
        const Vector3D& _tangent, const Vector3D& _bitangent, const Normal3D& _normal) :
        tangent (_tangent),
        bitangent (_bitangent), normal (_normal) {}

    // Create frame from normal (generates tangent and bitangent)
    CUDA_DEVICE_FUNCTION ReferenceFrame (const Normal3D& _normal) :
        normal (_normal)
    {
        makeCoordinateSystem (normal, &tangent, &bitangent);
    }

    // Create frame from normal and tangent (generates bitangent)
    CUDA_DEVICE_FUNCTION ReferenceFrame (const Normal3D& _normal, const Vector3D& _tangent) :
        tangent (_tangent), normal (_normal)
    {
        bitangent = cross (normal, tangent);
    }

    // Convert vector from world space to local frame
    CUDA_DEVICE_FUNCTION Vector3D toLocal (const Vector3D& v) const
    {
        return Vector3D (dot (tangent, v), dot (bitangent, v), dot (normal, v));
    }

    // Convert vector from local frame to world space
    CUDA_DEVICE_FUNCTION Vector3D fromLocal (const Vector3D& v) const
    {
        return Vector3D (dot (Vector3D (tangent.x, bitangent.x, normal.x), v),
                         dot (Vector3D (tangent.y, bitangent.y, normal.y), v),
                         dot (Vector3D (tangent.z, bitangent.z, normal.z), v));
    }
};

// Applies bump mapping to modify surface normal
// Bump mapping creates the illusion of surface detail without changing geometry
CUDA_DEVICE_FUNCTION CUDA_INLINE void applyBumpMapping (
    const Normal3D& modNormalInTF, ReferenceFrame* frameToModify)
{
    // Calculate length of normal's projection on tangent plane
    float projLength = std::sqrt (modNormalInTF.x * modNormalInTF.x + modNormalInTF.y * modNormalInTF.y);

    // Skip if projection is too small (normal is nearly vertical)
    if (projLength < 1e-3f)
        return;

    // Calculate rotation angle and quaternion components
    float tiltAngle = std::atan (projLength / modNormalInTF.z);
    float qSin, qCos;
    sincosf (tiltAngle / 2, &qSin, &qCos);
    float qX = (-modNormalInTF.y / projLength) * qSin;
    float qY = (modNormalInTF.x / projLength) * qSin;
    float qW = qCos;

    // Calculate modified tangent vectors using quaternion rotation
    Vector3D modTangentInTF (1 - 2 * qY * qY, 2 * qX * qY, -2 * qY * qW);
    Vector3D modBitangentInTF (2 * qX * qY, 1 - 2 * qX * qX, 2 * qX * qW);

    // Transform from tangent space to world space
    Matrix3x3 matTFtoW (
        frameToModify->tangent,
        frameToModify->bitangent,
        Vector3D (frameToModify->normal));

    // Create new shading frame with modified vectors
    ReferenceFrame bumpShadingFrame (
        matTFtoW * modTangentInTF,
        matTFtoW * modBitangentInTF,
        matTFtoW * modNormalInTF);

    *frameToModify = bumpShadingFrame;
}

// Maps uniform random points to a disk using concentric mapping
// This creates a more uniform distribution than naive mapping methods
CUDA_DEVICE_FUNCTION CUDA_INLINE void concentricSampleDisk (float u0, float u1, float* dx, float* dy)
{
    float r, theta;
    // Transform from [0,1]² to [-1,1]²
    float sx = 2 * u0 - 1;
    float sy = 2 * u1 - 1;

    // Handle degenerate case at origin
    if (sx == 0 && sy == 0)
    {
        *dx = 0;
        *dy = 0;
        return;
    }

    // Map square to disk using concentric mapping
    if (sx >= -sy)
    { // Region 1 or 2
        if (sx > sy)
        { // Region 1
            r = sx;
            theta = sy / sx;
        }
        else
        { // Region 2
            r = sy;
            theta = 2 - sx / sy;
        }
    }
    else
    { // Region 3 or 4
        if (sx > sy)
        { // Region 4
            r = -sy;
            theta = 6 + sx / sy;
        }
        else
        { // Region 3
            r = -sx;
            theta = 4 + sy / sx;
        }
    }
    // Convert to polar coordinates and scale
    theta *= Pi / 4;
    *dx = r * cos (theta);
    *dy = r * sin (theta);
}

// Generates points on a hemisphere with cosine-weighted distribution
// Used for importance sampling in diffuse reflection calculations
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D cosineSampleHemisphere (float u0, float u1)
{
    float x, y;
    concentricSampleDisk (u0, u1, &x, &y);
    // Project disk point onto hemisphere
    return Vector3D (x, y, std::sqrt (std::fmax (0.0f, 1.0f - x * x - y * y)));
}

// Texture coordinate adjustment for special texture formats
CUDA_DEVICE_FUNCTION CUDA_INLINE Point2D adjustTexCoord (
    shared::TexDimInfo dimInfo, const Point2D& texCoord)
{
    Point2D mTexCoord = texCoord;
    // Adjust coordinates for non-power-of-two block compressed textures
    if (dimInfo.isNonPowerOfTwo && dimInfo.isBCTexture)
    {
        uint32_t bcWidth = (dimInfo.dimX + 3) / 4 * 4;
        uint32_t bcHeight = (dimInfo.dimY + 3) / 4 * 4;
        mTexCoord.x *= static_cast<float> (dimInfo.dimX) / bcWidth;
        mTexCoord.y *= static_cast<float> (dimInfo.dimY) / bcHeight;
    }
    return mTexCoord;
}

// Generic texture sampling function
template <typename T>
CUDA_DEVICE_FUNCTION CUDA_INLINE T sample (
    CUtexObject texture, shared::TexDimInfo dimInfo, const Point2D& texCoord, float mipLevel)
{
    Point2D mTexCoord = adjustTexCoord (dimInfo, texCoord);
    return tex2DLod<T> (texture, mTexCoord.x, mTexCoord.y, mipLevel);
}

// Transform point from object space to world space
CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D transformPointFromObjectToWorldSpace (const Point3D& p)
{
    float3 xfmP = optixTransformPointFromObjectToWorldSpace (make_float3 (p.x, p.y, p.z));
    return Point3D (xfmP.x, xfmP.y, xfmP.z);
}

// Transform vector from object space to world space
CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D transformVectorFromObjectToWorldSpace (const Vector3D& v)
{
    float3 xfmV = optixTransformVectorFromObjectToWorldSpace (make_float3 (v.x, v.y, v.z));
    return Vector3D (xfmV.x, xfmV.y, xfmV.z);
}

// Transform normal from object space to world space
CUDA_DEVICE_FUNCTION CUDA_INLINE Normal3D transformNormalFromObjectToWorldSpace (const Normal3D& n)
{
    float3 xfmN = optixTransformNormalFromObjectToWorldSpace (make_float3 (n.x, n.y, n.z));
    return Normal3D (xfmN.x, xfmN.y, xfmN.z);
}