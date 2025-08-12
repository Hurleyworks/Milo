#pragma once

// much taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md

#include "../../common/common_shared.h"
#include "../../material/DeviceDisneyMaterial.h"

namespace claudia_shared
{
    static constexpr float probToSampleEnvLight = 0.25f;

    /* enum PickRayType
     {
         PickRayType_Primary = 0,
         PickRayType_Visibility,
         NumPickRayTypes
     };
     */
    enum RayType
    {
        RayType_Search = 0,
        RayType_Visibility,
        NumRayTypes
    };

    struct GBufferRayType
    {
        enum Value
        {
            Primary,
            NumTypes
        } value;

        CUDA_DEVICE_FUNCTION constexpr GBufferRayType (Value v = Primary) :
            value (v) {}

        CUDA_DEVICE_FUNCTION operator uint32_t() const
        {
            return static_cast<uint32_t> (value);
        }
    };

    /* struct PathTracingRayType
     {
         enum Value
         {
             Closest,
             Visibility,
             NumTypes
         } value;

         CUDA_DEVICE_FUNCTION constexpr PathTracingRayType (Value v = Closest) :
             value (v) {}

         CUDA_DEVICE_FUNCTION operator uint32_t() const
         {
             return static_cast<uint32_t> (value);
         }
     };*/

    constexpr uint32_t maxNumRayTypes = 2;

    // In PerspectiveCamera struct in shared.h
    struct PerspectiveCamera
    {
        float aspect;
        float fovY;
        Point3D position;
        Matrix3x3 orientation;
        float lensSize = 0.0f;      // Size of camera aperture
        float focusDistance = 5.0f; // Distance to focal plane

        CUDA_COMMON_FUNCTION Point2D calcScreenPosition (const Point3D& posInWorld) const
        {
            Matrix3x3 invOri = invert (orientation);
            Point3D posInView (invOri * (posInWorld - position));
            Point2D posAtZ1 (posInView.x / posInView.z, posInView.y / posInView.z);
            float h = 2 * std::tan (fovY / 2);
            float w = aspect * h;
            return Point2D (1 - (posAtZ1.x + 0.5f * w) / w,
                            1 - (posAtZ1.y + 0.5f * h) / h);
        }
    };

    struct LightSample
    {
        RGB emittance;
        Point3D position;
        Normal3D normal;
        unsigned int atInfinity : 1;
    };

    struct PickInfo
    {
        Point3D positionInWorld;
        Normal3D normalInWorld;
        Point3D rayOrigin;
        Vector3D rayDir;

        uint32_t instanceIndex;
        uint32_t matIndex;
        uint32_t primIndex;
        uint32_t instanceID;
        unsigned int matID : 16;
        unsigned int hit : 1;
    };

    struct HitPointParams
    {
        RGB albedo;
        Point3D positionInWorld;
        Point3D prevPositionInWorld;
        Normal3D normalInWorld;
        Point2D texCoord;
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint16_t qbcB;
        uint16_t qbcC;
    };

    struct GBuffer0Elements
    {
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint16_t qbcB;
        uint16_t qbcC;
    };

    struct GBuffer1Elements
    {
        Vector2D motionVector;
    };

    struct StaticPipelineLaunchParameters
    {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        uint32_t numAccumFrames;
        uint32_t bufferIndex;

        optixu::BlockBuffer2D<shared::PCG32RNG, 1> rngBuffer;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> flowAccumBuffer;

        PerspectiveCamera camera;
        PerspectiveCamera prevCamera; // Previous frame camera for temporal reprojection
        uint32_t useCameraSpaceNormal : 1;
        uint32_t bounceLimit; // Maximum path length for path tracing

        // Experimental
        uint32_t makeAllGlass : 1;
        uint32_t globalGlassType : 1;
        float globalGlassIOR;
        float globalTransmittanceDist;

        // skydome environment
        uint32_t enableEnvLight : 1;
        float envLightPowerCoeff;
        float envLightRotation;
        uint32_t useSolidBackground : 1;
        float3 backgroundColor; // Solid background color when not using HDR

        // Area light support
        shared::LightDistribution lightInstDist;
        uint32_t numLightInsts;        // Number of emissive instances
        uint32_t enableAreaLights : 1; // Enable/disable area lights
        float areaLightPowerCoeff;     // Area light power multiplier

        shared::RegularConstantContinuousDistribution2D envLightImportanceMap;
        CUtexObject envLightTexture;

        // Material data buffer
        shared::ROBuffer<shared::DisneyData> materialDataBuffer;

        // Geometry instance data buffer
        shared::ROBuffer<shared::GeometryInstanceData> geometryInstanceDataBuffer;

        // Instance data buffer array (double buffered for async updates)
        shared::ROBuffer<shared::InstanceData> instanceDataBufferArray[2];

        // Pick info buffer
        PickInfo* pickInfoBuffer[2]; // Double buffered

        // geometry buffers 
        optixu::NativeBlockBuffer2D<GBuffer0Elements> geoBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1Elements> geoBuffer1[2];

        // Firefly reduction
        float maxRadiance; // Maximum radiance value to clamp fireflies
    };
    struct PerFramePipelineLaunchParameters
    {
    };

    struct PipelineLaunchParameters
    {
        StaticPipelineLaunchParameters* s;
        PerFramePipelineLaunchParameters* f;
    };

    struct SearchRayPayload
    {
        RGB alpha;
        RGB contribution;
        Point3D origin;
        Vector3D direction;
        float prevDirPDensity; // PDF of the previous direction for MIS
        struct
        {
            uint32_t pathLength : 30;
            uint32_t terminate : 1;
            uint32_t deltaSampled : 1;
        };
    };

    using SearchRayPayloadSignature = optixu::PayloadSignature<shared::PCG32RNG, SearchRayPayload*, HitPointParams*, RGB*, Normal3D*>;
    using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;

} // namespace claudia_shared

// Global launch parameters for device code
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM claudia_shared::PipelineLaunchParameters claudia_plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS claudia_shared::PipelineLaunchParameters claudia_plp;
#endif

#include "../../common/deviceCommon.h"

namespace claudia_shared
{
    using namespace shared;

    // This struct is used to fetch geometry instance and material data from
    // the Shader Binding Table (SBT) in OptiX.
    struct HitGroupSBTRecordData
    {
        uint32_t geomInstSlot; // Geometry instance slot index in the global buffer
        uint32_t materialSlot; // Material slot index in the material buffer

        // Static member function to retrieve the SBT record data
        CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData& get()
        {
            // Use optixGetSbtDataPointer() to get the pointer to the SBT data
            // Cast the pointer to type HitGroupSBTRecordData and dereference it
            return *reinterpret_cast<HitGroupSBTRecordData*> (optixGetSbtDataPointer());
        }
    };

    // Define a struct called HitPointParameter to hold hit point info
    struct HitPointParameter
    {
        float b1, b2;      // Barycentric coordinates
        int32_t primIndex; // Index of the primitive hit by the ray

        // Static member function to get hit point parameters
        CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get()
        {
            HitPointParameter ret; // Create an instance of the struct

            // Get barycentric coordinates from OptiX API
            float2 bc = optixGetTriangleBarycentrics();

            // Store the barycentric coordinates in the struct
            ret.b1 = bc.x;
            ret.b2 = bc.y;

            // Get the index of the primitive hit by the ray from OptiX API
            ret.primIndex = optixGetPrimitiveIndex();

            // Return the populated struct
            return ret;
        }
    };

    // Check if current pixel is under cursor
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool isCursorPixel()
    {
        // return claudia_plp.f->mousePosition == make_int2 (optixGetLaunchIndex());
        return false;
    }

    // Get debug print status
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled()
    {
        // return claudia_plp.f->enableDebugPrint;
        return false;
    }

    // This function calculates various attributes of a surface point
    // given its barycentric coordinates (b1, b2) and the index (primIndex)
    // of the triangle it belongs to. It computes the world-space position,
    // shading normal, texture coordinates, and so forth for this surface point.
    // It also computes a hypothetical area PDF (hypAreaPDensity) that could
    // be used in light sampling.
    CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint (
        const shared::GeometryInstanceData& geomInst,
        uint32_t primIndex, float b1, float b2,
        const Point3D& referencePoint,
        Point3D* positionInWorld, Normal3D* shadingNormalInWorld, Vector3D* texCoord0DirInWorld,
        Normal3D* geometricNormalInWorld, Point2D* texCoord,
        float* hypAreaPDensity)
    {
        // Fetch the vertices of the triangle given its index
        const Triangle& tri = geomInst.triangleBuffer[primIndex];
        const Vertex& v0 = geomInst.vertexBuffer[tri.index0];
        const Vertex& v1 = geomInst.vertexBuffer[tri.index1];
        const Vertex& v2 = geomInst.vertexBuffer[tri.index2];

        // Transform vertex positions to world space
        const Point3D p[3] = {
            transformPointFromObjectToWorldSpace (v0.position),
            transformPointFromObjectToWorldSpace (v1.position),
            transformPointFromObjectToWorldSpace (v2.position),
        };

        // Calculate barycentric coordinates
        float b0 = 1 - (b1 + b2);

        // Compute the position in world space using barycentric coordinates
        *positionInWorld = b0 * p[0] + b1 * p[1] + b2 * p[2];

        // Compute interpolated shading normal and texture direction
        Normal3D shadingNormal = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        Vector3D texCoord0Dir = b0 * v0.texCoord0Dir + b1 * v1.texCoord0Dir + b2 * v2.texCoord0Dir;

        // Compute geometric normal and area of the triangle
        Normal3D geometricNormal (cross (p[1] - p[0], p[2] - p[0]));
        float area = 0.5f * length (geometricNormal);

        // Compute the texture coordinates
        *texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        // Transform shading normal and texture direction to world space
        *shadingNormalInWorld = normalize (transformNormalFromObjectToWorldSpace (shadingNormal));
        *texCoord0DirInWorld = normalize (transformVectorFromObjectToWorldSpace (texCoord0Dir));
        *geometricNormalInWorld = normalize (geometricNormal);

        // Check for invalid normals and give them a default value
        if (!shadingNormalInWorld->allFinite())
        {
            *shadingNormalInWorld = Normal3D (0, 0, 1);
            *texCoord0DirInWorld = Vector3D (1, 0, 0);
        }

        // Check for invalid texture directions and correct them
        if (!texCoord0DirInWorld->allFinite())
        {
            Vector3D bitangent;
            makeCoordinateSystem (*shadingNormalInWorld, texCoord0DirInWorld, &bitangent);
        }

        // Compute the probability of sampling this light
        float lightProb = 1.0f;
        if (claudia_plp.s->envLightTexture && claudia_plp.s->enableEnvLight)
            lightProb *= (1 - probToSampleEnvLight);

        // Check for invalid probabilities
        if (!isfinite (lightProb))
        {
            *hypAreaPDensity = 0.0f;
            return;
        }

        // Compute the hypothetical area PDF
        *hypAreaPDensity = lightProb / area;
    }

} // namespace claudia_shared

#endif // __CUDA_ARCH__