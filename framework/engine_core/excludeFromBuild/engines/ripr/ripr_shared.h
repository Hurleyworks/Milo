#pragma once

// much taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md

#include "../../common/common_shared.h"
#include "../../material/DeviceDisneyMaterial.h"

namespace ripr_shared
{
    static constexpr float probToSampleEnvLight = 0.25f;


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
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint32_t matSlot;
        Point3D positionInWorld;
        Normal3D normalInWorld;
        RGB albedo;
        RGB emittance;
        uint32_t hit : 1;
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
        int2 imageSize;
        optixu::BlockBuffer2D<shared::PCG32RNG, 1> rngBuffer;
        
        optixu::NativeBlockBuffer2D<GBuffer0Elements> GBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1Elements> GBuffer1[2];
        
        shared::ROBuffer<shared::DisneyData> materialDataBuffer;
        shared::ROBuffer<shared::InstanceData> instanceDataBufferArray[2];
        shared::ROBuffer<shared::GeometryInstanceData> geometryInstanceDataBuffer;
        shared::LightDistribution lightInstDist;
        shared::RegularConstantContinuousDistribution2D envLightImportanceMap;
        CUtexObject envLightTexture;
        
        optixu::NativeBlockBuffer2D<float4> beautyAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> flowAccumBuffer;
        
        PickInfo* pickInfos[2];
        
        // Experimental glass parameters
        uint32_t makeAllGlass : 1;
        uint32_t globalGlassType : 1;
        float globalGlassIOR;
        float globalTransmittanceDist;
        
        // Background
        uint32_t useSolidBackground : 1;
        float3 backgroundColor;
        
        // Area light support
        uint32_t numLightInsts;
        uint32_t enableAreaLights : 1;
        float areaLightPowerCoeff;
        
        // Firefly reduction
        float maxRadiance;
    };
    
    struct PerFramePipelineLaunchParameters
    {
        OptixTraversableHandle travHandle;
        uint32_t numAccumFrames;
        uint32_t frameIndex;
        
        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;
        
        float envLightPowerCoeff;
        float envLightRotation;
        
        int2 mousePosition;
        
        uint32_t maxPathLength : 4;
        uint32_t bufferIndex : 1;
        uint32_t resetFlowBuffer : 1;
        uint32_t enableJittering : 1;
        uint32_t enableEnvLight : 1;
        uint32_t enableBumpMapping : 1;
        uint32_t enableDebugPrint : 1;
        
        uint32_t debugSwitches;
        void setDebugSwitch(int32_t idx, bool b)
        {
            debugSwitches &= ~(1 << idx);
            debugSwitches |= b << idx;
        }
        CUDA_COMMON_FUNCTION bool getDebugSwitch(int32_t idx) const
        {
            return (debugSwitches >> idx) & 0b1;
        }
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

    // Primary ray payload for GBuffer generation
    using PrimaryRayPayloadSignature = optixu::PayloadSignature<HitPointParams*, PickInfo*>;
    
    // Path tracing payloads
    using SearchRayPayloadSignature = optixu::PayloadSignature<shared::PCG32RNG, SearchRayPayload*, HitPointParams*, RGB*, Normal3D*>;
    using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;

} // namespace ripr_shared

// Global launch parameters for device code
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM ripr_shared::PipelineLaunchParameters ripr_plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS ripr_shared::PipelineLaunchParameters ripr_plp;
#endif

#include "../../common/deviceCommon.h"

namespace ripr_shared
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


    struct HitPointParameter
    {
        float bcB, bcC;
        int32_t primIndex;

        CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get()
        {
            HitPointParameter ret;
            const float2 bc = optixGetTriangleBarycentrics();
            ret.bcB = bc.x;
            ret.bcC = bc.y;
            ret.primIndex = optixGetPrimitiveIndex();
            return ret;
        }
    };

    // Check if current pixel is under cursor
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool isCursorPixel()
    {
        // return ripr_plp.f->mousePosition == make_int2 (optixGetLaunchIndex());
        return false;
    }

    // Get debug print status
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled()
    {
        // return ripr_plp.f->enableDebugPrint;
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
        if (ripr_plp.s->envLightTexture && ripr_plp.f->enableEnvLight)
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

} // namespace ripr_shared

#endif // __CUDA_ARCH__