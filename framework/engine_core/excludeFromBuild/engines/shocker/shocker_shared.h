#pragma once

// much taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md

#include "../../common/common_shared.h"
#include "../../material/DeviceDisneyMaterial.h"

namespace shocker_shared
{
    static constexpr float probToSampleEnvLight = 0.25f;

    /* enum RayType
     {
         RayType_Search = 0,
         RayType_Visibility,
         NumRayTypes
     };*/

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

    struct PathTracingRayType
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
        uint32_t materialSlot;  // Material slot from SBT
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
        Vector2D motionVector;  // 8 bytes
        uint32_t materialSlot;  // 4 bytes
        uint32_t padding;       // 4 bytes padding to reach 16 bytes
    };

    struct PathTraceWriteOnlyPayload
    {
        Point3D nextOrigin;
        Vector3D nextDirection;
    };

    struct PathTraceReadWritePayload
    {
        shared::PCG32RNG rng;
        float initImportance;
        RGB alpha;
        RGB contribution;
        float prevDirPDensity;
        uint32_t maxLengthTerminate : 1;
        uint32_t terminate : 1;
        uint32_t pathLength : 6;
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
        void setDebugSwitch (int32_t idx, bool b)
        {
            debugSwitches &= ~(1 << idx);
            debugSwitches |= b << idx;
        }
        CUDA_COMMON_FUNCTION bool getDebugSwitch (int32_t idx) const
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

    using PrimaryRayPayloadSignature =
        optixu::PayloadSignature<HitPointParams*, PickInfo*>;
    using PathTraceRayPayloadSignature =
        optixu::PayloadSignature<PathTraceWriteOnlyPayload*, PathTraceReadWritePayload*>;
    using VisibilityRayPayloadSignature =
        optixu::PayloadSignature<float>;

    //// Primary ray payload for GBuffer generation
    // using PrimaryRayPayloadSignature = optixu::PayloadSignature<HitPointParams*, PickInfo*>;
    //
    //// Path tracing payloads
    // using SearchRayPayloadSignature = optixu::PayloadSignature<shared::PCG32RNG, SearchRayPayload*, HitPointParams*, RGB*, Normal3D*>;
    // using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;

} // namespace shocker_shared

// Global launch parameters for device code
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM shocker_shared::PipelineLaunchParameters shocker_plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS shocker_shared::PipelineLaunchParameters shocker_plp;
#endif

#include "../../common/deviceCommon.h"

namespace shocker_shared
{
    using namespace shared;

    CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint (
        const shared::InstanceData& inst,
        const shared::GeometryInstanceData& geomInst,
        uint32_t primIndex, float bcB, float bcC,
        Point3D* positionInWorld, Normal3D* shadingNormalInWorld, Vector3D* texCoord0DirInWorld,
        Normal3D* geometricNormalInWorld, Point2D* texCoord)
    {
        using namespace shared;
        const Triangle& tri = geomInst.triangleBuffer[primIndex];
        const Vertex& vA = geomInst.vertexBuffer[tri.index0];
        const Vertex& vB = geomInst.vertexBuffer[tri.index1];
        const Vertex& vC = geomInst.vertexBuffer[tri.index2];
        const float bcA = 1 - (bcB + bcC);

        // EN: Compute hit point properties in the local coordinates.
        const Point3D positionInObj = bcA * vA.position + bcB * vB.position + bcC * vC.position;
        *positionInWorld = inst.transform * positionInObj;
        *geometricNormalInWorld = normalize (
            inst.normalMatrix * Normal3D (cross (vB.position - vA.position, vC.position - vA.position)));
        const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        *texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

        // EN: Convert the local properties to ones in world coordinates.
        *shadingNormalInWorld = normalize (inst.normalMatrix * shadingNormalInObj);
        *texCoord0DirInWorld = inst.transform * texCoord0DirInObj;
        *texCoord0DirInWorld = normalize (
            *texCoord0DirInWorld - dot (*shadingNormalInWorld, *texCoord0DirInWorld) * *shadingNormalInWorld);
        if (!shadingNormalInWorld->allFinite())
        {
            *geometricNormalInWorld = Normal3D (0, 0, 1);
            *shadingNormalInWorld = Normal3D (0, 0, 1);
            *texCoord0DirInWorld = Vector3D (1, 0, 0);
        }
        if (!texCoord0DirInWorld->allFinite())
        {
            Vector3D bitangent;
            makeCoordinateSystem (*shadingNormalInWorld, texCoord0DirInWorld, &bitangent);
        }
    }

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

} // namespace shocker_shared

#endif // __CUDA_ARCH__