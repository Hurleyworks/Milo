#pragma once

// shocker_shared.h
// Shared definitions between host and device code for ShockerEngine
// Matches the pattern from sample code path_tracing_shared.h

#include "common/common_shared.h"
#include "model/ShockerDeviceCore.h"

namespace shocker_shared
{

    // Constants (matching sample code)
    static constexpr float probToSampleEnvLight = 0.25f;

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

    // Camera structure (matching sample code exactly)
    struct PerspectiveCamera
    {
        float aspect;
        float fovY;
        Point3D position;
        Matrix3x3 orientation;

        CUDA_COMMON_FUNCTION Point2D calcScreenPosition (const Point3D& posInWorld) const
        {
            const Matrix3x3 invOri = invert (orientation);
            const Point3D posInView (invOri * (posInWorld - position));
            const Point2D posAtZ1 (posInView.x / posInView.z, posInView.y / posInView.z);
            const float h = 2 * std::tan (fovY / 2);
            const float w = aspect * h;
            return Point2D (1 - (posAtZ1.x + 0.5f * w) / w,
                            1 - (posAtZ1.y + 0.5f * h) / h);
        }
    };

    // Hit point parameters
    struct HitPointParams
    {
        RGB albedo;
        Point3D positionInWorld;
        Point3D prevPositionInWorld;
        Normal3D shadingNormalInWorld;
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint16_t qbcB; // Quantized barycentric B
        uint16_t qbcC; // Quantized barycentric C
    };

    // Light sample structure
    struct LightSample
    {
        RGB emittance;
        Point3D position;
        Normal3D normal;
        uint32_t atInfinity : 1;

        CUDA_COMMON_FUNCTION LightSample() :
            atInfinity (false) {}
    };

    // Pick info structure for mouse interaction
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

    // Static pipeline launch parameters
    struct StaticPipelineLaunchParameters
    {
        int2 imageSize;

        // RNG buffer
        optixu::BlockBuffer2D<shared::PCG32RNG, 1> rngBuffer;

        // G-buffers (double buffered)
        optixu::NativeBlockBuffer2D<GBuffer0Elements> GBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1Elements> GBuffer1[2];

        // Disney material and instance data (using Shocker types from ShockerDeviceCore.h)
        shared::ROBuffer<shared::DisneyData> disneyMaterialBuffer;
        shared::ROBuffer<shocker::ShockerNodeData> instanceDataBufferArray[2]; // Double buffered for temporal effects
        shared::ROBuffer<shocker::ShockerSurfaceData> geometryInstanceDataBuffer;

        // Light data
        shared::LightDistribution lightInstDist;
        shared::RegularConstantContinuousDistribution2D envLightImportanceMap;
        CUtexObject envLightTexture;

        // Accumulation buffers
        optixu::NativeBlockBuffer2D<float4> beautyAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
        optixu::NativeBlockBuffer2D<float2> motionAccumBuffer;

        // Pick info for mouse interaction
        PickInfo* pickInfos[2];
    };

    // Per-frame pipeline launch parameters
    struct PerFramePipelineLaunchParameters
    {
        OptixTraversableHandle travHandle;
        uint32_t numAccumFrames;
        uint32_t frameIndex;

        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;

        // Environment light
        float envLightPowerCoeff;
        float envLightRotation;

        // Interaction
        int2 mousePosition;

        // Render settings (bit flags matching sample code)
        uint32_t maxPathLength : 4;
        uint32_t bufferIndex : 1;
        uint32_t resetFlowBuffer : 1; // For optical flow
        uint32_t enableJittering : 1;
        uint32_t enableEnvLight : 1;
        uint32_t enableBumpMapping : 1;
        uint32_t enableDebugPrint : 1;
        uint32_t enableDenoiser : 1;
        uint32_t renderMode : 3; // 0=GBuffer, 1=PathTrace, 2+=Debug modes

        // Debug switches
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

    // Main pipeline launch parameters structure
    struct PipelineLaunchParameters
    {
        StaticPipelineLaunchParameters* s;
        PerFramePipelineLaunchParameters* f;
    };

    // Buffer display modes for debugging
    enum class BufferToDisplay
    {
        NoisyBeauty = 0,
        Albedo,
        Normal,
        Flow,
        DenoisedBeauty,
    };

    // Payload structures for OptiX ray tracing

    // G-Buffer ray payload (simple, single bounce)
    struct GBufferPayload
    {
        HitPointParams* hitPoint;
        PickInfo* pickInfo;
    };

    // Path tracing payload (complex, multi-bounce)
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

    // Visibility/shadow ray payload
    struct VisibilityPayload
    {
        float visibility; // 0.0 = blocked, 1.0 = visible
    };

    // Payload signatures for OptiX (matching sample code pattern)
    using PrimaryRayPayloadSignature =
        optixu::PayloadSignature<HitPointParams*, PickInfo*>;
    using PathTraceRayPayloadSignature =
        optixu::PayloadSignature<PathTraceWriteOnlyPayload*, PathTraceReadWritePayload*>;
    using VisibilityRayPayloadSignature =
        optixu::PayloadSignature<float>;

} // namespace shocker_shared

// Global launch parameters for device code
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM shocker_shared::PipelineLaunchParameters plp;
#else
RT_PIPELINE_LAUNCH_PARAMETERS shocker_shared::PipelineLaunchParameters plp;
#endif

#include "common/common_device.cuh"

namespace shocker_shared
{

    // Hit group SBT record data
    struct HitGroupSBTRecordData
    {
        uint32_t geomInstSlot;

        CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData& get()
        {
            return *reinterpret_cast<HitGroupSBTRecordData*> (optixGetSbtDataPointer());
        }
    };

    // Hit point parameter helper
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
        return plp.f->mousePosition == make_int2 (optixGetLaunchIndex());
    }

    // Get debug print status
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled()
    {
        return plp.f->enableDebugPrint;
    }

    // Compute surface point from barycentric coordinates
    CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint (
        const shocker::ShockerNodeData& inst,
        const shocker::ShockerSurfaceData& geomInst,
        uint32_t primIndex, float bcB, float bcC,
        Point3D* positionInWorld,
        Normal3D* shadingNormalInWorld,
        Vector3D* texCoord0DirInWorld,
        Normal3D* geometricNormalInWorld,
        Point2D* texCoord)
    {
        const shared::Triangle& tri = geomInst.triangleBuffer[primIndex];
        const shared::Vertex& vA = geomInst.vertexBuffer[tri.index0];
        const shared::Vertex& vB = geomInst.vertexBuffer[tri.index1];
        const shared::Vertex& vC = geomInst.vertexBuffer[tri.index2];
        const float bcA = 1 - (bcB + bcC);

        // Compute position in world space
        const Point3D positionInObj = bcA * vA.position + bcB * vB.position + bcC * vC.position;
        *positionInWorld = inst.transform * positionInObj;

        // Compute geometric normal
        *geometricNormalInWorld = normalize (
            inst.normalMatrix * Normal3D (cross (vB.position - vA.position, vC.position - vA.position)));

        // Interpolate shading normal
        const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        *shadingNormalInWorld = normalize (inst.normalMatrix * shadingNormalInObj);

        // Interpolate texture coordinates
        *texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

        // Compute texture coordinate direction
        const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        *texCoord0DirInWorld = inst.transform * texCoord0DirInObj;
        *texCoord0DirInWorld = normalize (
            *texCoord0DirInWorld - dot (*shadingNormalInWorld, *texCoord0DirInWorld) * *shadingNormalInWorld);

        // Handle degenerate cases
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

} // namespace shocker_shared

#endif // __CUDA_ARCH__