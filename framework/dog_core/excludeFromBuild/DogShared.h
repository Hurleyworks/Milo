#pragma once

#include "common/common_shared.h"
#include "material/DeviceDisneyMaterial.h"

namespace DogShared
{
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

    struct HitPointParams
    {
        RGB albedo;
        Point3D positionInWorld;
        Point3D prevPositionInWorld;
        Normal3D shadingNormalInWorld;
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint16_t qbcB;
        uint16_t qbcC;
    };

    struct LightSample
    {
        RGB emittance;
        Point3D position;
        Normal3D normal;
        uint32_t atInfinity : 1;

        CUDA_COMMON_FUNCTION LightSample() :
            atInfinity (false) {}
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
        optixu::NativeBlockBuffer2D<shared::PCG32RNG> rngBuffer;

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

        PickInfo* pickInfos[2];
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

    enum class BufferToDisplay
    {
        NoisyBeauty = 0,
        Albedo,
        Normal,
        Flow,
        DenoisedBeauty,
    };

    using PrimaryRayPayloadSignature =
        optixu::PayloadSignature<DogShared::HitPointParams*, DogShared::PickInfo*>;

    using PathTraceRayPayloadSignature =
        optixu::PayloadSignature<DogShared::PathTraceWriteOnlyPayload*, DogShared::PathTraceReadWritePayload*>;

    using VisibilityRayPayloadSignature =
        optixu::PayloadSignature<float>;

} // namespace DogShared
