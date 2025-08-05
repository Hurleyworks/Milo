#pragma once

// much taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md

#include "common/common_shared.h"
#include "material/DeviceDisneyMaterial.h"

namespace milo_shared
{
    static constexpr float probToSampleEnvLight = 0.25f;

    enum PickRayType
    {
        PickRayType_Primary = 0,
        PickRayType_Visibility,
        NumPickRayTypes
    };

    enum RayType
    {
        RayType_Search = 0,
        RayType_Visibility,
        NumRayTypes
    };

    // In PerspectiveCamera struct in shared.h
    struct PerspectiveCamera
    {
        float aspect;
        float fovY;
        Point3D position;
        Matrix3x3 orientation;
        float lensSize = 0.0f; // Size of camera aperture
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

    struct PipelineLaunchParameters
    {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        uint32_t numAccumFrames;
        uint32_t bufferIndex;  // TODO: Temporarily commented to debug crash
        optixu::BlockBuffer2D<shared::PCG32RNG, 1> rngBuffer;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> flowAccumBuffer;
        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;  // Previous frame camera for temporal reprojection
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
        uint32_t numLightInsts;                   // Number of emissive instances
        uint32_t enableAreaLights : 1;            // Enable/disable area lights
        float areaLightPowerCoeff;                // Area light power multiplier
        
        shared::RegularConstantContinuousDistribution2D envLightImportanceMap;
        CUtexObject envLightTexture;
        
        // Material data buffer
        shared::ROBuffer<shared::DisneyData> materialDataBuffer;
        
        // Geometry instance data buffer
        shared::ROBuffer<shared::GeometryInstanceData> geometryInstanceDataBuffer;
        
        // Instance data buffer array (double buffered for async updates)
        shared::ROBuffer<shared::InstanceData> instanceDataBufferArray[2];
        
        // Pick info buffer
        PickInfo* pickInfoBuffer[2];  // Double buffered
        
        // Firefly reduction
        float maxRadiance;  // Maximum radiance value to clamp fireflies
    };

    struct SearchRayPayload
    {
        RGB alpha;
        RGB contribution;
        Point3D origin;
        Vector3D direction;
        float prevDirPDensity;  // PDF of the previous direction for MIS
        struct
        {
            uint32_t pathLength : 30;
            uint32_t terminate : 1;
            uint32_t deltaSampled : 1;
        };
    };

 
    using SearchRayPayloadSignature = optixu::PayloadSignature<shared::PCG32RNG, SearchRayPayload*, HitPointParams*, RGB*, Normal3D*>;
    using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;

} // namespace milo_shared