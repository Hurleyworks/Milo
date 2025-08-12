#pragma once

// much taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md

#include "../../common/common_shared.h"
#include "../../material/DeviceDisneyMaterial.h"

namespace claudia_shared
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

    // GBuffer ray types
    struct GBufferRayType {
        enum Value {
            Primary,
            NumTypes
        } value;
        
        CUDA_DEVICE_FUNCTION constexpr GBufferRayType(Value v = Primary) : value(v) {}
        CUDA_DEVICE_FUNCTION operator uint32_t() const { return static_cast<uint32_t>(value); }
    };
    
    // GBuffer data elements (matching RiPR exactly)
    struct GBuffer0Elements {
        uint32_t instSlot;
        uint32_t geomInstSlot;
        uint32_t primIndex;
        uint16_t qbcB;  // Quantized barycentric B
        uint16_t qbcC;  // Quantized barycentric C
    };
    
    struct GBuffer1Elements {
        Vector2D motionVector;
    };
    
    // GBuffer hit information
    struct GBufferHitInfo {
        Point3D positionInWorld;
        Point3D prevPositionInWorld;
        Normal3D geometricNormal;
        Normal3D shadingNormal;
        Vector3D tangent;
        Point2D texCoord;
        uint32_t objectID;
        uint32_t primitiveID;
    };

    // NEW: Split parameter structures for better performance and organization
    // Static parameters - rarely change, contain buffers and textures
    struct StaticPipelineLaunchParameters
    {
        // Image dimensions
        int2 imageSize;
        
        // RNG and accumulation buffers
        optixu::BlockBuffer2D<shared::PCG32RNG, 1> rngBuffer;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> flowAccumBuffer;
        
        // Environment texture and importance sampling
        CUtexObject envLightTexture;
        shared::RegularConstantContinuousDistribution2D envLightImportanceMap;
        
        // Material and geometry buffers
        shared::ROBuffer<shared::DisneyData> materialDataBuffer;
        shared::ROBuffer<shared::GeometryInstanceData> geometryInstanceDataBuffer;
        shared::ROBuffer<shared::InstanceData> instanceDataBufferArray[2];
        
        // Light distribution for area lights
        shared::LightDistribution lightInstDist;
        
        // GBuffer storage (double buffered)
        optixu::NativeBlockBuffer2D<GBuffer0Elements> GBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1Elements> GBuffer1[2];
        
        // Pick info buffers
        PickInfo* pickInfos[2];  // Note: renamed to match RiPR convention
    };

    // Per-frame parameters - change frequently
    struct PerFramePipelineLaunchParameters
    {
        // Traversable and frame info
        OptixTraversableHandle travHandle;
        uint32_t numAccumFrames;
        uint32_t frameIndex;
        
        // Camera state
        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;
        
        // Mouse interaction
        int2 mousePosition;
        
        // Environment light settings
        float envLightPowerCoeff;
        float envLightRotation;
        
        // Area light settings
        uint32_t numLightInsts;
        float areaLightPowerCoeff;
        
        // Render settings (bit flags)
        uint32_t maxPathLength : 4;
        uint32_t bufferIndex : 1;
        uint32_t resetFlowBuffer : 1;
        uint32_t enableJittering : 1;
        uint32_t enableEnvLight : 1;
        uint32_t enableBumpMapping : 1;
        uint32_t enableAreaLights : 1;
        uint32_t enableGBuffer : 1;
        uint32_t useSolidBackground : 1;
        uint32_t useCameraSpaceNormal : 1;
        uint32_t enableDebugPrint : 1;
        uint32_t enableDenoiser : 1;
        
        // Experimental glass settings
        uint32_t makeAllGlass : 1;
        uint32_t globalGlassType : 1;
        float globalGlassIOR;
        float globalTransmittanceDist;
        
        // Background and firefly control
        float3 backgroundColor;
        float maxRadiance;
    };

    // OLD: Keep for backward compatibility during migration
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
        
        // GBuffer storage (double buffered)
        optixu::NativeBlockBuffer2D<GBuffer0Elements> GBuffer0[2];
        optixu::NativeBlockBuffer2D<GBuffer1Elements> GBuffer1[2];
        
        // GBuffer control
        uint32_t enableGBuffer : 1;
        uint32_t resetMotionBuffer : 1;  // Reset motion vectors for temporal effects
    };

    // NEW: Split version of PipelineLaunchParameters (like RiPR)
    struct PipelineLaunchParametersSplit
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
    
    // GBuffer payload signatures
    using GBufferRayPayloadSignature = optixu::PayloadSignature<HitPointParams*, PickInfo*>;
    
    // Maximum number of ray types for SBT configuration
    constexpr uint32_t maxNumRayTypes = 2;

} // namespace claudia_shared

// Device-only code section
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#if defined(PURE_CUDA)
CUDA_CONSTANT_MEM claudia_shared::PipelineLaunchParametersSplit claudia_plp_split;
#else
RT_PIPELINE_LAUNCH_PARAMETERS claudia_shared::PipelineLaunchParametersSplit claudia_plp_split;
#endif

#include "../../common/deviceCommon.h"

namespace claudia_shared
{

    // Hit group SBT record data
    struct HitGroupSBTRecordData
    {
        uint32_t geomInstSlot;
        uint32_t materialSlot;

        CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData& get()
        {
            return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
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
        return claudia_plp_split.f->mousePosition == make_int2(optixGetLaunchIndex());
    }

    // Get debug print status
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled()
    {
        return claudia_plp_split.f->enableDebugPrint;
    }

    // Compute surface point from barycentric coordinates
    CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint(
        const shared::InstanceData& inst,
        const shared::GeometryInstanceData& geomInst,
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
        *positionInWorld = transformPointFromObjectToWorldSpace(positionInObj);

        // Compute geometric normal
        *geometricNormalInWorld = normalize(
            transformNormalFromObjectToWorldSpace(Normal3D(cross(vB.position - vA.position, vC.position - vA.position))));

        // Interpolate shading normal
        const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        *shadingNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(shadingNormalInObj));

        // Interpolate texture coordinates
        *texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

        // Compute texture coordinate direction
        const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        *texCoord0DirInWorld = transformVectorFromObjectToWorldSpace(texCoord0DirInObj);
        *texCoord0DirInWorld = normalize(
            *texCoord0DirInWorld - dot(*shadingNormalInWorld, *texCoord0DirInWorld) * *shadingNormalInWorld);

        // Handle degenerate cases
        if (!shadingNormalInWorld->allFinite())
        {
            *geometricNormalInWorld = Normal3D(0, 0, 1);
            *shadingNormalInWorld = Normal3D(0, 0, 1);
            *texCoord0DirInWorld = Vector3D(1, 0, 0);
        }
        if (!texCoord0DirInWorld->allFinite())
        {
            Vector3D bitangent;
            makeCoordinateSystem(*shadingNormalInWorld, texCoord0DirInWorld, &bitangent);
        }
    }

    // Normal map reading helper (to be implemented)
    CUDA_DEVICE_FUNCTION CUDA_INLINE Normal3D readNormalMap(CUtexObject normalMap, const Point2D& texCoord)
    {
        // TODO: Implement normal map reading
        // For now, return neutral normal
        return Normal3D(0.5f, 0.5f, 1.0f);
    }

    // Apply bump mapping to shading frame
    CUDA_DEVICE_FUNCTION CUDA_INLINE void applyBumpMapping(
        const Normal3D& modLocalNormal,
        ReferenceFrame* shadingFrame)
    {
        // TODO: Implement bump mapping
        // For now, do nothing
    }

} // namespace claudia_shared

#endif // __CUDA_ARCH__