#pragma once

// much taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md

#include "../../common/common_shared.h"
#include "../../material/DeviceDisneyMaterial.h"

namespace shocker_shared
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

    // PickInfo has been moved to common_shared.h for use across all engines
    using PickInfo = shared::PickInfo;

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
        uint32_t materialSlot; // Material slot from SBT
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
        Vector2D motionVector; // 8 bytes
        uint32_t materialSlot; // 4 bytes
        uint32_t padding;      // 4 bytes padding to reach 16 bytes
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

    template <bool useSolidAngleSampling>
    CUDA_DEVICE_FUNCTION CUDA_INLINE void sampleLight (
        const Point3D& shadingPoint,
        float ul, bool sampleEnvLight, float u0, float u1,
        shocker_shared::LightSample* lightSample, float* areaPDensity)
    {
        using namespace shared;
        CUtexObject texEmittance = 0;
        RGB emittance (0.0f, 0.0f, 0.0f);
        Point2D texCoord;
        if (sampleEnvLight)
        {
            float u, v;
            float uvPDF;
            shocker_plp.s->envLightImportanceMap.sample (u0, u1, &u, &v, &uvPDF);
            const float phi = 2 * pi_v<float> * u;
            const float theta = pi_v<float> * v;

            float posPhi = phi - shocker_plp.f->envLightRotation;
            posPhi = posPhi - floorf (posPhi / (2 * pi_v<float>)) * 2 * pi_v<float>;

            const Vector3D direction = fromPolarYUp (posPhi, theta);
            const Point3D position (direction.x, direction.y, direction.z);
            lightSample->position = position;
            lightSample->atInfinity = true;

            lightSample->normal = Normal3D (-position);

            // EN: convert the PDF in texture space to one with respect to area.
            // The true value is: lim_{l to inf} uvPDF / (2 * Pi * Pi * sin(theta)) / l^2
            const float sinTheta = std::sin (theta);
            if (sinTheta == 0.0f)
            {
                *areaPDensity = 0.0f;
                return;
            }
            *areaPDensity = uvPDF / (2 * pi_v<float> * pi_v<float> * sinTheta);

            texEmittance = shocker_plp.s->envLightTexture;

            // EN: Multiply a coefficient to make the return value possible to be handled as luminous emittance.
            emittance = RGB (pi_v<float> * shocker_plp.f->envLightPowerCoeff);
            texCoord.x = u;
            texCoord.y = v;
        }
        else
        {
            float lightProb = 1.0f;

            // First, sample an instance
            float instProb;
            float uGeomInst;
            const uint32_t instSlot = shocker_plp.s->lightInstDist.sample (ul, &instProb, &uGeomInst);
            lightProb *= instProb;
            const InstanceData& inst = shocker_plp.s->instanceDataBufferArray[shocker_plp.f->bufferIndex][instSlot];
            if (instProb == 0.0f)
            {
                *areaPDensity = 0.0f;
                return;
            }

            // Next, sample a geometry instance which belongs to the sampled instance
            float geomInstProb;
            float uPrim;
            const uint32_t geomInstIndexInInst = inst.lightGeomInstDist.sample (uGeomInst, &geomInstProb, &uPrim);
            const uint32_t geomInstSlot = inst.geomInstSlots[geomInstIndexInInst];
            lightProb *= geomInstProb;
            const GeometryInstanceData& geomInst = shocker_plp.s->geometryInstanceDataBuffer[geomInstSlot];
            if (geomInstProb == 0.0f)
            {
                *areaPDensity = 0.0f;
                return;
            }

            // Finally, sample a primitive which belongs to the sampled geometry instance
            float primProb;
            const uint32_t primIndex = geomInst.emitterPrimDist.sample (uPrim, &primProb);
            lightProb *= primProb;

            const DisneyData& mat = shocker_plp.s->materialDataBuffer[geomInst.materialSlot];

            const shared::Triangle& tri = geomInst.triangleBuffer[primIndex];
            const shared::Vertex& vA = geomInst.vertexBuffer[tri.index0];
            const shared::Vertex& vB = geomInst.vertexBuffer[tri.index1];
            const shared::Vertex& vC = geomInst.vertexBuffer[tri.index2];
            const Point3D pA = inst.transform * vA.position;
            const Point3D pB = inst.transform * vB.position;
            const Point3D pC = inst.transform * vC.position;

            Normal3D geomNormal (cross (pB - pA, pC - pA));

            float bcA, bcB, bcC;
            if constexpr (useSolidAngleSampling)
            {
                // Uniform sampling in solid angle subtended by the triangle for the shading point
                float dist;
                Vector3D dir;
                float dirPDF;
                {
                    const auto project = [] (const Vector3D& vA, const Vector3D& vB)
                    {
                        return normalize (vA - dot (vA, vB) * vB);
                    };

                    const Vector3D A = normalize (pA - shadingPoint);
                    const Vector3D B = normalize (pB - shadingPoint);
                    const Vector3D C = normalize (pC - shadingPoint);
                    const Vector3D cAB = normalize (cross (A, B));
                    const Vector3D cBC = normalize (cross (B, C));
                    const Vector3D cCA = normalize (cross (C, A));
                    const float cos_c = dot (A, B);
                    const float cosAlpha = -dot (cAB, cCA);
                    const float cosBeta = -dot (cBC, cAB);
                    const float cosGamma = -dot (cCA, cBC);
                    const float alpha = std::acos (cosAlpha);
                    const float sinAlpha = std::sqrt (1 - pow2 (cosAlpha));
                    const float sphArea = alpha + std::acos (cosBeta) + std::acos (cosGamma) - pi_v<float>;

                    const float sphAreaHat = sphArea * u0;
                    const float s = std::sin (sphAreaHat - alpha);
                    const float t = std::cos (sphAreaHat - alpha);
                    const float uu = t - cosAlpha;
                    const float vv = s + sinAlpha * cos_c;

                    const float q = ((vv * t - uu * s) * cosAlpha - vv) / ((vv * s + uu * t) * sinAlpha);

                    const Vector3D cHat = q * A + std::sqrt (1 - pow2 (q)) * project (C, A);
                    const float z = 1 - u1 * (1 - dot (cHat, B));
                    const Vector3D P = z * B + std::sqrt (1 - pow2 (z)) * project (cHat, B);

                    const auto restoreBarycentrics = [&geomNormal]
                        (const Point3D& org, const Vector3D& dir,
                         const Point3D& pA, const Point3D& pB, const Point3D& pC,
                         float* dist, float* bcB, float* bcC)
                    {
                        const Vector3D eAB = pB - pA;
                        const Vector3D eAC = pC - pA;
                        const Vector3D pVec = cross (dir, eAC);
                        const float recDet = 1.0f / dot (eAB, pVec);
                        const Vector3D tVec = org - pA;
                        *bcB = dot (tVec, pVec) * recDet;
                        const Vector3D qVec = cross (tVec, eAB);
                        *bcC = dot (dir, qVec) * recDet;
                        *dist = dot (eAC, qVec) * recDet;
                    };
                    dir = P;
                    restoreBarycentrics (shadingPoint, dir, pA, pB, pC, &dist, &bcB, &bcC);
                    bcA = 1 - (bcB + bcC);
                    dirPDF = 1 / sphArea;
                }

                geomNormal = normalize (geomNormal);
                const float lpCos = -dot (dir, geomNormal);
                if (lpCos > 0 && stc::isfinite (dirPDF))
                    *areaPDensity = lightProb * (dirPDF * lpCos / pow2 (dist));
                else
                    *areaPDensity = 0.0f;
            }
            else
            {
                // Uniform sampling on unit triangle
                bcA = 0.5f * u0;
                bcB = 0.5f * u1;
                const float offset = bcB - bcA;
                if (offset > 0)
                    bcB += offset;
                else
                    bcA -= offset;
                bcC = 1 - (bcA + bcB);

                const float recArea = 2.0f / length (geomNormal);
                *areaPDensity = lightProb * recArea;
            }
            lightSample->position = bcA * pA + bcB * pB + bcC * pC;
            lightSample->atInfinity = false;
            lightSample->normal = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
            lightSample->normal = normalize (inst.normalMatrix * lightSample->normal);

            if (mat.emissive)
            {
                texEmittance = mat.emissive;
                emittance = RGB (1.0f, 1.0f, 1.0f);
                texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;
            }
        }
        if (texEmittance)
        {
            const float4 texValue = tex2DLod<float4> (texEmittance, texCoord.x, texCoord.y, 0.0f);
            emittance *= RGB (getXYZ (texValue));
        }
        lightSample->emittance = emittance;
    }

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

    template <bool computeHypotheticalAreaPDensity, bool useSolidAngleSampling>
    CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint (
        const shared::InstanceData& inst,
        const shared::GeometryInstanceData& geomInst,
        uint32_t primIndex, float bcB, float bcC,
        const Point3D& referencePoint,
        Point3D* positionInWorld, Normal3D* shadingNormalInWorld, Vector3D* texCoord0DirInWorld,
        Normal3D* geometricNormalInWorld, Point2D* texCoord,
        float* hypAreaPDensity)
    {
        using namespace shared;
        const Triangle& tri = geomInst.triangleBuffer[primIndex];
        const Vertex& vA = geomInst.vertexBuffer[tri.index0];
        const Vertex& vB = geomInst.vertexBuffer[tri.index1];
        const Vertex& vC = geomInst.vertexBuffer[tri.index2];
        const Point3D pA = transformPointFromObjectToWorldSpace (vA.position);
        const Point3D pB = transformPointFromObjectToWorldSpace (vB.position);
        const Point3D pC = transformPointFromObjectToWorldSpace (vC.position);
        const float bcA = 1 - (bcB + bcC);

        // EN: Compute hit point properties in the local coordinates.
        *positionInWorld = bcA * pA + bcB * pB + bcC * pC;
        const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        *texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

        *geometricNormalInWorld = Normal3D (cross (pB - pA, pC - pA));
        float area;
        if constexpr (computeHypotheticalAreaPDensity && !useSolidAngleSampling)
        {
            area = 0.5f * length (*geometricNormalInWorld);
            *geometricNormalInWorld = *geometricNormalInWorld / (2 * area);
        }
        else
        {
            *geometricNormalInWorld = normalize (*geometricNormalInWorld);
            (void)area;
        }

        // EN: Convert the local properties to ones in world coordinates.
        *shadingNormalInWorld = normalize (transformNormalFromObjectToWorldSpace (shadingNormalInObj));
        *texCoord0DirInWorld = normalize (transformVectorFromObjectToWorldSpace (texCoord0DirInObj));
        if (!shadingNormalInWorld->allFinite())
        {
            *shadingNormalInWorld = Normal3D (0, 0, 1);
            *texCoord0DirInWorld = Vector3D (1, 0, 0);
        }
        if (!texCoord0DirInWorld->allFinite())
        {
            Vector3D bitangent;
            makeCoordinateSystem (*shadingNormalInWorld, texCoord0DirInWorld, &bitangent);
        }

        if constexpr (computeHypotheticalAreaPDensity)
        {
            // EN: Compute a hypothetical probability density with which the intersection point
            //     is sampled by explicit light sampling.
            float lightProb = 1.0f;
            if (shocker_plp.s->envLightTexture && shocker_plp.f->enableEnvLight)
                lightProb *= (1 - probToSampleEnvLight);
            const float instImportance = inst.lightGeomInstDist.integral();
            lightProb *= (pow2 (inst.uniformScale) * instImportance) / shocker_plp.s->lightInstDist.integral();
            lightProb *= geomInst.emitterPrimDist.integral() / instImportance;
            if (!stc::isfinite (lightProb))
            {
                *hypAreaPDensity = 0.0f;
                return;
            }
            lightProb *= geomInst.emitterPrimDist.evaluatePMF (primIndex);
            if constexpr (useSolidAngleSampling)
            {
                // TODO: ? compute in the local coordinates.
                const Vector3D A = normalize (pA - referencePoint);
                const Vector3D B = normalize (pB - referencePoint);
                const Vector3D C = normalize (pC - referencePoint);
                const Vector3D cAB = normalize (cross (A, B));
                const Vector3D cBC = normalize (cross (B, C));
                const Vector3D cCA = normalize (cross (C, A));
                const float cosAlpha = -dot (cAB, cCA);
                const float cosBeta = -dot (cBC, cAB);
                const float cosGamma = -dot (cCA, cBC);
                const float sphArea = std::acos (cosAlpha) + std::acos (cosBeta) + std::acos (cosGamma) - pi_v<float>;
                const float dirPDF = 1.0f / sphArea;
                Vector3D refDir = referencePoint - *positionInWorld;
                const float dist2ToRefPoint = sqLength (refDir);
                refDir /= std::sqrt (dist2ToRefPoint);
                const float lpCos = dot (refDir, *geometricNormalInWorld);
                if (lpCos > 0 && stc::isfinite (dirPDF))
                    *hypAreaPDensity = lightProb * (dirPDF * lpCos / dist2ToRefPoint);
                else
                    *hypAreaPDensity = 0.0f;
            }
            else
            {
                *hypAreaPDensity = lightProb / area;
            }
            Assert (stc::isfinite (*hypAreaPDensity), "hypP: %g, area: %g", *hypAreaPDensity, area);
        }
        else
        {
            (void)*hypAreaPDensity;
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