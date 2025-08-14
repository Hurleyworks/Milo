// some taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md
// and from Shocker GfxExp
// https://github.com/shocker-0x15/GfxEx

#include "principledDisney_shocker.h"

RT_PIPELINE_LAUNCH_PARAMETERS shocker_shared::PipelineLaunchParameters shocker_plp;

CUDA_DEVICE_KERNEL void RT_AH_NAME (visibility)()
{
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set (&visibility);
}

static constexpr bool useSolidAngleSampling = false;
static constexpr bool useImplicitLightSampling = true;
static constexpr bool useExplicitLightSampling = true;
static constexpr bool useMultipleImportanceSampling = useImplicitLightSampling && useExplicitLightSampling;
static_assert (useImplicitLightSampling || useExplicitLightSampling, "Invalid configuration for light sampling.");

template <typename RayType, bool withVisibility>
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performDirectLighting (
    const Point3D& shadingPoint, const Vector3D& vOutLocal, const ReferenceFrame& shadingFrame,
    const DisneyPrincipled& bsdf, const shocker_shared::LightSample& lightSample)
{
    return RGB (0.0f, 0.0f, 0.0f);
}


template <typename RayType>
CUDA_DEVICE_FUNCTION CUDA_INLINE bool evaluateVisibility (
    const Point3D& shadingPoint, const shocker_shared::LightSample& lightSample)
{
    using namespace shared;
    Vector3D shadowRayDir = lightSample.atInfinity ? Vector3D (lightSample.position) : (lightSample.position - shadingPoint);
    const float dist2 = shadowRayDir.sqLength();
    float dist = std::sqrt (dist2);
    shadowRayDir /= dist;
    if (lightSample.atInfinity)
        dist = 1e+10f;

    float visibility = 1.0f;
    VisibilityRayPayloadSignature::trace (
        shocker_plp.f->travHandle,
        shadingPoint.toNative(), shadowRayDir.toNative(), 0.0f, dist * 0.9999f, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        RayType::Visibility, maxNumRayTypes, RayType::Visibility,
        visibility);

    return visibility > 0.0f;
}


CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performNextEventEstimation (
    const Point3D& shadingPoint, const Vector3D& vOutLocal, const ReferenceFrame& shadingFrame,
    const DisneyPrincipled& bsdf, PCG32RNG& rng)
{
    RGB ret (0.0f);
    if constexpr (useExplicitLightSampling)
    {
        float uLight = rng.getFloat0cTo1o();
        bool selectEnvLight = false;
        float probToSampleCurLightType = 1.0f;
        if (shocker_plp.s->envLightTexture && shocker_plp.f->enableEnvLight)
        {
            if (shocker_plp.s->lightInstDist.integral() > 0.0f)
            {
                if (uLight < probToSampleEnvLight)
                {
                    probToSampleCurLightType = probToSampleEnvLight;
                    uLight /= probToSampleCurLightType;
                    selectEnvLight = true;
                }
                else
                {
                    probToSampleCurLightType = 1.0f - probToSampleEnvLight;
                    uLight = (uLight - probToSampleEnvLight) / probToSampleCurLightType;
                }
            }
            else
            {
                selectEnvLight = true;
            }
        }
        LightSample lightSample;
        float areaPDensity;
        sampleLight<useSolidAngleSampling> (
            shadingPoint,
            uLight, selectEnvLight, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
            &lightSample, &areaPDensity);
        areaPDensity *= probToSampleCurLightType;
        float misWeight = 1.0f;
        if constexpr (useMultipleImportanceSampling)
        {
            Vector3D shadowRay = lightSample.atInfinity ? Vector3D (lightSample.position) : (lightSample.position - shadingPoint);
            const float dist2 = shadowRay.sqLength();
            shadowRay /= std::sqrt (dist2);
            const Vector3D vInLocal = shadingFrame.toLocal (shadowRay);
            const float lpCos = std::fabs (dot (shadowRay, lightSample.normal));
            float bsdfPDensity = bsdf.evaluatePDF (vOutLocal, vInLocal) * lpCos / dist2;
            if (!stc::isfinite (bsdfPDensity))
                bsdfPDensity = 0.0f;
            const float lightPDensity = areaPDensity;
            misWeight = pow2 (lightPDensity) / (pow2 (bsdfPDensity) + pow2 (lightPDensity));
        }

        if (areaPDensity > 0.0f)
            ret = performDirectLighting<PathTracingRayType, true> (
                      shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample) *
                  (misWeight / areaPDensity);
    }

    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_rayGen_generic()
{
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = shocker_plp.f->bufferIndex;

    // Read GBuffer data
    const GBuffer0Elements gb0Elems = shocker_plp.s->GBuffer0[bufIdx].read (launchIndex);
    const GBuffer1Elements gb1Elems = shocker_plp.s->GBuffer1[bufIdx].read (launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric (gb0Elems.qbcB);
    const float bcC = decodeBarycentric (gb0Elems.qbcC);

    const PerspectiveCamera& camera = shocker_plp.f->camera;

    const bool useEnvLight = shocker_plp.s->envLightTexture && shocker_plp.f->enableEnvLight;
    RGB contribution (0.001f, 0.001f, 0.001f);
    if (instSlot != 0xFFFFFFFF)
    {
        const uint32_t geomInstSlot = gb0Elems.geomInstSlot;
        const InstanceData& inst = shocker_plp.s->instanceDataBufferArray[bufIdx][instSlot];
        const GeometryInstanceData& geomInst = shocker_plp.s->geometryInstanceDataBuffer[geomInstSlot];
        Point3D positionInWorld;
        Normal3D geometricNormalInWorld;
        Normal3D shadingNormalInWorld;
        Vector3D texCoord0DirInWorld;
        Point2D texCoord;
        computeSurfacePoint (
            inst, geomInst,
            gb0Elems.primIndex, bcB, bcC,
            &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
            &geometricNormalInWorld, &texCoord);

        RGB alpha (1.0f);
        const float initImportance = sRGB_calcLuminance (alpha);
        PCG32RNG rng = shocker_plp.s->rngBuffer.read (launchIndex);

        // EN: Shading on the first hit.
        Vector3D vIn;
        float dirPDensity;
        {
            // Get material slot from GBuffer1 (stored there from the gbuffer pass)
            const uint32_t materialSlot = gb1Elems.materialSlot;
            const shared::DisneyData& mat = shocker_plp.s->materialDataBuffer[materialSlot];

            // Debug output commented out - materialSlot is now working correctly
            // if (launchIndex.x == 512 && launchIndex.y == 384)
            // {
            //     printf("Center pixel: materialSlot from GBuffer = %u\n", materialSlot);
            // }

            const Vector3D vOut = normalize (camera.position - positionInWorld);
            const float frontHit = dot (vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
            // Offsetting assumes BRDF.
            positionInWorld = offsetRayOrigin (positionInWorld, frontHit * geometricNormalInWorld);

            ReferenceFrame shadingFrame (shadingNormalInWorld, texCoord0DirInWorld);
            if (shocker_plp.f->enableBumpMapping)
            {
                // const Normal3D modLocalNormal = mat.readModifiedNormal (mat.normal, mat.normalDimInfo, texCoord, 0.0f);
                // applyBumpMapping (modLocalNormal, &shadingFrame);
            }
            const Vector3D vOutLocal = shadingFrame.toLocal (vOut);

            // EN: Accumulate the contribution from a light source directly seeing.
            contribution = RGB (0.0f);
            if (vOutLocal.z > 0 && mat.emissive)
            {
                const float4 texValue = tex2DLod<float4> (mat.emissive, texCoord.x, texCoord.y, 0.0f);
                const RGB emittance (getXYZ (texValue));
                contribution += alpha * emittance / pi_v<float>;
            }

            // Create DisneyPrincipled instance directly instead of using BSDF
            DisneyPrincipled bsdf = DisneyPrincipled::create (
                mat, texCoord, 0.0f, shocker_plp.s->makeAllGlass, shocker_plp.s->globalGlassIOR,
                shocker_plp.s->globalTransmittanceDist, shocker_plp.s->globalGlassType);

            // Next event estimation (explicit light sampling) on the first hit.
            contribution += alpha * performNextEventEstimation (
                                        positionInWorld, vOutLocal, shadingFrame, bsdf, rng);

            // generate a next ray.
            Vector3D vInLocal;
            alpha *= bsdf.sampleThroughput (
                vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                &vInLocal, &dirPDensity);
            vIn = shadingFrame.fromLocal (vInLocal);
        }
    }
    else
    {
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (useEnvLight)
        {
            const float4 texValue = tex2DLod<float4> (shocker_plp.s->envLightTexture, bcB, bcC, 0.0f);
            const RGB luminance = shocker_plp.f->envLightPowerCoeff * RGB (getXYZ (texValue));
            contribution = luminance;
        }
    }

    RGB prevColorResult (0.0f, 0.0f, 0.0f);
    if (shocker_plp.f->numAccumFrames > 0)
        prevColorResult = RGB (getXYZ (shocker_plp.s->beautyAccumBuffer.read (launchIndex)));
    const float curWeight = 1.0f / (1 + shocker_plp.f->numAccumFrames);
    const RGB colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;
    shocker_plp.s->beautyAccumBuffer.write (launchIndex, make_float4 (colorResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_RG_NAME (pathTraceBaseline)()
{
    pathTrace_rayGen_generic();
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_closestHit_generic()
{
}

CUDA_DEVICE_KERNEL void RT_CH_NAME (pathTraceBaseline)()
{
    pathTrace_closestHit_generic();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME (pathTraceBaseline)()
{
    if constexpr (useImplicitLightSampling)
    {
        if (!shocker_plp.s->envLightTexture || !shocker_plp.f->enableEnvLight)
            return;

        PathTraceReadWritePayload* rwPayload;
        PathTraceRayPayloadSignature::get (nullptr, &rwPayload);

        const Vector3D rayDir = normalize (Vector3D (optixGetWorldRayDirection()));
        float posPhi, theta;
        toPolarYUp (rayDir, &posPhi, &theta);

        float phi = posPhi + shocker_plp.f->envLightRotation;
        phi = phi - floorf (phi / (2 * pi_v<float>)) * 2 * pi_v<float>;
        const Point2D texCoord (phi / (2 * pi_v<float>), theta / pi_v<float>);

        // Implicit Light Sampling
        const float4 texValue = tex2DLod<float4> (shocker_plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
        RGB luminance = shocker_plp.f->envLightPowerCoeff * RGB (getXYZ (texValue));
        float misWeight = 1.0f;
        if constexpr (useMultipleImportanceSampling)
        {
            const float uvPDF = shocker_plp.s->envLightImportanceMap.evaluatePDF (texCoord.x, texCoord.y);
            const float hypAreaPDensity = uvPDF / (2 * pi_v<float> * pi_v<float> * std::sin (theta));
            const float lightPDensity =
                (shocker_plp.s->lightInstDist.integral() > 0.0f ? probToSampleEnvLight : 1.0f) *
                hypAreaPDensity;
            const float bsdfPDensity = rwPayload->prevDirPDensity;
            misWeight = pow2 (bsdfPDensity) / (pow2 (bsdfPDensity) + pow2 (lightPDensity));
        }
        rwPayload->contribution += rwPayload->alpha * luminance * misWeight;
    }
}
