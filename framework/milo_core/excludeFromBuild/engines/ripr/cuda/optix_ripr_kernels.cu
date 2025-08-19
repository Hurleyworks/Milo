// some taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md
// and from RiPR GfxExp
// https://github.com/shocker-0x15/GfxEx



#include "principledDisney_ripr.h"

RT_PIPELINE_LAUNCH_PARAMETERS ripr_shared::PipelineLaunchParameters ripr_plp;

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
    const DisneyPrincipled& bsdf, const ripr_shared::LightSample& lightSample)
{
    using namespace shared;
    Vector3D shadowRayDir = lightSample.atInfinity ? Vector3D (lightSample.position) : (lightSample.position - shadingPoint);
    const float dist2 = shadowRayDir.sqLength();
    float dist = std::sqrt (dist2);
    shadowRayDir /= dist;
    const Vector3D shadowRayDirLocal = shadingFrame.toLocal (shadowRayDir);

    const float lpCos = dot (-shadowRayDir, lightSample.normal);
    const float spCos = shadowRayDirLocal.z;

    float visibility = 1.0f;
    if constexpr (withVisibility)
    {
        if (lightSample.atInfinity)
            dist = 1e+10f;
        ripr_shared::VisibilityRayPayloadSignature::trace (
            ripr_plp.f->travHandle,
            shadingPoint.toNative(), shadowRayDir.toNative(), 0.0f, dist * 0.9999f, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            RayType::Visibility, ripr_shared::maxNumRayTypes, RayType::Visibility,
            visibility);
    }

    if (visibility > 0 && lpCos > 0)
    {
        const RGB Le = lightSample.emittance / pi_v<float>; // assume diffuse emitter.
        const RGB fsValue = bsdf.evaluate (vOutLocal, shadowRayDirLocal);
        const float G = lpCos * std::fabs (spCos) / dist2;
        const RGB ret = fsValue * Le * G;
        return ret;
    }
    else
    {
        return RGB (0.0f, 0.0f, 0.0f);
    }
}

template <typename RayType>
CUDA_DEVICE_FUNCTION CUDA_INLINE bool evaluateVisibility (
    const Point3D& shadingPoint, const ripr_shared::LightSample& lightSample)
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
        ripr_plp.f->travHandle,
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

        if (ripr_plp.s->envLightTexture && ripr_plp.f->enableEnvLight)
        {
            if (ripr_plp.s->lightInstDist.integral() > 0.0f)
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
    const uint32_t bufIdx = ripr_plp.f->bufferIndex;

    // Read GBuffer data
    const GBuffer0Elements gb0Elems = ripr_plp.s->GBuffer0[bufIdx].read (launchIndex);
    const GBuffer1Elements gb1Elems = ripr_plp.s->GBuffer1[bufIdx].read (launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric (gb0Elems.qbcB);
    const float bcC = decodeBarycentric (gb0Elems.qbcC);
    

    const PerspectiveCamera& camera = ripr_plp.f->camera;

    const bool useEnvLight = ripr_plp.s->envLightTexture && ripr_plp.f->enableEnvLight;
    RGB contribution (0.0f, 0.0f, 0.0f);
    if (instSlot != 0xFFFFFFFF)
    {
        const uint32_t geomInstSlot = gb0Elems.geomInstSlot;
        const InstanceData& inst = ripr_plp.s->instanceDataBufferArray[bufIdx][instSlot];
        const GeometryInstanceData& geomInst = ripr_plp.s->geometryInstanceDataBuffer[geomInstSlot];


        Point3D positionInWorld;
        Normal3D geometricNormalInWorld;
        Normal3D shadingNormalInWorld;
        Vector3D texCoord0DirInWorld;
        Point2D texCoord;
        // Use the OptiX transform version of computeSurfacePoint for consistency with closest hit
        // This uses transformPointFromObjectToWorldSpace instead of inst.transform

        computeSurfacePoint (
            inst, geomInst,
            gb0Elems.primIndex, bcB, bcC,
            &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
            &geometricNormalInWorld, &texCoord);

        RGB alpha (1.0f);
        const float initImportance = sRGB_calcLuminance (alpha);
        PCG32RNG rng = ripr_plp.s->rngBuffer.read (launchIndex);

        // EN: Shading on the first hit.
        Vector3D vIn;
        float dirPDensity;
        {
            // Get material slot from GBuffer1 (stored there from the gbuffer pass)
            const uint32_t materialSlot = gb1Elems.materialSlot;
            const shared::DisneyData& mat = ripr_plp.s->materialDataBuffer[materialSlot];

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
            if (ripr_plp.f->enableBumpMapping)
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
                mat, texCoord, 0.0f, ripr_plp.s->makeAllGlass, ripr_plp.s->globalGlassIOR,
                ripr_plp.s->globalTransmittanceDist, ripr_plp.s->globalGlassType);

             // Next event estimation (explicit light sampling) on the first hit.
            contribution += alpha * performNextEventEstimation (
                                        positionInWorld, vOutLocal, shadingFrame, bsdf, rng);

            // generate a next ray.
            Vector3D vInLocal;
            RGB throughput = bsdf.sampleThroughput (
                vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                &vInLocal, &dirPDensity);

            // Apply the same calculation as RiPR: multiply by cosine and divide by PDF
            if (dirPDensity > 0.0f && stc::isfinite (dirPDensity))
            {
                alpha *= throughput * std::fabs (vInLocal.z) / dirPDensity;
            }
            else
            {
                alpha = RGB (0.0f);
                dirPDensity = 0.0f;
            }

            vIn = shadingFrame.fromLocal (vInLocal);
        }
        // Path extension loop
        PathTraceWriteOnlyPayload woPayload = {};
        PathTraceWriteOnlyPayload* woPayloadPtr = &woPayload;
        PathTraceReadWritePayload rwPayload = {};
        PathTraceReadWritePayload* rwPayloadPtr = &rwPayload;
        rwPayload.rng = rng;
        rwPayload.initImportance = initImportance;
        rwPayload.alpha = alpha;
        rwPayload.prevDirPDensity = dirPDensity;
        rwPayload.contribution = contribution;
        rwPayload.pathLength = 1;
        Point3D rayOrg = positionInWorld;
        Vector3D rayDir = vIn;
        while (true)
        {
            const bool isValidSampling = rwPayload.prevDirPDensity > 0.0f && stc::isfinite (rwPayload.prevDirPDensity);
            if (!isValidSampling)
                break;

            ++rwPayload.pathLength;
            if (rwPayload.pathLength >= ripr_plp.f->maxPathLength)
                rwPayload.maxLengthTerminate = true;
            rwPayload.terminate = true;

            // EN: Nothing to do in the closest-hit program when reaching the path length limit
            //     in the case implicit light sampling is unused.
            if constexpr (!useImplicitLightSampling)
            {
                if (rwPayload.maxLengthTerminate)
                    break;
                // Russian roulette
                const float continueProb =
                    std::fmin (sRGB_calcLuminance (rwPayload.alpha) / rwPayload.initImportance, 1.0f);
                if (rwPayload.rng.getFloat0cTo1o() >= continueProb)
                    break;
                rwPayload.alpha /= continueProb;
            }

            constexpr PathTracingRayType pathTraceRayType = PathTracingRayType::Closest;
            PathTraceRayPayloadSignature::trace (
                ripr_plp.f->travHandle, rayOrg.toNative(), rayDir.toNative(),
                0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                pathTraceRayType, maxNumRayTypes, pathTraceRayType,
                woPayloadPtr, rwPayloadPtr);
            if (rwPayload.terminate)
                break;
            rayOrg = woPayload.nextOrigin;
            rayDir = woPayload.nextDirection;
        }
        contribution = rwPayload.contribution;

        ripr_plp.s->rngBuffer.write (launchIndex, rwPayload.rng);
    }
    else
    {
        // EN: Accumulate the contribution from the environmental light source directly seeing.
        if (useEnvLight)
        {
            const float4 texValue = tex2DLod<float4> (ripr_plp.s->envLightTexture, bcB, bcC, 0.0f);
            const RGB luminance = ripr_plp.f->envLightPowerCoeff * RGB (getXYZ (texValue));
            contribution = luminance;
        }
    }

    RGB prevColorResult (0.0f, 0.0f, 0.0f);
    if (ripr_plp.f->numAccumFrames > 0)
        prevColorResult = RGB (getXYZ (ripr_plp.s->beautyAccumBuffer.read (launchIndex)));
    const float curWeight = 1.0f / (1 + ripr_plp.f->numAccumFrames);
    const RGB colorResult = (1 - curWeight) * prevColorResult + curWeight * contribution;

    ripr_plp.s->beautyAccumBuffer.write (launchIndex, make_float4 (colorResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_RG_NAME (pathTraceBaseline)()
{
    pathTrace_rayGen_generic();
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_closestHit_generic()
{
    const uint32_t bufIdx = ripr_plp.f->bufferIndex;
    const auto sbtr = HitGroupSBTRecordData::get();
    const uint32_t instanceId = optixGetInstanceId();
    const InstanceData& inst = ripr_plp.s->instanceDataBufferArray[bufIdx][instanceId];
    const GeometryInstanceData& geomInst = ripr_plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    PathTraceWriteOnlyPayload* woPayload;
    PathTraceReadWritePayload* rwPayload;
    PathTraceRayPayloadSignature::get (&woPayload, &rwPayload);
    PCG32RNG& rng = rwPayload->rng;

    const Point3D rayOrigin (optixGetWorldRayOrigin());

    const auto hp = HitPointParameter::get();
    Point3D positionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Normal3D geometricNormalInWorld;
    Point2D texCoord;
    float hypAreaPDensity;
    computeSurfacePoint<useMultipleImportanceSampling, useSolidAngleSampling> (
        inst, geomInst, hp.primIndex, hp.bcB, hp.bcC,
        rayOrigin,
        &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
        &geometricNormalInWorld, &texCoord, &hypAreaPDensity);
    if constexpr (!useMultipleImportanceSampling)
        (void)hypAreaPDensity;

    const shared::DisneyData& mat = ripr_plp.s->materialDataBuffer[sbtr.materialSlot];

    const Vector3D vOut = normalize (-Vector3D (optixGetWorldRayDirection()));
    const float frontHit = dot (vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame (shadingNormalInWorld, texCoord0DirInWorld);
    if (ripr_plp.f->enableBumpMapping)
    {
        //  const Normal3D modLocalNormal = mat.readModifiedNormal (mat.normal, mat.normalDimInfo, texCoord, 0.0f);
        //  applyBumpMapping (modLocalNormal, &shadingFrame);
    }
    positionInWorld = offsetRayOrigin (positionInWorld, frontHit * geometricNormalInWorld);
    const Vector3D vOutLocal = shadingFrame.toLocal (vOut);

    if constexpr (useImplicitLightSampling)
    {
        // Implicit Light Sampling
        if (vOutLocal.z > 0 && mat.emissive)
        {
             const float4 texValue = tex2DLod<float4> (mat.emissive, texCoord.x, texCoord.y, 0.0f);
             const RGB emittance (getXYZ (texValue));
             float misWeight = 1.0f;
             if constexpr (useMultipleImportanceSampling)
             {
                 const float dist2 = sqDistance (rayOrigin, positionInWorld);
                 const float lightPDensity = hypAreaPDensity * dist2 / vOutLocal.z;
                 const float bsdfPDensity = rwPayload->prevDirPDensity;
                 misWeight = pow2 (bsdfPDensity) / (pow2 (bsdfPDensity) + pow2 (lightPDensity));
             }
             rwPayload->contribution += rwPayload->alpha * emittance * (misWeight / pi_v<float>);
        }

        // Russian roulette
        const float continueProb = std::fmin (sRGB_calcLuminance (rwPayload->alpha) / rwPayload->initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb || rwPayload->maxLengthTerminate)
            return;
        rwPayload->alpha /= continueProb;
    }

    // Create DisneyPrincipled instance directly instead of using BSDF
    DisneyPrincipled bsdf = DisneyPrincipled::create (
        mat, texCoord, 0.0f, ripr_plp.s->makeAllGlass, ripr_plp.s->globalGlassIOR,
        ripr_plp.s->globalTransmittanceDist, ripr_plp.s->globalGlassType);

    // Next Event Estimation (Explicit Light Sampling)
    rwPayload->contribution += rwPayload->alpha * performNextEventEstimation (
                                                      positionInWorld, vOutLocal, shadingFrame, bsdf, rng);

    // generate a next ray.
    Vector3D vInLocal;
    float dirPDensity;
    RGB throughput = bsdf.sampleThroughput (
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);

    // Apply the same calculation as RiPR: multiply by cosine and divide by PDF
    if (dirPDensity > 0.0f && stc::isfinite (dirPDensity))
    {
        rwPayload->alpha *= throughput * std::fabs (vInLocal.z) / dirPDensity;
    }
    else
    {
        rwPayload->alpha = RGB (0.0f);
        dirPDensity = 0.0f;
    }

    const Vector3D vIn = shadingFrame.fromLocal (vInLocal);

    woPayload->nextOrigin = positionInWorld;
    woPayload->nextDirection = vIn;
    rwPayload->prevDirPDensity = dirPDensity;
    rwPayload->terminate = false;
}

CUDA_DEVICE_KERNEL void RT_CH_NAME (pathTraceBaseline)()
{
    pathTrace_closestHit_generic();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME (pathTraceBaseline)()
{
    if constexpr (useImplicitLightSampling)
    {
        if (!ripr_plp.s->envLightTexture || !ripr_plp.f->enableEnvLight)
            return;

        PathTraceReadWritePayload* rwPayload;
        PathTraceRayPayloadSignature::get (nullptr, &rwPayload);

        const Vector3D rayDir = normalize (Vector3D (optixGetWorldRayDirection()));
        float posPhi, theta;
        toPolarYUp (rayDir, &posPhi, &theta);

        float phi = posPhi + ripr_plp.f->envLightRotation;
        phi = phi - floorf (phi / (2 * pi_v<float>)) * 2 * pi_v<float>;
        const Point2D texCoord (phi / (2 * pi_v<float>), theta / pi_v<float>);

        // Implicit Light Sampling
        const float4 texValue = tex2DLod<float4> (ripr_plp.s->envLightTexture, texCoord.x, texCoord.y, 0.0f);
        RGB luminance = ripr_plp.f->envLightPowerCoeff * RGB (getXYZ (texValue));
        float misWeight = 1.0f;
        if constexpr (useMultipleImportanceSampling)
        {
            // Clamp texture coordinates to avoid floating-point precision issues in evaluatePDF
            float u = fmin(texCoord.x, 0.999999f);
            float v = fmin(texCoord.y, 0.999999f);
            const float uvPDF = ripr_plp.s->envLightImportanceMap.evaluatePDF (u, v);
            const float hypAreaPDensity = uvPDF / (2 * pi_v<float> * pi_v<float> * std::sin (theta));
            const float lightPDensity =
                (ripr_plp.s->lightInstDist.integral() > 0.0f ? probToSampleEnvLight : 1.0f) *
                hypAreaPDensity;
            const float bsdfPDensity = rwPayload->prevDirPDensity;
            misWeight = pow2 (bsdfPDensity) / (pow2 (bsdfPDensity) + pow2 (lightPDensity));
        }
        rwPayload->contribution += rwPayload->alpha * luminance * misWeight;
    }
}
