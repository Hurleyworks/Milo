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
    using namespace shared;
    Vector3D shadowRayDir = lightSample.atInfinity ? Vector3D (lightSample.position) : (lightSample.position - shadingPoint);
    const float dist2 = shadowRayDir.sqLength();
    float dist = std::sqrt (dist2);
    shadowRayDir /= dist;
    const Vector3D shadowRayDirLocal = shadingFrame.toLocal (shadowRayDir);

    const float lpCos = dot (-shadowRayDir, lightSample.normal);
    const float spCos = shadowRayDirLocal.z;
    
    // Debug for center pixel
    bool debugDL = (optixGetLaunchIndex().x == 512 && optixGetLaunchIndex().y == 384);

    float visibility = 1.0f;
    if constexpr (withVisibility)
    {
        if (lightSample.atInfinity)
            dist = 1e+10f;
        shocker_shared::VisibilityRayPayloadSignature::trace (
            shocker_plp.f->travHandle,
            shadingPoint.toNative(), shadowRayDir.toNative(), 0.0f, dist * 0.9999f, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            RayType::Visibility, shocker_shared::maxNumRayTypes, RayType::Visibility,
            visibility);
    }

    if (debugDL) {
        printf("DL: shadowRayDir=(%.3f,%.3f,%.3f), shadowRayDirLocal=(%.3f,%.3f,%.3f)\n",
            shadowRayDir.x, shadowRayDir.y, shadowRayDir.z,
            shadowRayDirLocal.x, shadowRayDirLocal.y, shadowRayDirLocal.z);
        printf("DL: vis=%.1f, lpCos=%.3f, spCos=%.3f, dist2=%.3f, atInf=%d\n",
            visibility, lpCos, spCos, dist2, lightSample.atInfinity);
    }
    
    if (visibility > 0 && lpCos > 0)
    {
        const RGB Le = lightSample.emittance / pi_v<float>; // assume diffuse emitter.
        
        // Check if light is below surface - if so, return 0
        if (spCos <= 0) {
            if (debugDL) {
                printf("DL: Light below surface, spCos=%.3f, returning 0\n", spCos);
            }
            return RGB (0.0f, 0.0f, 0.0f);
        }
        
        const RGB fsValue = bsdf.evaluate (vOutLocal, shadowRayDirLocal);
        const float G = lpCos * std::fabs (spCos) / dist2;
        const RGB ret = fsValue * Le * G;
        
        if (debugDL) {
            printf("DL ret: Le=(%.1f,%.1f,%.1f), fs=(%.3f,%.3f,%.3f), G=%.6f, ret=(%.3f,%.3f,%.3f)\n",
                Le.r, Le.g, Le.b,
                fsValue.r, fsValue.g, fsValue.b,
                G,
                ret.r, ret.g, ret.b);
        }
        
        return ret;
    }
    else
    {
        if (debugDL) {
            printf("DL: returning 0 (vis=%.1f, lpCos=%.3f)\n", visibility, lpCos);
        }
        return RGB (0.0f, 0.0f, 0.0f);
    }
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
        
        // Debug for center pixel
        bool debugNEE = (optixGetLaunchIndex().x == 512 && optixGetLaunchIndex().y == 384);
        
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

        if (debugNEE) {
            printf("NEE: selectEnv=%d, areaPDF=%.6f, lightEmit=(%.3f,%.3f,%.3f)\n",
                selectEnvLight, areaPDensity, 
                lightSample.emittance.r, lightSample.emittance.g, lightSample.emittance.b);
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
    
    // Debug flag for center pixel
    const bool debugPixel = (launchIndex.x == 512 && launchIndex.y == 384);

    // Read GBuffer data
    const GBuffer0Elements gb0Elems = shocker_plp.s->GBuffer0[bufIdx].read (launchIndex);
    const GBuffer1Elements gb1Elems = shocker_plp.s->GBuffer1[bufIdx].read (launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric (gb0Elems.qbcB);
    const float bcC = decodeBarycentric (gb0Elems.qbcC);

    const PerspectiveCamera& camera = shocker_plp.f->camera;

    const bool useEnvLight = shocker_plp.s->envLightTexture && shocker_plp.f->enableEnvLight;
    RGB contribution (0.0f, 0.0f, 0.0f);
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
        
        // Debug: check if normal is valid before using it
        if (debugPixel) {
            printf("After computeSurfacePoint: shadingNormal=(%.3f,%.3f,%.3f), geoNormal=(%.3f,%.3f,%.3f)\n",
                shadingNormalInWorld.x, shadingNormalInWorld.y, shadingNormalInWorld.z,
                geometricNormalInWorld.x, geometricNormalInWorld.y, geometricNormalInWorld.z);
        }

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

            ReferenceFrame shadingFrame (shadingNormalInWorld, texCoord0DirInWorld);
            if (shocker_plp.f->enableBumpMapping)
            {
                // const Normal3D modLocalNormal = mat.readModifiedNormal (mat.normal, mat.normalDimInfo, texCoord, 0.0f);
                // applyBumpMapping (modLocalNormal, &shadingFrame);
            }
            // Offsetting assumes BRDF - do this AFTER setting up the shading frame
            positionInWorld = offsetRayOrigin (positionInWorld, frontHit * geometricNormalInWorld);
            const Vector3D vOutLocal = shadingFrame.toLocal (vOut);
            
            if (debugPixel) {
                printf("Geometry: vOut=(%.3f,%.3f,%.3f), vOutLocal=(%.3f,%.3f,%.3f), normal=(%.3f,%.3f,%.3f)\n",
                    vOut.x, vOut.y, vOut.z,
                    vOutLocal.x, vOutLocal.y, vOutLocal.z,
                    shadingNormalInWorld.x, shadingNormalInWorld.y, shadingNormalInWorld.z);
                printf("TexCoord0Dir=(%.3f,%.3f,%.3f)\n",
                    texCoord0DirInWorld.x, texCoord0DirInWorld.y, texCoord0DirInWorld.z);
                // Check the reference frame basis
                Vector3D testZ(0, 0, 1);
                Vector3D testZLocal = shadingFrame.toLocal(testZ);
                printf("ReferenceFrame test: (0,0,1) -> (%.3f,%.3f,%.3f)\n",
                    testZLocal.x, testZLocal.y, testZLocal.z);
            }

            // EN: Accumulate the contribution from a light source directly seeing.
            contribution = RGB (0.0f);
          /*  if (vOutLocal.z > 0 && mat.emissive)
            {
                const float4 texValue = tex2DLod<float4> (mat.emissive, texCoord.x, texCoord.y, 0.0f);
                const RGB emittance (getXYZ (texValue));
                contribution += alpha * emittance / pi_v<float>;
            }*/

            // Create DisneyPrincipled instance directly instead of using BSDF
            DisneyPrincipled bsdf = DisneyPrincipled::create (
                mat, texCoord, 0.0f, shocker_plp.s->makeAllGlass, shocker_plp.s->globalGlassIOR,
                shocker_plp.s->globalTransmittanceDist, shocker_plp.s->globalGlassType);

            // Next event estimation (explicit light sampling) on the first hit.
            RGB neeContrib = performNextEventEstimation (
                                        positionInWorld, vOutLocal, shadingFrame, bsdf, rng);
            contribution += alpha * neeContrib;
            
            if (debugPixel) {
                printf("First hit NEE: contrib=(%.3f,%.3f,%.3f), alpha=(%.3f,%.3f,%.3f), nee=(%.3f,%.3f,%.3f)\n",
                    contribution.r, contribution.g, contribution.b,
                    alpha.r, alpha.g, alpha.b,
                    neeContrib.r, neeContrib.g, neeContrib.b);
            }

            // generate a next ray.
            Vector3D vInLocal;
            RGB throughput = bsdf.sampleThroughput (
                vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
                &vInLocal, &dirPDensity);
            
            if (debugPixel) {
                printf("First hit sampleThroughput: throughput=(%.3f,%.3f,%.3f), dirPDF=%.6f, vInLocal.z=%.3f\n",
                    throughput.r, throughput.g, throughput.b, dirPDensity, vInLocal.z);
            }
            
            // Apply the same calculation as RiPR: multiply by cosine and divide by PDF
            if (dirPDensity > 0.0f && stc::isfinite(dirPDensity)) {
                alpha *= throughput * std::fabs(vInLocal.z) / dirPDensity;
            } else {
                alpha = RGB(0.0f);
                dirPDensity = 0.0f;
            }
            
            if (debugPixel) {
                printf("After alpha update: alpha=(%.3f,%.3f,%.3f)\n",
                    alpha.r, alpha.g, alpha.b);
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
            if (!isValidSampling) {
                if (debugPixel) {
                    printf("Path terminated: invalid sampling, prevDirPDF=%.6f\n", rwPayload.prevDirPDensity);
                }
                break;
            }

            ++rwPayload.pathLength;
            
            if (debugPixel) {
                printf("Path bounce %d: alpha=(%.3f,%.3f,%.3f), contrib=(%.3f,%.3f,%.3f)\n",
                    rwPayload.pathLength,
                    rwPayload.alpha.r, rwPayload.alpha.g, rwPayload.alpha.b,
                    rwPayload.contribution.r, rwPayload.contribution.g, rwPayload.contribution.b);
            }
            if (rwPayload.pathLength >= shocker_plp.f->maxPathLength)
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
                shocker_plp.f->travHandle, rayOrg.toNative(), rayDir.toNative(),
                0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                pathTraceRayType, maxNumRayTypes, pathTraceRayType,
                woPayloadPtr, rwPayloadPtr);
            if (rwPayload.terminate)
                break;
            rayOrg = woPayload.nextOrigin;
            rayDir = woPayload.nextDirection;
        }
        contribution = rwPayload.contribution;
        
        if (debugPixel) {
            printf("After path loop: final contribution=(%.3f,%.3f,%.3f)\n",
                contribution.r, contribution.g, contribution.b);
        }

        shocker_plp.s->rngBuffer.write (launchIndex, rwPayload.rng);
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
    
    if (debugPixel) {
        printf("Final accumulation: prevColor=(%.3f,%.3f,%.3f), curWeight=%.3f, final=(%.3f,%.3f,%.3f), numFrames=%d\n",
            prevColorResult.r, prevColorResult.g, prevColorResult.b,
            curWeight,
            colorResult.r, colorResult.g, colorResult.b,
            shocker_plp.f->numAccumFrames);
    }
    
    shocker_plp.s->beautyAccumBuffer.write (launchIndex, make_float4 (colorResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_RG_NAME (pathTraceBaseline)()
{
    pathTrace_rayGen_generic();
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_closestHit_generic()
{
    const uint32_t bufIdx = shocker_plp.f->bufferIndex;
    const auto sbtr = HitGroupSBTRecordData::get();
    const InstanceData& inst = shocker_plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];
    const GeometryInstanceData& geomInst = shocker_plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

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


    const shared::DisneyData& mat = shocker_plp.s->materialDataBuffer[sbtr.materialSlot];

    const Vector3D vOut = normalize (-Vector3D (optixGetWorldRayDirection()));
    const float frontHit = dot (vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;

    ReferenceFrame shadingFrame (shadingNormalInWorld, texCoord0DirInWorld);
    if (shocker_plp.f->enableBumpMapping)
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
           /* const float4 texValue = tex2DLod<float4> (mat.emissive, texCoord.x, texCoord.y, 0.0f);
            const RGB emittance (getXYZ (texValue));
            float misWeight = 1.0f;
            if constexpr (useMultipleImportanceSampling)
            {
                const float dist2 = sqDistance (rayOrigin, positionInWorld);
                const float lightPDensity = hypAreaPDensity * dist2 / vOutLocal.z;
                const float bsdfPDensity = rwPayload->prevDirPDensity;
                misWeight = pow2 (bsdfPDensity) / (pow2 (bsdfPDensity) + pow2 (lightPDensity));
            }
            rwPayload->contribution += rwPayload->alpha * emittance * (misWeight / pi_v<float>);*/
        }

        // Russian roulette
        const float continueProb = std::fmin (sRGB_calcLuminance (rwPayload->alpha) / rwPayload->initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb || rwPayload->maxLengthTerminate)
            return;
        rwPayload->alpha /= continueProb;
    }

     // Create DisneyPrincipled instance directly instead of using BSDF
    DisneyPrincipled bsdf = DisneyPrincipled::create (
        mat, texCoord, 0.0f, shocker_plp.s->makeAllGlass, shocker_plp.s->globalGlassIOR,
        shocker_plp.s->globalTransmittanceDist, shocker_plp.s->globalGlassType);

    RGB neeContribCH = performNextEventEstimation (
                                                      positionInWorld, vOutLocal, shadingFrame, bsdf, rng);
    RGB addedContribCH = rwPayload->alpha * neeContribCH;
    rwPayload->contribution += addedContribCH;
    
    // Debug output for specific pixel
    if (optixGetLaunchIndex().x == 512 && optixGetLaunchIndex().y == 384) {
        printf("CH NEE: nee=(%.3f,%.3f,%.3f), alpha=(%.3f,%.3f,%.3f), added=(%.3f,%.3f,%.3f), total=(%.3f,%.3f,%.3f)\n",
            neeContribCH.r, neeContribCH.g, neeContribCH.b,
            rwPayload->alpha.r, rwPayload->alpha.g, rwPayload->alpha.b,
            addedContribCH.r, addedContribCH.g, addedContribCH.b,
            rwPayload->contribution.r, rwPayload->contribution.g, rwPayload->contribution.b);
    }

    // generate a next ray.
    Vector3D vInLocal;
    float dirPDensity;
    RGB throughput = bsdf.sampleThroughput (
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);
    
    // Debug output for specific pixel
    if (optixGetLaunchIndex().x == 512 && optixGetLaunchIndex().y == 384) {
        printf("CH sampleThroughput: throughput=(%.3f,%.3f,%.3f), dirPDF=%.6f, vInLocal.z=%.3f\n",
            throughput.r, throughput.g, throughput.b, dirPDensity, vInLocal.z);
    }
    
    // Apply the same calculation as RiPR: multiply by cosine and divide by PDF
    if (dirPDensity > 0.0f && stc::isfinite(dirPDensity)) {
        rwPayload->alpha *= throughput * std::fabs(vInLocal.z) / dirPDensity;
    } else {
        rwPayload->alpha = RGB(0.0f);
        dirPDensity = 0.0f;
    }
    
    // Debug output for specific pixel
    if (optixGetLaunchIndex().x == 512 && optixGetLaunchIndex().y == 384) {
        printf("CH After alpha update: alpha=(%.3f,%.3f,%.3f)\n",
            rwPayload->alpha.r, rwPayload->alpha.g, rwPayload->alpha.b);
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
        if (!shocker_plp.s->envLightTexture || !shocker_plp.f->enableEnvLight)
            return;

        PathTraceReadWritePayload* rwPayload;
        PathTraceRayPayloadSignature::get (nullptr, &rwPayload);
        
        // Debug for center pixel
        if (optixGetLaunchIndex().x == 512 && optixGetLaunchIndex().y == 384) {
            printf("MISS SHADER HIT! alpha=(%.3f,%.3f,%.3f)\n",
                rwPayload->alpha.r, rwPayload->alpha.g, rwPayload->alpha.b);
        }

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
