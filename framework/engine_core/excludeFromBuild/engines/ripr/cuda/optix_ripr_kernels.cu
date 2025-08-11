// optix_ripr_kernels.cu
// OptiX ray generation and hit programs for the RiPR engine path tracing
// STUB VERSION - no implementation

#include "principledDisney_ripr.h"
#include "../ripr_shared.h"

using namespace ripr_shared;

// Global declaration of pipeline launch parameters - must match the name in pipeline configuration
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

template <bool useSolidAngleSampling>
CUDA_DEVICE_FUNCTION CUDA_INLINE void sampleLight (
    const Point3D& shadingPoint,
    float ul, bool sampleEnvLight, float u0, float u1,
    ripr_shared::LightSample* lightSample, float* areaPDensity)
{
    using namespace shared;
    CUtexObject texEmittance = 0;
    RGB emittance (0.0f, 0.0f, 0.0f);
    Point2D texCoord;
    if (sampleEnvLight)
    {
        float u, v;
        float uvPDF;
        ripr_plp.s->envLightImportanceMap.sample (u0, u1, &u, &v, &uvPDF);
        const float phi = 2 * pi_v<float> * u;
        const float theta = pi_v<float> * v;

        float posPhi = phi - ripr_plp.f->envLightRotation;
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

        texEmittance = ripr_plp.s->envLightTexture;

        // EN: Multiply a coefficient to make the return value possible to be handled as luminous emittance.
        emittance = RGB (pi_v<float> * ripr_plp.f->envLightPowerCoeff);
        texCoord.x = u;
        texCoord.y = v;
    }
    else
    {
        float lightProb = 1.0f;

        // EN: First, sample an instance.
        float instProb;
        float uGeomInst;
        const uint32_t instSlot = ripr_plp.s->lightInstDist.sample (ul, &instProb, &uGeomInst);
        lightProb *= instProb;
        const ripr::RiPRNodeData& inst = ripr_plp.s->instanceDataBufferArray[ripr_plp.f->bufferIndex][instSlot];
        if (instProb == 0.0f)
        {
            *areaPDensity = 0.0f;
            return;
        }
        // Assert(inst.lightGeomInstDist.integral() > 0.0f,
        //        "Non-emissive inst %u, prob %g, u: %g(0x%08x).", instIndex, instProb, ul, *(uint32_t*)&ul);

        // EN: Next, sample a geometry instance which belongs to the sampled instance.
        float geomInstProb;
        float uPrim;
        const uint32_t geomInstIndexInInst = inst.lightGeomInstDist.sample (uGeomInst, &geomInstProb, &uPrim);
        const uint32_t geomInstSlot = inst.geomInstSlots[geomInstIndexInInst];
        lightProb *= geomInstProb;
        const ripr::RiPRSurfaceData& geomInst = ripr_plp.s->geometryInstanceDataBuffer[geomInstSlot];
        if (geomInstProb == 0.0f)
        {
            *areaPDensity = 0.0f;
            return;
        }
        // Assert(geomInst.emitterPrimDist.integral() > 0.0f,
        //        "Non-emissive geom inst %u, prob %g, u: %g.", geomInstIndex, geomInstProb, uGeomInst);

        // EN: Finally, sample a primitive which belongs to the sampled geometry instance.
        float primProb;
        const uint32_t primIndex = geomInst.emitterPrimDist.sample (uPrim, &primProb);
        lightProb *= primProb;

        // printf("%u-%u-%u: %g\n", instIndex, geomInstIndex, primIndex, lightProb);

        const shared::DisneyData& mat = ripr_plp.s->disneyMaterialBuffer[geomInst.disneyMaterialSlot];

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
            // Uniform sampling in solid angle subtended by the triangle for the shading point.
            float dist;
            Vector3D dir;
            float dirPDF;
            {
                const auto project = [] (const Vector3D& vA, const Vector3D& vB)
                {
                    return normalize (vA - dot (vA, vB) * vB);
                };

                // TODO: ? compute in the local coordinates.
                const Vector3D A = normalize (pA - shadingPoint);
                const Vector3D B = normalize (pB - shadingPoint);
                const Vector3D C = normalize (pC - shadingPoint);
                const Vector3D cAB = normalize (cross (A, B));
                const Vector3D cBC = normalize (cross (B, C));
                const Vector3D cCA = normalize (cross (C, A));
                // float cos_a = dot(B, C);
                // float cos_b = dot(C, A);
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

                const auto restoreBarycentrics = [&geomNormal] (const Point3D& org, const Vector3D& dir,
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
            // A Low-Distortion Map Between Triangle and Square
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

CUDA_DEVICE_KERNEL void RT_RG_NAME (raygen)()
{
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = ripr_plp.f->bufferIndex;

    const GBuffer0Elements gb0Elems = ripr_plp.s->GBuffer0[bufIdx].read (launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric (gb0Elems.qbcB);
    const float bcC = decodeBarycentric (gb0Elems.qbcC);

    const PerspectiveCamera& camera = ripr_plp.f->camera;

    const bool useEnvLight = ripr_plp.s->envLightTexture && ripr_plp.f->enableEnvLight;
    RGB contribution (0.001f, 0.001f, 0.001f);

    if (instSlot != 0xFFFFFFFF)
    {
        const uint32_t geomInstSlot = gb0Elems.geomInstSlot;
        const ripr::RiPRNodeData& inst = ripr_plp.s->instanceDataBufferArray[bufIdx][instSlot];
        const ripr::RiPRSurfaceData& geomInst = ripr_plp.s->geometryInstanceDataBuffer[geomInstSlot];
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
        PCG32RNG rng = ripr_plp.s->rngBuffer.read (launchIndex);

        Vector3D vIn;
        float dirPDensity;
        {
            const shared::DisneyData& mat = ripr_plp.s->disneyMaterialBuffer[geomInst.disneyMaterialSlot];

            const Vector3D vOut = normalize (camera.position - positionInWorld);
            const float frontHit = dot (vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
            // Offsetting assumes BRDF.
            positionInWorld = offsetRayOrigin (positionInWorld, frontHit * geometricNormalInWorld);

            ReferenceFrame shadingFrame (shadingNormalInWorld, texCoord0DirInWorld);
            if (ripr_plp.f->enableBumpMapping)
            {
                // FIXME
                // const Normal3D modLocalNormal = mat.readModifiedNormal (mat.normal, mat.normalDimInfo, texCoord, 0.0f);
                // applyBumpMapping (modLocalNormal, &shadingFrame);
            }
            const Vector3D vOutLocal = shadingFrame.toLocal (vOut);

            // JP: ??????????????????
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
                mat, texCoord, 0.0f);
            //  bsdf.setup (mat, texCoord, 0.0f);

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

CUDA_DEVICE_KERNEL void RT_CH_NAME (shading)()
{
    const uint32_t bufIdx = ripr_plp.f->bufferIndex;
    const auto sbtr = HitGroupSBTRecordData::get();
    const ripr::RiPRNodeData& inst = ripr_plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];
    const ripr::RiPRSurfaceData& geomInst = ripr_plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    PathTraceWriteOnlyPayload* woPayload;
    PathTraceReadWritePayload* rwPayload;
    PathTraceRayPayloadSignature::get (&woPayload, &rwPayload);
    PCG32RNG& rng = rwPayload->rng;

    const Point3D rayOrigin (optixGetWorldRayOrigin());
}

CUDA_DEVICE_KERNEL void RT_MS_NAME (miss)()
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
            const float uvPDF = ripr_plp.s->envLightImportanceMap.evaluatePDF (texCoord.x, texCoord.y);
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
