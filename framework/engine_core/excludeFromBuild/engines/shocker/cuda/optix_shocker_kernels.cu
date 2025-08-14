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

CUDA_DEVICE_FUNCTION CUDA_INLINE void pathTrace_rayGen_generic()
{
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = shocker_plp.f->bufferIndex;

    const GBuffer0Elements gb0Elems = shocker_plp.s->GBuffer0[bufIdx].read (launchIndex);
    const uint32_t instSlot = gb0Elems.instSlot;
    const float bcB = decodeBarycentric (gb0Elems.qbcB);
    const float bcC = decodeBarycentric (gb0Elems.qbcC);

    const PerspectiveCamera& camera = shocker_plp.f->camera;

    const bool useEnvLight = shocker_plp.s->envLightTexture && shocker_plp.f->enableEnvLight;
    RGB contribution (0.001f, 0.001f, 0.001f);
    if (instSlot != 0xFFFFFFFF)
    {
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
