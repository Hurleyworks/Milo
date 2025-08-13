

#include "principledDisney_claudia.h"
#include "../claudia_shared.h"

using namespace claudia_shared;

// Global declaration of pipeline launch parameters - must match the name in pipeline configuration
RT_PIPELINE_LAUNCH_PARAMETERS claudia_shared::PipelineLaunchParameters claudia_plp;

CUDA_DEVICE_KERNEL void RT_RG_NAME (setupGBuffers)()
{
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = claudia_plp.s->bufferIndex;

    const PerspectiveCamera& camera = claudia_plp.s->camera;
    float jx = 0.5f;
    float jy = 0.5f;
    if (true)
    {
        PCG32RNG rng = claudia_plp.s->rngBuffer.read (launchIndex);
        jx = rng.getFloat0cTo1o();
        jy = rng.getFloat0cTo1o();
        claudia_plp.s->rngBuffer.write (launchIndex, rng);
    }
    const float x = (launchIndex.x + jx) / claudia_plp.s->imageSize.x;
    const float y = (launchIndex.y + jy) / claudia_plp.s->imageSize.y;
    const float vh = 2 * std::tan (camera.fovY * 0.5f);
    const float vw = camera.aspect * vh;

    const Point3D origin = camera.position;
    const Vector3D direction = normalize (camera.orientation * Vector3D (vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    hitPointParams.albedo = RGB (0.0f);
    hitPointParams.positionInWorld = Point3D (NAN);
    hitPointParams.prevPositionInWorld = Point3D (NAN);
    hitPointParams.normalInWorld = Normal3D (NAN);
    hitPointParams.instSlot = 0xFFFFFFFF;
    hitPointParams.geomInstSlot = 0xFFFFFFFF;
    hitPointParams.primIndex = 0xFFFFFFFF;
    hitPointParams.qbcB = 0;
    hitPointParams.qbcC = 0;

    PickInfo pickInfo = {};

    HitPointParams* hitPointParamsPtr = &hitPointParams;
    PickInfo* pickInfoPtr = &pickInfo;
    PrimaryRayPayloadSignature::trace (
        claudia_plp.s->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        GBufferRayType::Primary, maxNumRayTypes, GBufferRayType::Primary,
        hitPointParamsPtr, pickInfoPtr);

    const Point2D curRasterPos (launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    const Point2D prevRasterPos =
        claudia_plp.s->prevCamera.calcScreenPosition (hitPointParams.prevPositionInWorld) * Point2D (claudia_plp.s->imageSize.x, claudia_plp.s->imageSize.y);
    Vector2D motionVector = curRasterPos - prevRasterPos;
    // FIXME
    //  if (claudia_plp.s->resetFlowBuffer || isnan (hitPointParams.prevPositionInWorld.x))
    motionVector = Vector2D (0.0f, 0.0f);

    GBuffer0Elements gb0Elems = {};
    gb0Elems.instSlot = hitPointParams.instSlot;
    gb0Elems.geomInstSlot = hitPointParams.geomInstSlot;
    gb0Elems.primIndex = hitPointParams.primIndex;
    gb0Elems.qbcB = hitPointParams.qbcB;
    gb0Elems.qbcC = hitPointParams.qbcC;
    GBuffer1Elements gb1Elems = {};
    gb1Elems.motionVector = motionVector;

    claudia_plp.s->geoBuffer0[bufIdx].write (launchIndex, gb0Elems);
    claudia_plp.s->geoBuffer1[bufIdx].write (launchIndex, gb1Elems);

    if (launchIndex.x == claudia_plp.s->mousePosition.x &&
        launchIndex.y == claudia_plp.s->mousePosition.y)
    {
        pickInfo.instSlot = hitPointParams.instSlot;
        pickInfo.geomInstSlot = hitPointParams.geomInstSlot;
        pickInfo.primIndex = hitPointParams.primIndex;
        pickInfo.positionInWorld = hitPointParams.positionInWorld;
        pickInfo.normalInWorld = hitPointParams.normalInWorld;
        pickInfo.albedo = hitPointParams.albedo;
        *claudia_plp.s->pickInfoBuffer[bufIdx] = pickInfo;
    }

    // EN: Output information required for the denoiser.
    RGB prevAlbedoResult (0.0f, 0.0f, 0.0f);
    Normal3D prevNormalResult (0.0f, 0.0f, 0.0f);
    if (claudia_plp.s->numAccumFrames > 0)
    {
        prevAlbedoResult = RGB (getXYZ (claudia_plp.s->albedoAccumBuffer.read (launchIndex)));
        prevNormalResult = Normal3D (getXYZ (claudia_plp.s->normalAccumBuffer.read (launchIndex)));
    }
    const float curWeight = 1.0f / (1 + claudia_plp.s->numAccumFrames);
    const RGB albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * hitPointParams.albedo;
    const Normal3D normalResult = (1 - curWeight) * prevNormalResult + curWeight * hitPointParams.normalInWorld;
    claudia_plp.s->albedoAccumBuffer.write (launchIndex, make_float4 (albedoResult.toNative(), 1.0f));
    claudia_plp.s->normalAccumBuffer.write (launchIndex, make_float4 (normalResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_CH_NAME (setupGBuffers)()
{
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    // const uint32_t bufIdx = plp.f->bufferIndex;

    if (launchIndex.x % 100 == 0 && launchIndex.y % 100 == 0)
    {
        printf ("Claudia GBuffer CH: HIT DETECTED! pixel (%u, %u), instance %u, ray t=%.3f\n",
                launchIndex.x, launchIndex.y, optixGetInstanceId(), optixGetRayTmax());
    }

    const auto sbtr = HitGroupSBTRecordData::get();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME (setupGBuffers)()
{
}