

#include "principledDisney_shocker.h"
#include "../shocker_shared.h"

using namespace shocker_shared;

// Global declaration of pipeline launch parameters - must match the name in pipeline configuration
RT_PIPELINE_LAUNCH_PARAMETERS shocker_shared::PipelineLaunchParameters shocker_plp;

CUDA_DEVICE_KERNEL void RT_RG_NAME (setupGBuffers)()
{
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = shocker_plp.f->bufferIndex;

    const PerspectiveCamera& camera = shocker_plp.f->camera;
    float jx = 0.5f;
    float jy = 0.5f;
    if (true)
    {
        PCG32RNG rng = shocker_plp.s->rngBuffer.read (launchIndex);
        jx = rng.getFloat0cTo1o();
        jy = rng.getFloat0cTo1o();
        shocker_plp.s->rngBuffer.write (launchIndex, rng);
    }
    const float x = (launchIndex.x + jx) / shocker_plp.s->imageSize.x;
    const float y = (launchIndex.y + jy) / shocker_plp.s->imageSize.y;
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
        shocker_plp.f->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        GBufferRayType::Primary, maxNumRayTypes, GBufferRayType::Primary,
        hitPointParamsPtr, pickInfoPtr);

    const Point2D curRasterPos (launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    const Point2D prevRasterPos =
        shocker_plp.f->prevCamera.calcScreenPosition (hitPointParams.prevPositionInWorld) * Point2D (shocker_plp.s->imageSize.x, shocker_plp.s->imageSize.y);
    Vector2D motionVector = curRasterPos - prevRasterPos;
    // FIXME
    //  if (shocker_plp.f->resetFlowBuffer || isnan (hitPointParams.prevPositionInWorld.x))
    motionVector = Vector2D (0.0f, 0.0f);

    GBuffer0Elements gb0Elems = {};
    gb0Elems.instSlot = hitPointParams.instSlot;
    gb0Elems.geomInstSlot = hitPointParams.geomInstSlot;
    gb0Elems.primIndex = hitPointParams.primIndex;
    gb0Elems.qbcB = hitPointParams.qbcB;
    gb0Elems.qbcC = hitPointParams.qbcC;
    GBuffer1Elements gb1Elems = {};
    gb1Elems.motionVector = motionVector;

    shocker_plp.s->GBuffer0[bufIdx].write (launchIndex, gb0Elems);
    shocker_plp.s->GBuffer1[bufIdx].write (launchIndex, gb1Elems);

    if (launchIndex.x == shocker_plp.f->mousePosition.x &&
        launchIndex.y == shocker_plp.f->mousePosition.y)
    {
        pickInfo.instSlot = hitPointParams.instSlot;
        pickInfo.geomInstSlot = hitPointParams.geomInstSlot;
        pickInfo.primIndex = hitPointParams.primIndex;
        pickInfo.positionInWorld = hitPointParams.positionInWorld;
        pickInfo.normalInWorld = hitPointParams.normalInWorld;
        pickInfo.albedo = hitPointParams.albedo;
        *shocker_plp.s->pickInfos[bufIdx] = pickInfo;
    }

    // EN: Output information required for the denoiser.
    RGB prevAlbedoResult (0.0f, 0.0f, 0.0f);
    Normal3D prevNormalResult (0.0f, 0.0f, 0.0f);
    if (shocker_plp.f->numAccumFrames > 0)
    {
        prevAlbedoResult = RGB (getXYZ (shocker_plp.s->albedoAccumBuffer.read (launchIndex)));
        prevNormalResult = Normal3D (getXYZ (shocker_plp.s->normalAccumBuffer.read (launchIndex)));
    }
    const float curWeight = 1.0f / (1 + shocker_plp.f->numAccumFrames);
    const RGB albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * hitPointParams.albedo;
    const Normal3D normalResult = (1 - curWeight) * prevNormalResult + curWeight * hitPointParams.normalInWorld;
    shocker_plp.s->albedoAccumBuffer.write (launchIndex, make_float4 (albedoResult.toNative(), 1.0f));
    shocker_plp.s->normalAccumBuffer.write (launchIndex, make_float4 (normalResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_CH_NAME (setupGBuffers)()
{

    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
#if 0
    if (launchIndex.x % 100 == 0 && launchIndex.y % 100 == 0)
    {
        printf ("Shocker GBuffer CH: HIT DETECTED! pixel (%u, %u), instance %u, ray t=%.3f\n",
                launchIndex.x, launchIndex.y, optixGetInstanceId(), optixGetRayTmax());
    }
#endif

    auto sbtr = HitGroupSBTRecordData::get();
    const shared::DisneyData& mat = shocker_plp.s->materialDataBuffer[sbtr.materialSlot];
    const shared::GeometryInstanceData& geomInst = shocker_plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    // Get instance data using buffer index from launch parameters
    const uint32_t bufIdx = shocker_plp.f->bufferIndex;
    const shared::InstanceData& inst = shocker_plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get (&hitPointParams, &pickInfo);

    const auto hp = HitPointParameter::get();
    hitPointParams->instSlot = optixGetInstanceId();
    hitPointParams->geomInstSlot = sbtr.geomInstSlot;
    hitPointParams->primIndex = hp.primIndex;

    Point3D positionInWorld;
    Point3D prevPositionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Point2D texCoord;
    {
        const Triangle& tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex& vA = geomInst.vertexBuffer[tri.index0];
        const Vertex& vB = geomInst.vertexBuffer[tri.index1];
        const Vertex& vC = geomInst.vertexBuffer[tri.index2];
        const float bcB = hp.bcB;
        const float bcC = hp.bcC;
        const float bcA = 1 - (bcB + bcC);
        hitPointParams->qbcB = encodeBarycentric (bcB);
        hitPointParams->qbcC = encodeBarycentric (bcC);
        const Point3D positionInObj = bcA * vA.position + bcB * vB.position + bcC * vC.position;
        const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

        positionInWorld = transformPointFromObjectToWorldSpace (positionInObj);
        prevPositionInWorld = inst.curToPrevTransform * positionInWorld;
        shadingNormalInWorld = normalize (transformNormalFromObjectToWorldSpace (shadingNormalInObj));
        texCoord0DirInWorld = transformVectorFromObjectToWorldSpace (texCoord0DirInObj);
        texCoord0DirInWorld = normalize (
            texCoord0DirInWorld - dot (shadingNormalInWorld, texCoord0DirInWorld) * shadingNormalInWorld);
        if (!shadingNormalInWorld.allFinite())
        {
            shadingNormalInWorld = Normal3D (0, 0, 1);
            texCoord0DirInWorld = Vector3D (1, 0, 0);
        }
    }

    hitPointParams->positionInWorld = positionInWorld;
    hitPointParams->prevPositionInWorld = prevPositionInWorld;

    // Create DisneyPrincipled instance directly instead of using BSDF
    DisneyPrincipled bsdf = DisneyPrincipled::create (
        mat, texCoord, 0.0f, shocker_plp.s->makeAllGlass, shocker_plp.s->globalGlassIOR,
        shocker_plp.s->globalTransmittanceDist, shocker_plp.s->globalGlassType);

    ReferenceFrame shadingFrame (shadingNormalInWorld, texCoord0DirInWorld);
    /* if (plp.f->enableBumpMapping)
     {
         const Normal3D modLocalNormal = mat.readModifiedNormal (mat.normal, mat.normalDimInfo, texCoord, 0.0f);
         applyBumpMapping (modLocalNormal, &shadingFrame);
     }*/
    const Vector3D vOut (-Vector3D (optixGetWorldRayDirection()));
    const Vector3D vOutLocal = shadingFrame.toLocal (normalize (vOut));

    hitPointParams->normalInWorld = shadingFrame.normal;
    hitPointParams->albedo = bsdf.evaluateDHReflectanceEstimate (vOutLocal);

    if (launchIndex.x == shocker_plp.f->mousePosition.x &&
        launchIndex.y == shocker_plp.f->mousePosition.y)
    {
        pickInfo->hit = true;
        pickInfo->matSlot = sbtr.materialSlot;
        RGB emittance (0.0f, 0.0f, 0.0f);
        if (mat.emissive)
        {
            float4 texValue = tex2DLod<float4> (mat.emissive, texCoord.x, texCoord.y, 0.0f);
            emittance = RGB (getXYZ (texValue));
        }
        pickInfo->emittance = emittance;
    }
}

CUDA_DEVICE_KERNEL void RT_MS_NAME (setupGBuffers)()
{
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    const Vector3D vOut (-Vector3D (optixGetWorldRayDirection()));
    const Point3D p (-vOut);

    float posPhi, posTheta;
    toPolarYUp (Vector3D (p), &posPhi, &posTheta);

    const float phi = posPhi + shocker_plp.f->envLightRotation;

    float u = phi / (2 * pi_v<float>);
    u -= floorf (u);
    const float v = posTheta / pi_v<float>;

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get (&hitPointParams, &pickInfo);

    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = p;
    hitPointParams->normalInWorld = Normal3D (vOut);
    hitPointParams->qbcB = encodeBarycentric (u);
    hitPointParams->qbcC = encodeBarycentric (v);

     if (launchIndex.x == shocker_plp.f->mousePosition.x &&
        launchIndex.y == shocker_plp.f->mousePosition.y)
    {
        pickInfo->hit = true;
        pickInfo->matSlot = 0xFFFFFFFF;
        RGB emittance (0.0f, 0.0f, 0.0f);
        if (shocker_plp.s->envLightTexture)
        {
            float4 texValue = tex2DLod<float4> (shocker_plp.s->envLightTexture, u, v, 0.0f);
            emittance = RGB (getXYZ (texValue));
            emittance *= pi_v<float> * shocker_plp.f->envLightPowerCoeff;
        }
        pickInfo->emittance = emittance;
    }
}