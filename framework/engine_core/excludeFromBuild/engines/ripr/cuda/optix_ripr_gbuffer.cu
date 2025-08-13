

#include "principledDisney_ripr.h"
#include "../ripr_shared.h"

using namespace ripr_shared;

// Global declaration of pipeline launch parameters - must match the name in pipeline configuration
RT_PIPELINE_LAUNCH_PARAMETERS ripr_shared::PipelineLaunchParameters ripr_plp;

CUDA_DEVICE_KERNEL void RT_RG_NAME (setupGBuffers)()
{
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = ripr_plp.s->bufferIndex;

    const PerspectiveCamera& camera = ripr_plp.s->camera;
    float jx = 0.5f;
    float jy = 0.5f;
    if (true)
    {
        PCG32RNG rng = ripr_plp.s->rngBuffer.read (launchIndex);
        jx = rng.getFloat0cTo1o();
        jy = rng.getFloat0cTo1o();
        ripr_plp.s->rngBuffer.write (launchIndex, rng);
    }
    const float x = (launchIndex.x + jx) / ripr_plp.s->imageSize.x;
    const float y = (launchIndex.y + jy) / ripr_plp.s->imageSize.y;
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
        ripr_plp.s->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        GBufferRayType::Primary, maxNumRayTypes, GBufferRayType::Primary,
        hitPointParamsPtr, pickInfoPtr);

    const Point2D curRasterPos (launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    const Point2D prevRasterPos =
        ripr_plp.s->prevCamera.calcScreenPosition (hitPointParams.prevPositionInWorld) * Point2D (ripr_plp.s->imageSize.x, ripr_plp.s->imageSize.y);
    Vector2D motionVector = curRasterPos - prevRasterPos;
    // FIXME
    //  if (ripr_plp.s->resetFlowBuffer || isnan (hitPointParams.prevPositionInWorld.x))
    motionVector = Vector2D (0.0f, 0.0f);

    GBuffer0Elements gb0Elems = {};
    gb0Elems.instSlot = hitPointParams.instSlot;
    gb0Elems.geomInstSlot = hitPointParams.geomInstSlot;
    gb0Elems.primIndex = hitPointParams.primIndex;
    gb0Elems.qbcB = hitPointParams.qbcB;
    gb0Elems.qbcC = hitPointParams.qbcC;
    GBuffer1Elements gb1Elems = {};
    gb1Elems.motionVector = motionVector;

    ripr_plp.s->geoBuffer0[bufIdx].write (launchIndex, gb0Elems);
    ripr_plp.s->geoBuffer1[bufIdx].write (launchIndex, gb1Elems);

    if (launchIndex.x == ripr_plp.s->mousePosition.x &&
        launchIndex.y == ripr_plp.s->mousePosition.y)
    {
        pickInfo.instSlot = hitPointParams.instSlot;
        pickInfo.geomInstSlot = hitPointParams.geomInstSlot;
        pickInfo.primIndex = hitPointParams.primIndex;
        pickInfo.positionInWorld = hitPointParams.positionInWorld;
        pickInfo.normalInWorld = hitPointParams.normalInWorld;
        pickInfo.albedo = hitPointParams.albedo;
        *ripr_plp.s->pickInfoBuffer[bufIdx] = pickInfo;
    }

    // EN: Output information required for the denoiser.
    RGB prevAlbedoResult (0.0f, 0.0f, 0.0f);
    Normal3D prevNormalResult (0.0f, 0.0f, 0.0f);
    if (ripr_plp.s->numAccumFrames > 0)
    {
        prevAlbedoResult = RGB (getXYZ (ripr_plp.s->albedoAccumBuffer.read (launchIndex)));
        prevNormalResult = Normal3D (getXYZ (ripr_plp.s->normalAccumBuffer.read (launchIndex)));
    }
    const float curWeight = 1.0f / (1 + ripr_plp.s->numAccumFrames);
    const RGB albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * hitPointParams.albedo;
    const Normal3D normalResult = (1 - curWeight) * prevNormalResult + curWeight * hitPointParams.normalInWorld;
    ripr_plp.s->albedoAccumBuffer.write (launchIndex, make_float4 (albedoResult.toNative(), 1.0f));
    ripr_plp.s->normalAccumBuffer.write (launchIndex, make_float4 (normalResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_CH_NAME (setupGBuffers)()
{

    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
#if 0
    if (launchIndex.x % 100 == 0 && launchIndex.y % 100 == 0)
    {
        printf ("RiPR GBuffer CH: HIT DETECTED! pixel (%u, %u), instance %u, ray t=%.3f\n",
                launchIndex.x, launchIndex.y, optixGetInstanceId(), optixGetRayTmax());
    }
#endif

    auto sbtr = HitGroupSBTRecordData::get();
    const shared::DisneyData& mat = ripr_plp.s->materialDataBuffer[sbtr.materialSlot];
    const shared::GeometryInstanceData& geomInst = ripr_plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];

    // Get instance data using buffer index from launch parameters
    const uint32_t bufIdx = ripr_plp.s->bufferIndex;
    const shared::InstanceData& inst = ripr_plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];

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
        mat, texCoord, 0.0f, ripr_plp.s->makeAllGlass, ripr_plp.s->globalGlassIOR,
        ripr_plp.s->globalTransmittanceDist, ripr_plp.s->globalGlassType);

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

    if (launchIndex.x == ripr_plp.s->mousePosition.x &&
        launchIndex.y == ripr_plp.s->mousePosition.y)
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

    const float phi = posPhi + ripr_plp.s->envLightRotation;

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

     if (launchIndex.x == ripr_plp.s->mousePosition.x &&
        launchIndex.y == ripr_plp.s->mousePosition.y)
    {
        pickInfo->hit = true;
        pickInfo->matSlot = 0xFFFFFFFF;
        RGB emittance (0.0f, 0.0f, 0.0f);
        if (ripr_plp.s->envLightTexture)
        {
            float4 texValue = tex2DLod<float4> (ripr_plp.s->envLightTexture, u, v, 0.0f);
            emittance = RGB (getXYZ (texValue));
            emittance *= pi_v<float> * ripr_plp.s->envLightPowerCoeff;
        }
        pickInfo->emittance = emittance;
    }
}