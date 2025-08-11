// optix_shocker_gbuffer.cu
// OptiX G-buffer generation kernels for the Shocker engine
// STUB VERSION - no implementation

#include "principledDisney_shocker.h"
#include "../shocker_shared.h"




using namespace shocker_shared;

CUDA_DEVICE_KERNEL void RT_RG_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    const PerspectiveCamera& camera = plp.f->camera;
    float jx = 0.5f;
    float jy = 0.5f;
    if (plp.f->enableJittering)
    {
        PCG32RNG rng = plp.s->rngBuffer.read (launchIndex);
        jx = rng.getFloat0cTo1o();
        jy = rng.getFloat0cTo1o();
        plp.s->rngBuffer.write (launchIndex, rng);
    }
    const float x = (launchIndex.x + jx) / plp.s->imageSize.x;
    const float y = (launchIndex.y + jy) / plp.s->imageSize.y;
    const float vh = 2 * std::tan (camera.fovY * 0.5f);
    const float vw = camera.aspect * vh;

    const Point3D origin = camera.position;
    const Vector3D direction = normalize (camera.orientation * Vector3D (vw * (0.5f - x), vh * (0.5f - y), 1));

    HitPointParams hitPointParams;
    hitPointParams.albedo = RGB (0.0f);
    hitPointParams.positionInWorld = Point3D (NAN);
    hitPointParams.prevPositionInWorld = Point3D (NAN);
    hitPointParams.shadingNormalInWorld = Normal3D (NAN);
    hitPointParams.instSlot = 0xFFFFFFFF;
    hitPointParams.geomInstSlot = 0xFFFFFFFF;
    hitPointParams.primIndex = 0xFFFFFFFF;
    hitPointParams.qbcB = 0;
    hitPointParams.qbcC = 0;

    PickInfo pickInfo = {};

    // // Debug: Print traversable handle and ray info for first pixel
    //if (launchIndex.x == 0 && launchIndex.y == 0) {
    //    printf("ShockerGBuffer RG: travHandle=%llu, origin=(%.2f,%.2f,%.2f), dir=(%.2f,%.2f,%.2f)\n",
    //           plp.f->travHandle, origin.x, origin.y, origin.z, direction.x, direction.y, direction.z);
    //}

    HitPointParams* hitPointParamsPtr = &hitPointParams;
    PickInfo* pickInfoPtr = &pickInfo;
    
    //// Debug: Print ray type values
    //if (launchIndex.x == 0 && launchIndex.y == 0) {
    //    printf("ShockerGBuffer RG: Ray types - SBT offset=%u, SBT stride=%u, miss index=%u\n",
    //           (uint32_t)GBufferRayType::Primary, (uint32_t)maxNumRayTypes, (uint32_t)GBufferRayType::Primary);
    //}
    
    PrimaryRayPayloadSignature::trace (
        plp.f->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        GBufferRayType::Primary, maxNumRayTypes, GBufferRayType::Primary,
        hitPointParamsPtr, pickInfoPtr);

    // Debug: ONLY print if there's a hit
    if (hitPointParams.instSlot != 0xFFFFFFFF) {
        // Found a hit! Print details for sampled pixels
        if (launchIndex.x % 50 == 0 && launchIndex.y % 50 == 0) {
            printf("ShockerGBuffer RG: HIT at [%u,%u]! instSlot=%u, primIndex=%u, pos=(%.2f,%.2f,%.2f)\n",
                   launchIndex.x, launchIndex.y, hitPointParams.instSlot, hitPointParams.primIndex,
                   hitPointParams.positionInWorld.x, hitPointParams.positionInWorld.y, hitPointParams.positionInWorld.z);
        }
    }

    const Point2D curRasterPos (launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    const Point2D prevRasterPos =
        plp.f->prevCamera.calcScreenPosition (hitPointParams.prevPositionInWorld) * Point2D (plp.s->imageSize.x, plp.s->imageSize.y);
    Vector2D motionVector = curRasterPos - prevRasterPos;
    if (plp.f->resetFlowBuffer || isnan (hitPointParams.prevPositionInWorld.x))
        motionVector = Vector2D (0.0f, 0.0f);

    GBuffer0Elements gb0Elems = {};
    gb0Elems.instSlot = hitPointParams.instSlot;
    gb0Elems.geomInstSlot = hitPointParams.geomInstSlot;
    gb0Elems.primIndex = hitPointParams.primIndex;
    gb0Elems.qbcB = hitPointParams.qbcB;
    gb0Elems.qbcC = hitPointParams.qbcC;
    GBuffer1Elements gb1Elems = {};
    gb1Elems.motionVector = motionVector;

    plp.s->GBuffer0[bufIdx].write (launchIndex, gb0Elems);
    plp.s->GBuffer1[bufIdx].write (launchIndex, gb1Elems);

      if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y)
    {
        pickInfo.instSlot = hitPointParams.instSlot;
        pickInfo.geomInstSlot = hitPointParams.geomInstSlot;
        pickInfo.primIndex = hitPointParams.primIndex;
        pickInfo.positionInWorld = hitPointParams.positionInWorld;
        pickInfo.normalInWorld = hitPointParams.shadingNormalInWorld;
        pickInfo.albedo = hitPointParams.albedo;
        *plp.s->pickInfos[bufIdx] = pickInfo;
    }

  
    // EN: Output information required for the denoiser.
    RGB prevAlbedoResult (0.0f, 0.0f, 0.0f);
    Normal3D prevNormalResult (0.0f, 0.0f, 0.0f);
    if (plp.f->numAccumFrames > 0)
    {
        prevAlbedoResult = RGB (getXYZ (plp.s->albedoAccumBuffer.read (launchIndex)));
        prevNormalResult = Normal3D (getXYZ (plp.s->normalAccumBuffer.read (launchIndex)));
    }
    const float curWeight = 1.0f / (1 + plp.f->numAccumFrames);
    const RGB albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * hitPointParams.albedo;
    const Normal3D normalResult = (1 - curWeight) * prevNormalResult + curWeight * hitPointParams.shadingNormalInWorld;
    plp.s->albedoAccumBuffer.write (launchIndex, make_float4 (albedoResult.toNative(), 1.0f));
    plp.s->normalAccumBuffer.write (launchIndex, make_float4 (normalResult.toNative(), 1.0f));
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = plp.f->bufferIndex;

    // Debug: Print for ANY hit to confirm CH is being called
    // Just print for specific pixels to avoid spam
    if (launchIndex.x % 100 == 0 && launchIndex.y % 100 == 0) {
        printf("ShockerGBuffer CH: HIT DETECTED! pixel (%u, %u), instance %u, ray t=%.3f\n", 
               launchIndex.x, launchIndex.y, optixGetInstanceId(), optixGetRayTmax());
    }

    const auto sbtr = HitGroupSBTRecordData::get();
    const shocker::ShockerNodeData& inst = plp.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];
    const shocker::ShockerSurfaceData& geomInst = plp.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const shared::DisneyData& mat = plp.s->disneyMaterialBuffer[geomInst.disneyMaterialSlot];

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get (&hitPointParams, &pickInfo);

    const auto hp = HitPointParameter::get();
    hitPointParams->instSlot = optixGetInstanceId();
    hitPointParams->geomInstSlot = sbtr.geomInstSlot;
    hitPointParams->primIndex = hp.primIndex;

    // Debug: Print hit information for specific pixels
    if (launchIndex.x == 400 && launchIndex.y == 300) {
        printf("ShockerGBuffer CH Detail: primIndex=%u, geomInstSlot=%u, bcB=%.3f, bcC=%.3f\n",
               hp.primIndex, sbtr.geomInstSlot, hp.bcB, hp.bcC);
    }

    Point3D positionInWorld;
    Point3D prevPositionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Point2D texCoord;
    {
        const shared::Triangle& tri = geomInst.triangleBuffer[hp.primIndex];
        const shared::Vertex& vA = geomInst.vertexBuffer[tri.index0];
        const shared::Vertex& vB = geomInst.vertexBuffer[tri.index1];
        const shared::Vertex& vC = geomInst.vertexBuffer[tri.index2];
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

        // Debug: Print world position for center pixel
        if (launchIndex.x == 400 && launchIndex.y == 300) {
            printf("ShockerGBuffer CH: World pos=(%.2f, %.2f, %.2f), normal=(%.2f, %.2f, %.2f)\n",
                   positionInWorld.x, positionInWorld.y, positionInWorld.z,
                   shadingNormalInWorld.x, shadingNormalInWorld.y, shadingNormalInWorld.z);
        }
    }

    hitPointParams->positionInWorld = positionInWorld;
    hitPointParams->prevPositionInWorld = prevPositionInWorld;

    // TODO: Implement Disney BSDF evaluation
    // For now, just use base color as albedo
    ReferenceFrame shadingFrame (shadingNormalInWorld, texCoord0DirInWorld);
    if (plp.f->enableBumpMapping && mat.normal)
    {
        // TODO: Implement bump mapping with Disney normal texture
        // const Normal3D modLocalNormal = readNormalMap(mat.normal, texCoord);
        // applyBumpMapping (modLocalNormal, &shadingFrame);
    }

    hitPointParams->shadingNormalInWorld = shadingFrame.normal;
    
    // Read base color for albedo
    RGB albedo(0.8f, 0.8f, 0.8f); // Default gray
    if (mat.baseColor) {
        float4 texValue = tex2DLod<float4>(mat.baseColor, texCoord.x, texCoord.y, 0.0f);
        albedo = RGB(texValue.x, texValue.y, texValue.z);
    }
    hitPointParams->albedo = albedo;

    // Debug: Print albedo for first hit
    static bool firstHit = true;
    if (firstHit && launchIndex.x < 10 && launchIndex.y < 10) {
        printf("ShockerGBuffer CH: Albedo set to (%.2f, %.2f, %.2f) at pixel (%u, %u)\n",
               albedo.r, albedo.g, albedo.b, launchIndex.x, launchIndex.y);
        firstHit = false;
    }

    // JP: ??????????????????????
    // EN: Export the information of the pixel on which the mouse is.
    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y)
    {
        pickInfo->hit = true;
        pickInfo->matSlot = geomInst.disneyMaterialSlot;
        RGB emittance (0.0f, 0.0f, 0.0f);
        if (mat.emissive)
        {
            float4 texValue = tex2DLod<float4> (mat.emissive, texCoord.x, texCoord.y, 0.0f);
            emittance = RGB (getXYZ (texValue));
            // Apply emissive strength if available
            if (mat.emissiveStrength) {
                float4 strengthValue = tex2DLod<float4> (mat.emissiveStrength, texCoord.x, texCoord.y, 0.0f);
                emittance *= strengthValue.x;
            }
        }
        pickInfo->emittance = emittance;
    }
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    const Vector3D vOut (-Vector3D (optixGetWorldRayDirection()));
    const Point3D p (-vOut);

    float posPhi, posTheta;
    toPolarYUp (Vector3D (p), &posPhi, &posTheta);

    const float phi = posPhi + plp.f->envLightRotation;

    float u = phi / (2 * pi_v<float>);
    u -= floorf (u);
    const float v = posTheta / pi_v<float>;

    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    PrimaryRayPayloadSignature::get (&hitPointParams, &pickInfo);

    // Debug: Confirm miss shader is setting values  
    if (launchIndex.x == 400 && launchIndex.y == 300) {
        printf("ShockerGBuffer MS: Setting miss values for pixel (%u,%u)\n", launchIndex.x, launchIndex.y);
    }

    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = p;
    hitPointParams->shadingNormalInWorld = Normal3D (vOut);
    hitPointParams->qbcB = encodeBarycentric (u);
    hitPointParams->qbcC = encodeBarycentric (v);
    
    // IMPORTANT: Miss shader should keep instSlot as 0xFFFFFFFF to indicate no hit!
    // The ray gen was detecting "hits" because miss shader wasn't leaving this unchanged

    // JP: ??????????????????????
    // EN: Export the information of the pixel on which the mouse is.
    if (launchIndex.x == plp.f->mousePosition.x &&
        launchIndex.y == plp.f->mousePosition.y)
    {
        pickInfo->hit = true;
        pickInfo->matSlot = 0xFFFFFFFF;
        RGB emittance (0.0f, 0.0f, 0.0f);
        if (plp.s->envLightTexture && plp.f->enableEnvLight)
        {
            float4 texValue = tex2DLod<float4> (plp.s->envLightTexture, u, v, 0.0f);
            emittance = RGB (getXYZ (texValue));
            emittance *= pi_v<float> * plp.f->envLightPowerCoeff;
        }
        pickInfo->emittance = emittance;
    }
}