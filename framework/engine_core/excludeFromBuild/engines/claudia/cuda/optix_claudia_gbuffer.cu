// optix_claudia_gbuffer.cu
// OptiX G-buffer generation kernels for the Claudia engine

#include "principledDisney_claudia.h"
#include "../claudia_shared.h"

using namespace claudia_shared;



// The global declaration of claudia_plp_split is now in claudia_shared.h

// Ray generation program for GBuffer setup
CUDA_DEVICE_KERNEL void RT_RG_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = claudia_plp_split.f->bufferIndex;

    const PerspectiveCamera& camera = claudia_plp_split.f->camera;
    
    // Jittered sampling for anti-aliasing
    float jx = 0.5f;
    float jy = 0.5f;
    if (claudia_plp_split.f->enableJittering)
    {
        PCG32RNG rng = claudia_plp_split.s->rngBuffer.read(launchIndex);
        jx = rng.getFloat0cTo1o();
        jy = rng.getFloat0cTo1o();
        claudia_plp_split.s->rngBuffer.write(launchIndex, rng);
    }
    
    // Convert pixel coordinates to normalized device coordinates
    const float x = (launchIndex.x + jx) / claudia_plp_split.s->imageSize.x;
    const float y = (launchIndex.y + jy) / claudia_plp_split.s->imageSize.y;
    
    // Calculate view frustum dimensions
    const float vh = 2 * std::tan(camera.fovY * 0.5f);
    const float vw = camera.aspect * vh;

    // Generate primary ray
    const Point3D origin = camera.position;
    const Vector3D direction = normalize(camera.orientation * Vector3D(vw * (0.5f - x), vh * (0.5f - y), 1));

    // Initialize hit point parameters
    HitPointParams hitPointParams;
    hitPointParams.albedo = RGB(0.0f);
    hitPointParams.positionInWorld = Point3D(NAN);
    hitPointParams.prevPositionInWorld = Point3D(NAN);
    hitPointParams.normalInWorld = Normal3D(NAN);
    hitPointParams.instSlot = 0xFFFFFFFF;
    hitPointParams.geomInstSlot = 0xFFFFFFFF;
    hitPointParams.primIndex = 0xFFFFFFFF;
    hitPointParams.qbcB = 0;
    hitPointParams.qbcC = 0;

    PickInfo pickInfo = {};

    HitPointParams* hitPointParamsPtr = &hitPointParams;
    PickInfo* pickInfoPtr = &pickInfo;
    
    // Trace primary ray for GBuffer generation
    GBufferRayPayloadSignature::trace(
        claudia_plp_split.f->travHandle, origin.toNative(), direction.toNative(),
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        GBufferRayType::Primary, maxNumRayTypes, GBufferRayType::Primary,
        hitPointParamsPtr, pickInfoPtr);

    // Calculate motion vectors for temporal effects
    const Point2D curRasterPos(launchIndex.x + 0.5f, launchIndex.y + 0.5f);
    const Point2D prevRasterPos =
        claudia_plp_split.f->prevCamera.calcScreenPosition(hitPointParams.prevPositionInWorld) * 
        Point2D(claudia_plp_split.s->imageSize.x, claudia_plp_split.s->imageSize.y);
    Vector2D motionVector = curRasterPos - prevRasterPos;
    
    // Reset motion vectors if requested or if no valid previous position
    if (claudia_plp_split.f->resetFlowBuffer || isnan(hitPointParams.prevPositionInWorld.x))
        motionVector = Vector2D(0.0f, 0.0f);

    // Store GBuffer data
    GBuffer0Elements gb0Elems = {};
    gb0Elems.instSlot = hitPointParams.instSlot;
    gb0Elems.geomInstSlot = hitPointParams.geomInstSlot;
    gb0Elems.primIndex = hitPointParams.primIndex;
    gb0Elems.qbcB = hitPointParams.qbcB;
    gb0Elems.qbcC = hitPointParams.qbcC;
    
    GBuffer1Elements gb1Elems = {};
    gb1Elems.motionVector = motionVector;

    claudia_plp_split.s->GBuffer0[bufIdx].write(launchIndex, gb0Elems);
    claudia_plp_split.s->GBuffer1[bufIdx].write(launchIndex, gb1Elems);

    // Handle pick info for mouse interaction
    if (launchIndex.x == claudia_plp_split.f->mousePosition.x &&
        launchIndex.y == claudia_plp_split.f->mousePosition.y)
    {
        pickInfo.instanceIndex = hitPointParams.instSlot;
        pickInfo.primIndex = hitPointParams.primIndex;
        pickInfo.positionInWorld = hitPointParams.positionInWorld;
        pickInfo.normalInWorld = hitPointParams.normalInWorld;
        pickInfo.rayOrigin = origin;
        pickInfo.rayDir = direction;
        // Only write pick info if buffer is allocated
        if (claudia_plp_split.s->pickInfos[bufIdx])
        {
            *claudia_plp_split.s->pickInfos[bufIdx] = pickInfo;
        }
    }

    // Accumulate albedo and normal for denoising
    RGB prevAlbedoResult(0.0f, 0.0f, 0.0f);
    Normal3D prevNormalResult(0.0f, 0.0f, 0.0f);
    if (claudia_plp_split.f->numAccumFrames > 0)
    {
        prevAlbedoResult = RGB(getXYZ(claudia_plp_split.s->albedoAccumBuffer.read(launchIndex)));
        prevNormalResult = Normal3D(getXYZ(claudia_plp_split.s->normalAccumBuffer.read(launchIndex)));
    }
    
    const float curWeight = 1.0f / (1 + claudia_plp_split.f->numAccumFrames);
    const RGB albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * hitPointParams.albedo;
    const Normal3D normalResult = (1 - curWeight) * prevNormalResult + curWeight * hitPointParams.normalInWorld;
    
    claudia_plp_split.s->albedoAccumBuffer.write(launchIndex, make_float4(albedoResult.toNative(), 1.0f));
    claudia_plp_split.s->normalAccumBuffer.write(launchIndex, make_float4(normalResult.toNative(), 1.0f));
}

// Closest hit program for GBuffer setup
CUDA_DEVICE_KERNEL void RT_CH_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint32_t bufIdx = claudia_plp_split.f->bufferIndex;

    // Get SBT data and instance information
    const auto sbtr = HitGroupSBTRecordData::get();
    const shared::InstanceData& inst = claudia_plp_split.s->instanceDataBufferArray[bufIdx][optixGetInstanceId()];
    const shared::GeometryInstanceData& geomInst = claudia_plp_split.s->geometryInstanceDataBuffer[sbtr.geomInstSlot];
    const shared::DisneyData& mat = claudia_plp_split.s->materialDataBuffer[geomInst.materialSlot];

    // Get payload pointers
    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    GBufferRayPayloadSignature::get(&hitPointParams, &pickInfo);

    // Get hit parameters
    const auto hp = HitPointParameter::get();
    hitPointParams->instSlot = optixGetInstanceId();
    hitPointParams->geomInstSlot = sbtr.geomInstSlot;
    hitPointParams->primIndex = hp.primIndex;

    // Calculate hit point attributes
    Point3D positionInWorld;
    Point3D prevPositionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Point2D texCoord;
    
    {
        // Get triangle vertices
        const shared::Triangle& tri = geomInst.triangleBuffer[hp.primIndex];
        const shared::Vertex& vA = geomInst.vertexBuffer[tri.index0];
        const shared::Vertex& vB = geomInst.vertexBuffer[tri.index1];
        const shared::Vertex& vC = geomInst.vertexBuffer[tri.index2];
        
        // Barycentric coordinates
        const float bcB = hp.bcB;
        const float bcC = hp.bcC;
        const float bcA = 1 - (bcB + bcC);
        
        // Encode barycentric coordinates for storage
        hitPointParams->qbcB = encodeBarycentric(bcB);
        hitPointParams->qbcC = encodeBarycentric(bcC);
        
        // Interpolate vertex attributes
        const Point3D positionInObj = bcA * vA.position + bcB * vB.position + bcC * vC.position;
        const Normal3D shadingNormalInObj = bcA * vA.normal + bcB * vB.normal + bcC * vC.normal;
        const Vector3D texCoord0DirInObj = bcA * vA.texCoord0Dir + bcB * vB.texCoord0Dir + bcC * vC.texCoord0Dir;
        texCoord = bcA * vA.texCoord + bcB * vB.texCoord + bcC * vC.texCoord;

        // Transform to world space
        positionInWorld = transformPointFromObjectToWorldSpace(positionInObj);
        prevPositionInWorld = inst.curToPrevTransform * positionInWorld;
        shadingNormalInWorld = normalize(transformNormalFromObjectToWorldSpace(shadingNormalInObj));
        
        // Project texture coordinate direction onto surface
        texCoord0DirInWorld = transformVectorFromObjectToWorldSpace(texCoord0DirInObj);
        texCoord0DirInWorld = normalize(
            texCoord0DirInWorld - dot(shadingNormalInWorld, texCoord0DirInWorld) * shadingNormalInWorld);
        
        // Handle degenerate normals
        if (!shadingNormalInWorld.allFinite())
        {
            shadingNormalInWorld = Normal3D(0, 0, 1);
            texCoord0DirInWorld = Vector3D(1, 0, 0);
        }
    }

    // Store hit point data
    hitPointParams->positionInWorld = positionInWorld;
    hitPointParams->prevPositionInWorld = prevPositionInWorld;
    hitPointParams->texCoord = texCoord;

    // Build shading frame for normal mapping
    ReferenceFrame shadingFrame(shadingNormalInWorld, texCoord0DirInWorld);
    
    // Apply normal mapping if enabled
    if (mat.normal)
    {
        // TODO: Implement normal mapping
        // const Normal3D modLocalNormal = readNormalMap(mat.normal, texCoord);
        // applyBumpMapping(modLocalNormal, &shadingFrame);
    }

    hitPointParams->normalInWorld = shadingFrame.normal;
    
    // Read base color for albedo
    RGB albedo(0.8f, 0.8f, 0.8f); // Default gray
    if (mat.baseColor) {
        float4 texValue = tex2DLod<float4>(mat.baseColor, texCoord.x, texCoord.y, 0.0f);
        albedo = RGB(texValue.x, texValue.y, texValue.z);
    }
    hitPointParams->albedo = albedo;

    // Handle pick info for mouse selection
    if (launchIndex.x == claudia_plp_split.f->mousePosition.x &&
        launchIndex.y == claudia_plp_split.f->mousePosition.y)
    {
        pickInfo->hit = true;
        pickInfo->matIndex = geomInst.materialSlot;
        
        // Check for emissive materials
        RGB emittance(0.0f, 0.0f, 0.0f);
        if (mat.emissive)
        {
            float4 texValue = tex2DLod<float4>(mat.emissive, texCoord.x, texCoord.y, 0.0f);
            emittance = RGB(getXYZ(texValue));
            
            // Apply emissive strength if available
            if (mat.emissiveStrength) {
                float4 strengthValue = tex2DLod<float4>(mat.emissiveStrength, texCoord.x, texCoord.y, 0.0f);
                emittance *= strengthValue.x;
            }
        }
        // Store emittance in albedo for pick info (temporary solution)
        pickInfo->normalInWorld = shadingFrame.normal;
    }
}

// Miss program for GBuffer setup
CUDA_DEVICE_KERNEL void RT_MS_NAME(setupGBuffers)() {
    const uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    // Get ray direction for environment mapping
    const Vector3D vOut(-Vector3D(optixGetWorldRayDirection()));
    const Point3D p(-vOut);

    // Convert to spherical coordinates for environment texture lookup
    float posPhi, posTheta;
    toPolarYUp(Vector3D(p), &posPhi, &posTheta);

    const float phi = posPhi + claudia_plp_split.f->envLightRotation;

    // Calculate UV coordinates for environment map
    float u = phi / (2 * pi_v<float>);
    u -= floorf(u);
    const float v = posTheta / pi_v<float>;

    // Get payload pointers
    HitPointParams* hitPointParams;
    PickInfo* pickInfo;
    GBufferRayPayloadSignature::get(&hitPointParams, &pickInfo);

    // Store environment hit data
    hitPointParams->positionInWorld = p;
    hitPointParams->prevPositionInWorld = p;
    hitPointParams->normalInWorld = Normal3D(vOut);
    hitPointParams->qbcB = encodeBarycentric(u);
    hitPointParams->qbcC = encodeBarycentric(v);
    
    // Set default albedo for environment
    RGB envColor(0.0f, 0.0f, 0.0f);
    if (claudia_plp_split.f->enableEnvLight && claudia_plp_split.s->envLightTexture)
    {
        float4 texValue = tex2DLod<float4>(claudia_plp_split.s->envLightTexture, u, v, 0.0f);
        envColor = RGB(getXYZ(texValue)) * claudia_plp_split.f->envLightPowerCoeff;
    }
    else if (claudia_plp_split.f->useSolidBackground)
    {
        envColor = RGB(claudia_plp_split.f->backgroundColor);
    }
    hitPointParams->albedo = envColor;
    
    // IMPORTANT: Keep instSlot as 0xFFFFFFFF to indicate miss
    // This is already initialized in ray gen, but ensure it stays that way
    
    // Handle pick info for environment selection
    if (launchIndex.x == claudia_plp_split.f->mousePosition.x &&
        launchIndex.y == claudia_plp_split.f->mousePosition.y)
    {
        pickInfo->hit = true;
        pickInfo->matIndex = 0xFFFFFFFF; // Indicate environment hit
        
        // Store environment radiance for pick info
        if (claudia_plp_split.s->envLightTexture && claudia_plp_split.f->enableEnvLight)
        {
            float4 texValue = tex2DLod<float4>(claudia_plp_split.s->envLightTexture, u, v, 0.0f);
            RGB emittance = RGB(getXYZ(texValue));
            emittance *= pi_v<float> * claudia_plp_split.f->envLightPowerCoeff;
            // Store in normalInWorld temporarily (will need better solution)
            pickInfo->normalInWorld = Normal3D(emittance.r, emittance.g, emittance.b);
        }
    }
}