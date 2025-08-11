
// some taken from OptiX_Utility
// https://github.com/shocker-0x15/OptiX_Utility/blob/master/LICENSE.md
// and from Shocker GfxExp
// https://github.com/shocker-0x15/GfxEx

#include "principledDisney_milo.h"


RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;

// R2 Sequence Sampling Implementation
// ===================================
// 
// This implementation uses the R2 sequence for pixel sampling instead of random jittering.
// The R2 sequence provides optimal 2D coverage with minimal discrepancy.
//
// Benefits over Random Sampling:
// - Better Distribution: Optimal 2D coverage with minimal gaps/clusters
// - Deterministic: Reproducible results, easier debugging
// - Low Discrepancy: Converges faster than random sampling
// - No Clustering: Avoids the clumping artifacts of random sampling
// - Temporal Stability: Smooth progression between frames
//
// The Math:
// The plastic constant g ≈ 1.324717 is the unique real solution to x³ = x + 1.
// This number has special properties that make it ideal for generating well-distributed
// 2D point sets. The R2 sequence is a recent discovery (2018) that provides better
// 2D coverage than traditional sequences like Halton or Sobol.
//
// Implementation:
// - Each pixel gets a unique sample based on its position
// - Samples change every frame using frameNumber offset
// - Sample index = pixelIndex + frameNumber * (width * height)
// - This ensures no correlation between adjacent pixels and temporal variation

// Math constants
#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

// R2 sequence constants
// g = plastic constant = solution to x^3 = x + 1
// Provides optimal 2D coverage with minimal discrepancy
__constant__ float R2_G = 1.32471795724474602596f;
__constant__ float R2_A1 = 0.7548776662466927f;  // 1/g
__constant__ float R2_A2 = 0.5698402909980532f;  // 1/(g*g)

// Helper function for fractional part
CUDA_DEVICE_FUNCTION CUDA_INLINE float fract(float x) {
    return x - floorf(x);
}

// R2 sequence generator
CUDA_DEVICE_FUNCTION CUDA_INLINE float2 R2Sequence(uint32_t index) {
    return make_float2(
        fract(R2_A1 * index),
        fract(R2_A2 * index)
    );
}

// basic math from https://github.com/jbikker/lighthouse2
// Samples a point on a polygonal lens shape (9-sided polygon for smooth bokeh)
CUDA_DEVICE_FUNCTION Vector3D sampleLensPoint (float r0, float r1, float lensSize)
{
    // Convert first random number to select polygon edge
    const float blade = (int)(r0 * 9);
    float r2 = (r0 - blade * (1.0f / 9.0f)) * 9.0f;

    // Get vertices of the selected edge
    float x1, y1, x2, y2;
    float angle1 = blade * M_PI / 4.5f;
    float angle2 = (blade + 1.0f) * M_PI / 4.5f;
    x1 = cos (angle1);
    y1 = sin (angle1);
    x2 = cos (angle2);
    y2 = sin (angle2);

    // Handle point reflection to ensure uniform sampling
    if ((r1 + r2) > 1)
    {
        r1 = 1.0f - r1;
        r2 = 1.0f - r2;
    }

    // Interpolate between vertices
    float x = x1 * r1 + x2 * r2;
    float y = y1 * r1 + y2 * r2;

    return Vector3D (x * lensSize, y * lensSize, 0.0f);
}

// basic math from https://github.com/jbikker/lighthouse2
// Generate camera ray with configurable focus distance and lens size
CUDA_DEVICE_FUNCTION void generateCameraRay (
    PCG32RNG& rng,
    const PerspectiveCamera& camera,
    const Point2D& pixel,
    Point3D* origin,
    Vector3D* direction)
{
    // Calculate position on image plane
    float h = 2.0f * std::tan (camera.fovY * 0.5f);
    float w = camera.aspect * h;
    Vector3D imagePlanePoint = (w * (0.5f - pixel.x)) * camera.orientation.c0 +
                               (h * (0.5f - pixel.y)) * camera.orientation.c1 +
                               camera.orientation.c2;

    if (camera.lensSize > 0.0f)
    {
        // Generate random point on lens
        float r0 = rng.getFloat0cTo1o();
        float r1 = rng.getFloat0cTo1o();
        Vector3D lensPoint = sampleLensPoint (r0, r1, camera.lensSize);

        // Set ray origin to sampled lens point
        *origin = camera.position +
                  lensPoint.x * camera.orientation.c0 +
                  lensPoint.y * camera.orientation.c1;

        // Calculate focus point at specified distance
        Point3D focusPoint = camera.position + imagePlanePoint * camera.focusDistance;
        *direction = normalize (focusPoint - *origin);
    }
    else
    {
        // Pinhole camera if lens size is 0
        *origin = camera.position;
        *direction = normalize (imagePlanePoint);
    }
}

// This function is for computing the direct lighting on a surface point (shadigPoint)
// from a light source (lightSample) using a BRDF (bsdf).
// It takes into account visibility, distances, and angles to compute the final light
// contribution at that point. Works for both environment and area lights.
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB computeDirectLighting (
    const Point3D& shadingPoint, const Vector3D& vOutLocal, const ReferenceFrame& shadingFrame,
    const DisneyPrincipled& bsdf, const milo_shared::LightSample& lightSample)
{
    // Calculate the direction of the shadow ray
    Vector3D shadowRayDir = lightSample.atInfinity ? Vector3D (lightSample.position) : (lightSample.position - shadingPoint);

    // Calculate the distance squared and distance between the light and the shading point
    float dist2 = shadowRayDir.sqLength();
    float dist = std::sqrt (dist2);

    // Normalize the shadow ray direction
    shadowRayDir /= dist;

    // Convert shadow ray direction to local coordinate system
    Vector3D shadowRayDirLocal = shadingFrame.toLocal (shadowRayDir);

    // Compute the cosine of the angle between the light direction and light normal
    float lpCos = dot (-shadowRayDir, lightSample.normal);

    // Compute the cosine of the angle between shadow ray and normal at the shading point in local coords
    float spCos = shadowRayDirLocal.z;

    // Initialize visibility to 1 (completely visible)
    float visibility = 1.0f;

    // Set a high distance for lights at infinity
    if (lightSample.atInfinity)
        dist = 1e+10f;

    // Perform visibility ray tracing to check if the light is occluded
    milo_shared::VisibilityRayPayloadSignature::trace (
        plp.travHandle,
        shadingPoint.toNative(), shadowRayDir.toNative(), 0.0f, dist * 0.9999f, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        RayType::RayType_Visibility, milo_shared::NumRayTypes, RayType::RayType_Visibility,
        visibility);

    // If the point is visible and faces the light
    if (visibility > 0 && lpCos > 0)
    {
        // Calculate emittance assuming the light is a diffuse emitter
        RGB Le = lightSample.emittance / Pi;

        // Evaluate the  BRDF
        RGB fsValue = bsdf.evaluate (vOutLocal, shadowRayDirLocal);

        // Calculate the geometry term
        float G = lpCos * std::fabs (spCos) / dist2;

        // Final lighting contribution
        RGB ret = fsValue * Le * G;
        return ret;
    }
    else
    {
        // Return black if the point is not visible or does not face the light
        return RGB (0.0f, 0.0f, 0.0f);
    }
}

// This function samples an environmental light based on a set of
// random numbers (u0 and u1) and an importance map.It returns the
// sampled light direction, emittance, and some other attributes
// in lightSample.It also returns the probability density of the
// sampled area in areaPDensity.
CUDA_DEVICE_FUNCTION CUDA_INLINE void sampleEnviroLight (
    const Point3D& shadingPoint,
    float ul, bool sampleEnvLight, float u0, float u1,
    milo_shared::LightSample* lightSample, float* areaPDensity)
{
    CUtexObject texEmittance = 0;          // Texture object for light emittance
    RGB emittance (0.0f, 0.0f, 0.0f); // Light emittance color
    Point2D texCoord;                      // Texture coordinates

    float u, v;  // Parameters for sampling
    float uvPDF; // PDF for UV sampling

    // Sample the importance map to get UV coordinates and PDF
    plp.envLightImportanceMap.sample (u0, u1, &u, &v, &uvPDF);

    // Convert UV to spherical coordinates
    float phi = 2 * Pi * u;
    float theta = Pi * v;
    if (theta == 0.0f)
    {
        // fix for NAN
        *areaPDensity = 0.0f;
        return;
    }

    // Apply rotation to the environment light
    float posPhi = phi - plp.envLightRotation;
    posPhi = posPhi - floorf (posPhi / (2 * Pi)) * 2 * Pi;

    // Convert spherical to Cartesian coordinates
    Vector3D direction = fromPolarYUp (posPhi, theta);
    Point3D position (direction.x, direction.y, direction.z);

    // Set light sample attributes
    lightSample->position = position;
    lightSample->atInfinity = true;
    lightSample->normal = Normal3D (-position);

    // convert the PDF in texture space to one with respect to area.
    // The true value is: lim_{l to inf} uvPDF / (2 * Pi * Pi * sin(theta)) / l^2
    const float sinTheta = std::sin (theta);
    if (sinTheta == 0.0f)
    {
        *areaPDensity = 0.0f;
        return;
    }

    // Compute the area PDF
    *areaPDensity = uvPDF / (2 * Pi * Pi * std::sin (theta));

    //  printf ("areaPDensity: %f\n", *areaPDensity);

    // Retrieve the environment light texture
    texEmittance = plp.envLightTexture;

    // Set a base emittance value
    emittance = RGB (Pi * plp.envLightPowerCoeff);
    texCoord.x = u;
    texCoord.y = v;

    // If a texture is available, update emittance based on texture values
    if (texEmittance)
    {
        float4 texValue = tex2DLod<float4> (texEmittance, texCoord.x, texCoord.y, 0.0f);
        emittance *= RGB (texValue.x, texValue.y, texValue.z);

        if (isnan (emittance.r) || isnan (emittance.g) || isnan (emittance.b))
        {
            printf ("enviro texture emittance: %f, %f, %f\n", emittance.r, emittance.g, emittance.b);
        }
    }

    // Set the emittance in the light sample
    lightSample->emittance = emittance;
}

// Sample area lights in the scene
CUDA_DEVICE_FUNCTION CUDA_INLINE void sampleAreaLight(
    const Point3D& shadingPoint,
    float ul, float u0, float u1,
    milo_shared::LightSample* lightSample, float* areaPDensity) 
{
    *areaPDensity = 0.0f;
    
    if (!plp.enableAreaLights || plp.numLightInsts == 0) {
        return;
    }
    
    float lightProb = 1.0f;
    
    // First, sample an instance from the light instance distribution
    float instProb;
    float uGeomInst;
    const uint32_t instSlot = plp.lightInstDist.sample(ul, &instProb, &uGeomInst);
    lightProb *= instProb;
    
    if (instProb == 0.0f) {
        return;
    }
    
    const shared::InstanceData& inst = plp.instanceDataBufferArray[plp.bufferIndex][instSlot];
    
    // Next, sample a geometry instance from this instance
    float geomInstProb;
    float uPrim;
    const uint32_t geomInstIndexInInst = inst.lightGeomInstDist.sample(uGeomInst, &geomInstProb, &uPrim);
    const uint32_t geomInstSlot = inst.geomInstSlots[geomInstIndexInInst];
    lightProb *= geomInstProb;
    
    if (geomInstProb == 0.0f) {
        return;
    }
    
    const shared::GeometryInstanceData& geomInst = plp.geometryInstanceDataBuffer[geomInstSlot];
    
    // Finally, sample a primitive from the geometry instance
    float primProb;
    const uint32_t primIndex = geomInst.emitterPrimDist.sample(uPrim, &primProb);
    lightProb *= primProb;
    
    // Get the triangle and its vertices
    const shared::Triangle& tri = geomInst.triangleBuffer[primIndex];
    const shared::Vertex& vA = geomInst.vertexBuffer[tri.index0];
    const shared::Vertex& vB = geomInst.vertexBuffer[tri.index1];
    const shared::Vertex& vC = geomInst.vertexBuffer[tri.index2];
    
    // Transform vertices to world space
    const Point3D pA = transformPointFromObjectToWorldSpace(vA.position);
    const Point3D pB = transformPointFromObjectToWorldSpace(vB.position);
    const Point3D pC = transformPointFromObjectToWorldSpace(vC.position);
    
    // Sample point on triangle using uniform barycentric sampling
    float sqrtU0 = sqrtf(u0);
    float bc0 = 1.0f - sqrtU0;
    float bc1 = u1 * sqrtU0;
    float bc2 = 1.0f - bc0 - bc1;
    
    // Compute sampled position
    lightSample->position = bc0 * pA + bc1 * pB + bc2 * pC;
    
    // Compute normal (average of vertex normals, transformed to world)
    Normal3D nA = transformNormalFromObjectToWorldSpace(vA.normal);
    Normal3D nB = transformNormalFromObjectToWorldSpace(vB.normal);
    Normal3D nC = transformNormalFromObjectToWorldSpace(vC.normal);
    lightSample->normal = normalize(bc0 * nA + bc1 * nB + bc2 * nC);
    
    // Get material emittance
    const shared::DisneyData& mat = plp.materialDataBuffer[geomInst.materialSlot];
    
    // Sample emittance texture if available
    Point2D texCoord = bc0 * vA.texCoord + bc1 * vB.texCoord + bc2 * vC.texCoord;
    if (mat.emissive) {
        float4 texValue = tex2DLod<float4>(mat.emissive, texCoord.x, texCoord.y, 0.0f);
        lightSample->emittance = RGB(texValue.x, texValue.y, texValue.z) * plp.areaLightPowerCoeff;
    } else {
        lightSample->emittance = RGB(0.0f, 0.0f, 0.0f);
    }
    
    lightSample->atInfinity = false;
    
    // Compute area of the triangle in world space
    Vector3D edge1 = pB - pA;
    Vector3D edge2 = pC - pA;
    float area = 0.5f * length(cross(edge1, edge2));
    
    // Final area PDF
    *areaPDensity = lightProb / area;
}

// Next Event Estimation (NEE) is a technique used in path tracing to improve
// the convergence of the rendered image. Instead of randomly bouncing rays around the scene,
// NEE takes a shortcut and directly samples a light source to check if it contributes to
// the illumination of a point.

// In a traditional path tracer, rays are shot from the camera and bounce around the scene
// until they hit a light source. This can take many bounces and lead to a noisy image.

// With NEE, when a ray hits a surface, the algorithm also sends a direct ray to a light source
// to see if it's visible from that point. This helps to quickly account for direct illumination,
// making the image converge faster and reducing noise.

// This function is for performing Next Event Estimation (NEE) in path tracing.
// It samples a light source, computes the direct lighting from that source,
// and combines it with the BRDF and visibility information.The function also
// uses Multiple Importance Sampling (MIS)
// to balance the contributions from the BRDF and the light source.
CUDA_DEVICE_FUNCTION CUDA_INLINE RGB performNextEventEstimation (
    const Point3D& shadingPoint, const Vector3D& vOutLocal, const ReferenceFrame& shadingFrame,
    const DisneyPrincipled& bsdf,
    PCG32RNG& rng)
{
    RGB ret (0.0f); // Initialize the return value

    // Determine light type sampling probability based on availability
    float envLightProb = 0.0f;
    float areaLightProb = 0.0f;
    
    if (plp.enableEnvLight && plp.envLightTexture) {
        envLightProb = 0.5f;  // Could be based on relative power
    }
    
    if (plp.enableAreaLights && plp.numLightInsts > 0) {
        areaLightProb = 0.5f;
    }
    
    // Normalize probabilities
    float totalProb = envLightProb + areaLightProb;
    if (totalProb == 0.0f) {
        return RGB(0.0f);  // No lights available
    }
    
    envLightProb /= totalProb;
    areaLightProb /= totalProb;
    
    // Select light type
    float uLightType = rng.getFloat0cTo1o();
    bool selectEnvLight = uLightType < envLightProb;
    
    milo_shared::LightSample lightSample; // Sampled light information
    float areaPDensity = 0.0f;      // Area probability density
    float probToSampleCurLightType = selectEnvLight ? envLightProb : areaLightProb;
    
    if (selectEnvLight && envLightProb > 0.0f) {
        // Sample environment light
        sampleEnviroLight (
            shadingPoint,
            rng.getFloat0cTo1o(), true, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
            &lightSample, &areaPDensity);
    } else if (areaLightProb > 0.0f) {
        // Sample area light
        sampleAreaLight(
            shadingPoint,
            rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
            &lightSample, &areaPDensity);
    }
    
    if (areaPDensity <= 0.0f) {
        return RGB(0.0f);
    }

    areaPDensity *= probToSampleCurLightType; // Update the area PDF with the light type selection probability

    // Calculate the shadow ray direction
    Vector3D shadowRay = lightSample.atInfinity ? Vector3D (lightSample.position) : (lightSample.position - shadingPoint);
    float dist2 = shadowRay.sqLength();                   // Distance squared to the light
    shadowRay /= std::sqrt (dist2);                       // Normalize the shadow ray
    Vector3D vInLocal = shadingFrame.toLocal (shadowRay); // Convert to local coordinates

    float bsdfPDensity = bsdf.evaluatePDF (vOutLocal, vInLocal);
    if (!isfinite (bsdfPDensity) || bsdfPDensity <= 0.0f)
    {
        return RGB (0.0f); // Invalid sampling case, skip contribution
    }

    // Calculate the light source PDF and MIS weight
    float lightPDensity = areaPDensity;
    float misWeight = pow2 (lightPDensity) / (pow2 (bsdfPDensity) + pow2 (lightPDensity));

    // Compute the direct lighting contribution if the area PDF is positive
    if (areaPDensity > 0.0f)
        ret = computeDirectLighting (
                  shadingPoint, vOutLocal, shadingFrame, bsdf, lightSample) *
              (misWeight / areaPDensity);

    return ret; // Return the final lighting contribution
}

// This function calculates various attributes of a surface point
// given its barycentric coordinates (b1, b2) and the index (primIndex)
// of the triangle it belongs to. It computes the world-space position,
// shading normal, texture coordinates, and so forth for this surface point.
// It also computes a hypothetical area PDF (hypAreaPDensity) that could
// be used in light sampling.
CUDA_DEVICE_FUNCTION CUDA_INLINE void computeSurfacePoint (
    const shared::GeometryInstanceData& geomInst,
    uint32_t primIndex, float b1, float b2,
    const Point3D& referencePoint,
    Point3D* positionInWorld, Normal3D* shadingNormalInWorld, Vector3D* texCoord0DirInWorld,
    Normal3D* geometricNormalInWorld, Point2D* texCoord,
    float* hypAreaPDensity)
{
    // Fetch the vertices of the triangle given its index
    const Triangle& tri = geomInst.triangleBuffer[primIndex];
    const Vertex& v0 = geomInst.vertexBuffer[tri.index0];
    const Vertex& v1 = geomInst.vertexBuffer[tri.index1];
    const Vertex& v2 = geomInst.vertexBuffer[tri.index2];

    // Transform vertex positions to world space
    const Point3D p[3] = {
        transformPointFromObjectToWorldSpace (v0.position),
        transformPointFromObjectToWorldSpace (v1.position),
        transformPointFromObjectToWorldSpace (v2.position),
    };

    // Calculate barycentric coordinates
    float b0 = 1 - (b1 + b2);

    // Compute the position in world space using barycentric coordinates
    *positionInWorld = b0 * p[0] + b1 * p[1] + b2 * p[2];

    // Compute interpolated shading normal and texture direction
    Normal3D shadingNormal = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
    Vector3D texCoord0Dir = b0 * v0.texCoord0Dir + b1 * v1.texCoord0Dir + b2 * v2.texCoord0Dir;

    // Compute geometric normal and area of the triangle
    Normal3D geometricNormal (cross (p[1] - p[0], p[2] - p[0]));
    float area = 0.5f * length (geometricNormal);

    // Compute the texture coordinates
    *texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

    // Transform shading normal and texture direction to world space
    *shadingNormalInWorld = normalize (transformNormalFromObjectToWorldSpace (shadingNormal));
    *texCoord0DirInWorld = normalize (transformVectorFromObjectToWorldSpace (texCoord0Dir));
    *geometricNormalInWorld = normalize (geometricNormal);

    // Check for invalid normals and give them a default value
    if (!shadingNormalInWorld->allFinite())
    {
        *shadingNormalInWorld = Normal3D (0, 0, 1);
        *texCoord0DirInWorld = Vector3D (1, 0, 0);
    }

    // Check for invalid texture directions and correct them
    if (!texCoord0DirInWorld->allFinite())
    {
        Vector3D bitangent;
        makeCoordinateSystem (*shadingNormalInWorld, texCoord0DirInWorld, &bitangent);
    }

    // Compute the probability of sampling this light
    float lightProb = 1.0f;
    if (plp.envLightTexture && plp.enableEnvLight)
        lightProb *= (1 - probToSampleEnvLight);

    // Check for invalid probabilities
    if (!isfinite (lightProb))
    {
        *hypAreaPDensity = 0.0f;
        return;
    }

    // Compute the hypothetical area PDF
    *hypAreaPDensity = lightProb / area;
}

// Define a struct called HitPointParameter to hold hit point info
struct HitPointParameter
{
    float b1, b2;      // Barycentric coordinates
    int32_t primIndex; // Index of the primitive hit by the ray

    // Static member function to get hit point parameters
    CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get()
    {
        HitPointParameter ret; // Create an instance of the struct

        // Get barycentric coordinates from OptiX API
        float2 bc = optixGetTriangleBarycentrics();

        // Store the barycentric coordinates in the struct
        ret.b1 = bc.x;
        ret.b2 = bc.y;

        // Get the index of the primitive hit by the ray from OptiX API
        ret.primIndex = optixGetPrimitiveIndex();

        // Return the populated struct
        return ret;
    }
};

// This struct is used to fetch geometry instance and material data from
// the Shader Binding Table (SBT) in OptiX.
struct HitGroupSBTRecordData
{
    uint32_t geomInstSlot;      // Geometry instance slot index in the global buffer
    uint32_t materialSlot;      // Material slot index in the material buffer

    // Static member function to retrieve the SBT record data
    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData& get()
    {
        // Use optixGetSbtDataPointer() to get the pointer to the SBT data
        // Cast the pointer to type HitGroupSBTRecordData and dereference it
        return *reinterpret_cast<HitGroupSBTRecordData*> (optixGetSbtDataPointer());
    }
};

// Define the ray generating kernel for path tracing
CUDA_DEVICE_KERNEL void RT_RG_NAME (pathTracing)()
{
    // Get the launch index for this thread
    uint2 launchIndex = make_uint2 (optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    // Initialize the random number generator
    PCG32RNG rng = plp.rngBuffer[launchIndex];

    Point3D origin;
    Vector3D direction;
    const PerspectiveCamera& camera = plp.camera;

    // different approach for DOF
    if (plp.camera.lensSize > 0.0f)
    {
        // Use R2 sequence for pixel sampling with DOF
        uint32_t pixelIndex = launchIndex.y * plp.imageSize.x + launchIndex.x;
        uint32_t sampleIndex = pixelIndex + plp.numAccumFrames * (plp.imageSize.x * plp.imageSize.y);
        float2 r2Sample = R2Sequence(sampleIndex);
        
        Point2D pixel (
            (launchIndex.x + r2Sample.x) / plp.imageSize.x,
            (launchIndex.y + r2Sample.y) / plp.imageSize.y);

        generateCameraRay (rng, plp.camera, pixel, &origin, &direction);
    }
    else
    {
        // Generate jitter offsets using R2 sequence for better distribution
        uint32_t pixelIndex = launchIndex.y * plp.imageSize.x + launchIndex.x;
        uint32_t sampleIndex = pixelIndex + plp.numAccumFrames * (plp.imageSize.x * plp.imageSize.y);
        float2 r2Sample = R2Sequence(sampleIndex);
        
        float jx = r2Sample.x;
        float jy = r2Sample.y;

        // Update the RNG buffer (still needed for other sampling)
        plp.rngBuffer.write (launchIndex, rng);

        // Compute normalized screen coordinates
        float x = (launchIndex.x + jx) / plp.imageSize.x;
        float y = (launchIndex.y + jy) / plp.imageSize.y;

        // Compute vertical and horizontal view angles
        float vh = 2 * std::tan (plp.camera.fovY * 0.5f);
        float vw = plp.camera.aspect * vh;

        // Setup ray origin and direction
        origin = camera.position;
        direction = normalize (camera.orientation * Vector3D (vw * (0.5f - x), vh * (0.5f - y), 1));
    }
    
    // Debug: Print traversable handle and ray info for first pixel (similar to Shocker)
    if (launchIndex.x == 0 && launchIndex.y == 0) {
        printf("MiloEngine RG: travHandle=%llu, origin=(%.2f,%.2f,%.2f), dir=(%.2f,%.2f,%.2f)\n",
               plp.travHandle, origin.x, origin.y, origin.z, direction.x, direction.y, direction.z);
    }

    // Initialize ray payload
    SearchRayPayload payload;
    payload.alpha = RGB (1.0f, 1.0f, 1.0f);
    payload.contribution = RGB (0.0f, 0.0f, 0.0f);
    payload.pathLength = 1;
    payload.prevDirPDensity = 1.0f;  // Camera rays have uniform PDF
    payload.deltaSampled = 0;
    payload.terminate = false;
    SearchRayPayload* payloadPtr = &payload;

    RGB firstHitAlbedo (0.0f, 0.0f, 0.0f);
    Normal3D firstHitNormal (0.0f, 0.0f, 0.0f);
    RGB* firstHitAlbedoPtr = &firstHitAlbedo;
    Normal3D* firstHitNormalPtr = &firstHitNormal;

    // Initialize variables for storing hit point properties
    HitPointParams hitPointParams;
    hitPointParams.positionInWorld = Point3D (NAN);
    hitPointParams.prevPositionInWorld = Point3D (NAN);
    hitPointParams.normalInWorld = Normal3D (NAN);
    hitPointParams.texCoord = Point2D (NAN);
    HitPointParams* hitPointParamsPtr = &hitPointParams;

    // Main path tracing loop
    while (true)
    {
        // Trace the ray and collect results
        SearchRayPayloadSignature::trace (
            plp.travHandle, origin.toNative(), direction.toNative(),
            0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Search, NumRayTypes, RayType_Search,
            rng, payloadPtr, hitPointParamsPtr, firstHitAlbedoPtr, firstHitNormalPtr);

        // Break out of the loop if conditions are met
        if (payload.terminate || payload.pathLength >= plp.bounceLimit)
            break;

        // Update ray origin and direction for the next iteration
        origin = payload.origin;
        direction = payload.direction;
        ++payload.pathLength;
    }

    // Store the updated RNG state back to the buffer
    plp.rngBuffer[launchIndex] = rng;

    RGB prevAlbedoResult (0.0f, 0.0f, 0.0f);
    RGB prevColorResult (0.0f, 0.0f, 0.0f);
    Normal3D prevNormalResult (0.0f, 0.0f, 0.0f);

    if (plp.numAccumFrames > 0)
    {
        prevColorResult = RGB (getXYZ (plp.colorAccumBuffer.read (launchIndex)));
        prevAlbedoResult = RGB (getXYZ (plp.albedoAccumBuffer.read (launchIndex)));
        prevNormalResult = Normal3D (getXYZ (plp.normalAccumBuffer.read (launchIndex)));
    }

    float curWeight = 1.0f / (1 + plp.numAccumFrames);

    // Clamp contribution to reduce fireflies
    RGB clampedContribution = payload.contribution;
    clampedContribution.r = fminf(clampedContribution.r, plp.maxRadiance);
    clampedContribution.g = fminf(clampedContribution.g, plp.maxRadiance);
    clampedContribution.b = fminf(clampedContribution.b, plp.maxRadiance);

    RGB colorResult = (1 - curWeight) * prevColorResult + curWeight * clampedContribution;
#if 0
    if (isnan (colorResult.r) || isnan (colorResult.g) || isnan (colorResult.b))
    {
        // Add this line to print the payload.contribution values
        printf ("payload.contribution: %f, %f, %f\n", payload.contribution.r, payload.contribution.g, payload.contribution.b);
        colorResult = RGB (make_float3 (1000000.0f, 0.0f, 0.0f)); // super red
    }
    else if (isinf (colorResult.r) || isinf (colorResult.g) || isinf (colorResult.b))
    {
        printf ("payload.contribution: %f, %f, %f\n", payload.contribution.r, payload.contribution.g, payload.contribution.b);
        colorResult = RGB (make_float3 (0.0f, 1000000.0f, 0.0f)); // super green
    }
    else if (colorResult.r < 0.0f || colorResult.g < 0.0f || colorResult.b < 0.0f)
    {
        printf ("payload.contribution is negative: %f, %f, %f\n", payload.contribution.r, payload.contribution.g, payload.contribution.b);
        colorResult = RGB (make_float3 (0.0f, 0.0f, 1000000.0f)); // super blue
    }
#endif
    RGB albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * firstHitAlbedo;

#if 0
    if (albedoResult.r < 0.0f || albedoResult.r > 1.0f
        || albedoResult.g < 0.0f || albedoResult.g > 1.0f 
        || albedoResult.b < 0.0f || albedoResult.b > 1.0f)
        {
            // Add this line to print the payload.contribution values
            printf ("firstHitAlbedo  %f, %f, %f\n", firstHitAlbedo.r, firstHitAlbedo.g, firstHitAlbedo.b);
            albedoResult = RGB (make_float3 (1000000.0f, 0.0f, 0.0f)); // super red
        }

#endif

    Normal3D normalResult = (1 - curWeight) * prevNormalResult + curWeight * firstHitNormal;
#if 0
    if (isnan (normalResult.x) || isnan (normalResult.y) || isnan (normalResult.z))
    {
        // Add this line to print the payload.contribution values
        printf ("firstHitNormal: %f, %f, %f\n", firstHitNormal.x, firstHitNormal.y, firstHitNormal.z);
            normalResult = Normal3D (make_float3 (1000000.0f, 0.0f, 0.0f)); // super red
    }
#endif
    plp.colorAccumBuffer.write (launchIndex, make_float4 (colorResult.toNative(), 1.0f));
    plp.albedoAccumBuffer.write (launchIndex, make_float4 (albedoResult.toNative(), 1.0f));
    plp.normalAccumBuffer.write (launchIndex, make_float4 (normalResult.toNative(), 1.0f));
    
    // Calculate motion vectors
    Vector2D motionVector (0.0f, 0.0f);
    if (!isnan (hitPointParams.positionInWorld.x) && !isnan (hitPointParams.prevPositionInWorld.x))
    {
        // Current pixel position (center of pixel)
        Point2D curRasterPos (launchIndex.x + 0.5f, launchIndex.y + 0.5f);
        
        // Calculate previous frame position using previous camera
        Point2D prevRasterPos = plp.prevCamera.calcScreenPosition (hitPointParams.prevPositionInWorld) 
                               * Point2D (plp.imageSize.x, plp.imageSize.y);
        
        // Motion vector is the difference
        motionVector = curRasterPos - prevRasterPos;
    }
    
    // Write motion vector to flow accumulation buffer
    plp.flowAccumBuffer.write (launchIndex, make_float4 (motionVector.x, motionVector.y, 0.0f, 1.0f));
}
// Miss shader that handles environment lighting and background
CUDA_DEVICE_KERNEL void RT_MS_NAME (miss)()
{
    // Get payload data
    SearchRayPayload* payload;
    HitPointParams* hitPntParams;
    SearchRayPayloadSignature::get (nullptr, &payload, &hitPntParams, nullptr, nullptr);

    // Store normalized direction as surface normal
    Vector3D vOut (-Vector3D (optixGetWorldRayDirection()));
    hitPntParams->normalInWorld = Normal3D (vOut);

    // Calculate raw HDR environment value without power coefficient
    RGB environmentValue (0.0f, 0.0f, 0.0f);
    float theta = 0.0f;
    Point2D texCoord (0.0f, 0.0f);

    if (plp.envLightTexture)
    {
        Vector3D rayDir = normalize (Vector3D (optixGetWorldRayDirection()));
        float posPhi;
        toPolarYUp (rayDir, &posPhi, &theta);
        float phi = posPhi + plp.envLightRotation;
        phi = phi - floorf (phi / (2 * Pi)) * 2 * Pi;
        texCoord = Point2D (phi / (2 * Pi), theta / Pi);
        float4 texValue = tex2DLod<float4> (plp.envLightTexture, texCoord.x, texCoord.y, 0.0f);
        environmentValue = RGB (texValue.x, texValue.y, texValue.z);
    }

    // For background color, use raw HDR or solid color without power coefficient
    RGB background;
    if (plp.useSolidBackground || !plp.envLightTexture)
    {
        background = RGB (plp.backgroundColor.x, plp.backgroundColor.y, plp.backgroundColor.z);
    }
    else
    {
        background = environmentValue; // Use raw environment value for background
    }

    // Apply MIS weight and power coefficient for surface lighting only
    float misWeight = 1.0f;
    if (payload->pathLength > 1 && !payload->deltaSampled)
    {
        float uvPDF = plp.envLightImportanceMap.evaluatePDF (texCoord.x, texCoord.y);
        float hypAreaPDensity = uvPDF / (2 * Pi * Pi * std::sin (theta));
        float lightPDensity = hypAreaPDensity;
        if (plp.lightInstDist.integral() > 0.0f)
        {
            lightPDensity *= probToSampleEnvLight;
        }
        float bsdfPDensity = 0.25f;
        misWeight = pow2 (bsdfPDensity) / (pow2 (bsdfPDensity) + pow2 (lightPDensity));

        // Apply power coefficient only for surface lighting
        payload->contribution += payload->alpha * (environmentValue * plp.envLightPowerCoeff) * misWeight;
    }
    else
    {
        // First bounce - use raw background without power coefficient
       payload->contribution = background;
       // payload->contribution = RGB(1.0f, 0.0f, 0.0f) * plp.envLightPowerCoeff;
    }

    payload->terminate = true;
}

CUDA_DEVICE_KERNEL void RT_CH_NAME (shading)()
{
    // Get material and geometry instance data from global buffers
    auto sbtr = HitGroupSBTRecordData::get();
    const shared::DisneyData& mat = plp.materialDataBuffer[sbtr.materialSlot];
    const shared::GeometryInstanceData& geomInst = plp.geometryInstanceDataBuffer[sbtr.geomInstSlot];
    
    // Get instance data using buffer index from launch parameters
    const uint32_t bufIdx = plp.bufferIndex;  
    const shared::InstanceData& inst = plp.instanceDataBufferArray[bufIdx][optixGetInstanceId()];

    // Initialize random number generator and payload
    PCG32RNG rng;
    SearchRayPayload* payload;
    RGB* firstHitAlbedo;
    Normal3D* firstHitNormal;
    HitPointParams* hitPntParams;
    SearchRayPayloadSignature::get (&rng, &payload, &hitPntParams, &firstHitAlbedo, &firstHitNormal);

    // Calculate hit point parameters
    auto hp = HitPointParameter::get();
    Point3D positionInWorld;
    Normal3D shadingNormalInWorld;
    Vector3D texCoord0DirInWorld;
    Normal3D geometricNormalInWorld;
    Point2D texCoord;
    float hypAreaPDensity;
    computeSurfacePoint (
        geomInst, hp.primIndex, hp.b1, hp.b2,
        Point3D (optixGetWorldRayOrigin()),
        &positionInWorld, &shadingNormalInWorld, &texCoord0DirInWorld,
        &geometricNormalInWorld, &texCoord, &hypAreaPDensity);

    // Setup shading frame
    Vector3D vOut = normalize (-Vector3D (optixGetWorldRayDirection()));
    float frontHit = dot (vOut, geometricNormalInWorld) >= 0.0f ? 1.0f : -1.0f;
    ReferenceFrame shadingFrame (shadingNormalInWorld, texCoord0DirInWorld);

    // Offset hit point to avoid self-intersection
    positionInWorld = offsetRayOrigin (positionInWorld, frontHit * geometricNormalInWorld);
    Vector3D vOutLocal = shadingFrame.toLocal (vOut);
    
    // Calculate previous position for motion vectors (only on first hit)
    if (payload->pathLength == 1)
    {
        hitPntParams->positionInWorld = positionInWorld;
        hitPntParams->prevPositionInWorld = inst.curToPrevTransform * positionInWorld;
        hitPntParams->normalInWorld = shadingNormalInWorld;
        hitPntParams->texCoord = texCoord;
    }

    // Create DisneyPrincipled instance directly instead of using BSDF
    DisneyPrincipled bsdf = DisneyPrincipled::create (
        mat, texCoord, 0.0f, plp.makeAllGlass, plp.globalGlassIOR,
        plp.globalTransmittanceDist, plp.globalGlassType);

    // Delta in PBR rendering refers to a perfect specular reflection or transmission that occurs at a single angle.
    // It represents an infinitely narrow spike of reflection, like what you'd see in a perfect mirror, where all light
    // reflects at exactly the angle predicted by the law of reflection. In rendering systems, delta distributions are
    // handled as special cases since they can't be sampled like regular BRDFs. They're primarily used to model
    // idealized surfaces like perfect mirrors, smooth glass, and pristine metals.
    bool isDeltaMaterial = mat.transparency > 0.9f && mat.metallic <= 0.0f;

    // Handle emissive surfaces
    RGB emission = bsdf.evaluateEmission();
    if (emission.r > 0.0f || emission.g > 0.0f || emission.b > 0.0f)
    {
        if (payload->pathLength == 1)
        {
            // Direct camera hit - no MIS needed
            payload->contribution += payload->alpha * emission;
        }
        else if (plp.enableAreaLights && !payload->deltaSampled)
        {
            // Indirect hit with MIS
            // We need to compute the probability of having sampled this light via NEE
            
            // First check if this instance is emissive
            if (inst.isEmissive)
            {
                // Compute area of this triangle
                const shared::Triangle& tri = geomInst.triangleBuffer[hp.primIndex];
                const shared::Vertex& v0 = geomInst.vertexBuffer[tri.index0];
                const shared::Vertex& v1 = geomInst.vertexBuffer[tri.index1];
                const shared::Vertex& v2 = geomInst.vertexBuffer[tri.index2];
                
                Point3D p0 = transformPointFromObjectToWorldSpace(v0.position);
                Point3D p1 = transformPointFromObjectToWorldSpace(v1.position);
                Point3D p2 = transformPointFromObjectToWorldSpace(v2.position);
                
                float area = 0.5f * length(cross(p1 - p0, p2 - p0));
                
                // Get the various sampling probabilities
                float instProb = plp.lightInstDist.evaluatePMF(optixGetInstanceId());
                float geomInstProb = inst.lightGeomInstDist.evaluatePMF(0); // Assuming single geom inst
                float primProb = geomInst.emitterPrimDist.evaluatePMF(hp.primIndex);
                
                // Light sampling PDF (area measure)
                float lightPDF = instProb * geomInstProb * primProb / area;
                
                // Account for light type selection probability
                float envLightProb = (plp.enableEnvLight && plp.envLightTexture) ? 0.5f : 0.0f;
                float areaLightProb = 1.0f - envLightProb;
                lightPDF *= areaLightProb;
                
                // Convert to solid angle measure
                Point3D prevPos = payload->origin - payload->direction * 0.001f; // Approximate previous position
                Vector3D toLight = positionInWorld - prevPos;
                float dist2 = toLight.sqLength();
                float cosTheta = std::abs(dot(normalize(toLight), geometricNormalInWorld));
                lightPDF *= dist2 / std::max(cosTheta, 1e-6f);
                
                // BSDF PDF from previous direction
                float bsdfPDF = payload->prevDirPDensity;
                
                // MIS weight (power heuristic)
                float misWeight = bsdfPDF * bsdfPDF / (bsdfPDF * bsdfPDF + lightPDF * lightPDF);
                
                payload->contribution += payload->alpha * emission * misWeight;
            }
            else
            {
                // Non-emissive instance hit but material is emissive - use full contribution
                payload->contribution += payload->alpha * emission;
            }
        }
        else
        {
            // Delta sampled or area lights disabled - no MIS
            payload->contribution += payload->alpha * emission;
        }
    }
    // Only do NEE for non-delta materials
    if (!isDeltaMaterial)
    {
        payload->contribution += payload->alpha * performNextEventEstimation (positionInWorld, vOutLocal, shadingFrame, bsdf, rng);
    }

    // Sample new direction
    Vector3D vInLocal;
    float dirPDensity;
    RGB sampledValue = bsdf.sampleThroughput (
        vOutLocal, rng.getFloat0cTo1o(), rng.getFloat0cTo1o(),
        &vInLocal, &dirPDensity);

    if (dirPDensity > 0.0f)
    {
        // Update payload for next bounce
        payload->alpha = payload->alpha * (sampledValue * std::fabs (vInLocal.z) / dirPDensity);
        payload->origin = positionInWorld + shadingNormalInWorld * (vInLocal.z > 0 ? 0.001f : -0.001f);
        payload->direction = shadingFrame.fromLocal (vInLocal);
        payload->prevDirPDensity = dirPDensity;
        payload->deltaSampled = isDeltaMaterial;
        payload->terminate = false;

        // Store first hit data
        if (payload->pathLength == 1)
        {
            if (isDeltaMaterial)
            {
                // For perfectly transparent materials, use base color
                *firstHitAlbedo = RGB (mat.baseColor);
            }
            else
            {
                *firstHitAlbedo = bsdf.evaluateDHReflectanceEstimate (vOutLocal);
            }
            *firstHitNormal = shadingNormalInWorld;
        }
    }
    else
    {
        payload->terminate = true;
    }

    SearchRayPayloadSignature::set (&rng, nullptr, nullptr, nullptr, nullptr);
}


// Determines how light passes through transparent objects for more accurate shadows
CUDA_DEVICE_KERNEL void RT_AH_NAME (visibility)()
{
    // Get material and geometry instance data
    auto sbtr = HitGroupSBTRecordData::get();
    const shared::DisneyData& mat = plp.materialDataBuffer[sbtr.materialSlot];
    const shared::GeometryInstanceData& geomInst = plp.geometryInstanceDataBuffer[sbtr.geomInstSlot];

    // Get barycentric coordinates
    float2 bc = optixGetTriangleBarycentrics();

    // Get UV coordinates from hit point
    auto hp = HitPointParameter::get();
    const Triangle& tri = geomInst.triangleBuffer[hp.primIndex];
    const Vertex& v0 = geomInst.vertexBuffer[tri.index0];
    const Vertex& v1 = geomInst.vertexBuffer[tri.index1];
    const Vertex& v2 = geomInst.vertexBuffer[tri.index2];

    float b0 = 1.0f - (bc.x + bc.y);
    Point2D texCoord = b0 * v0.texCoord + bc.x * v1.texCoord + bc.y * v2.texCoord;

    // Read material properties at this point
    float transparency = tex2DLod<float> (mat.transparency, texCoord.x, texCoord.y, 0.0f);
    if (mat.useAlphaForTransparency)
    {
        float4 baseColorValue = tex2DLod<float4> (mat.baseColor, texCoord.x, texCoord.y, 0.0f);
        float alpha = baseColorValue.w;

        // For binary alpha, use threshold approach
        float alphaThreshold = 0.5f;
        if (alpha < alphaThreshold)
        {
            // Make fully transparent
            transparency = 1.0f;
        }
        else if (transparency < 0.1f)
        {
            // Only override if not already transparent
            transparency = 0.0f;
        }
    }
    float transmittance = tex2DLod<float> (mat.transmittance, texCoord.x, texCoord.y, 0.0f);
    float transmittanceDistance = tex2DLod<float> (mat.transmittanceDistance, texCoord.x, texCoord.y, 0.0f);
    transmittanceDistance = 0.5f;
    float4 baseColorValue = tex2DLod<float4> (mat.baseColor, texCoord.x, texCoord.y, 0.0f);
    RGB baseColor (baseColorValue.x, baseColorValue.y, baseColorValue.z);

    // Get current visibility value
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::get (&visibility);

  
    // Skip if the material is opaque (no transparency)
    if (transparency <= 0.0f)
    {
        visibility = 0.0f;
        VisibilityRayPayloadSignature::set (&visibility);
        optixTerminateRay();
        return;
    }

    // Calculate ray direction and normal for Fresnel calculations
    Vector3D rayDir = normalize (Vector3D (optixGetWorldRayDirection()));
    Normal3D normal = normalize (b0 * v0.normal + bc.x * v1.normal + bc.y * v2.normal);
    normal = normalize (transformNormalFromObjectToWorldSpace (normal));

    // Ensure normal faces against ray direction
    float NdotI = dot (normal, rayDir);
    if (NdotI > 0.0f)
        normal = -normal;

    // Calculate Fresnel for incident ray
    float cosTheta = abs (dot (normal, rayDir));
    float ior = plp.globalGlassIOR; // Use global glass IOR from pipeline params
    float F = mx_fresnel_dielectric (cosTheta, ior);

    // Calculate how much light passes through (transmission)
    float transmission = (1.0f - F) * transparency * transmittance;

    // Apply color absorption using Beer's law if not thin-walled
    bool thinWalled = (plp.globalGlassType == 0);
    if (!thinWalled && transmittanceDistance > 0.0f)
    {
        // Estimate approximate ray distance through the material
        // This is a simplification; for accurate results we'd need entry/exit points
        float estDistance = transmittanceDistance;

        // Apply Beer's law: T = exp(-absorption * distance)
        RGB transmissionColor;
        transmissionColor.r = exp (-baseColor.r * estDistance);
        transmissionColor.g = exp (-baseColor.g * estDistance);
        transmissionColor.b = exp (-baseColor.b * estDistance);

        // Convert RGB transmission to scalar (using luminance formula)
        float coloredTransmission = 0.2126f * transmissionColor.r +
                                    0.7152f * transmissionColor.g +
                                    0.0722f * transmissionColor.b;

        // Apply colored absorption to transmission
        transmission *= coloredTransmission;
    }

    // Update visibility based on transmission
    visibility *= transmission;
    VisibilityRayPayloadSignature::set (&visibility);

    // Continue ray if we still have meaningful visibility
    if (visibility > 0.01f)
        return;

    // Terminate ray if visibility too low
    visibility = 0.0f;
    VisibilityRayPayloadSignature::set (&visibility);
    optixTerminateRay();
}


#if 0
// FIXME this needs work.
CUDA_DEVICE_KERNEL void RT_AH_NAME (visibility)()
{
    // Get material and geometry instance data
    auto sbtr = HitGroupSBTRecordData::get();
    const shared::DisneyData& mat = plp.materialDataBuffer[sbtr.materialSlot];
    const shared::GeometryInstanceData& geomInst = plp.geometryInstanceDataBuffer[sbtr.geomInstSlot];

    // Get barycentric coordinates
    float2 bc = optixGetTriangleBarycentrics();

    // Get UV coordinates from hit point
    auto hp = HitPointParameter::get();
    const Triangle& tri = geomInst.triangleBuffer[hp.primIndex];
    const Vertex& v0 = geomInst.vertexBuffer[tri.index0];
    const Vertex& v1 = geomInst.vertexBuffer[tri.index1];
    const Vertex& v2 = geomInst.vertexBuffer[tri.index2];

    float b0 = 1.0f - (bc.x + bc.y);
    Point2D texCoord = b0 * v0.texCoord + bc.x * v1.texCoord + bc.y * v2.texCoord;

    // Read transparency value
    float transparency = tex2DLod<float> (mat.transparency, texCoord.x, texCoord.y, 0.0f);

    // Get current visibility value
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::get (&visibility);

    if (transparency > 0.0f)
    {
        // Attenuate visibility by transparency
        visibility *= transparency;
        VisibilityRayPayloadSignature::set (&visibility);

        // Continue ray if still enough visibility
        if (visibility > 0.01f)
            return;
    }

    // Terminate ray if opaque or visibility too low
    visibility = 0.0f;
    VisibilityRayPayloadSignature::set (&visibility);
    optixTerminateRay();
}

#endif