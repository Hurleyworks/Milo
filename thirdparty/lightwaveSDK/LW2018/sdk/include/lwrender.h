/*
 * LWSDK Header File
 *
 * LWRENDER.H -- LightWave Rendering State
 *
 * This header contains the basic declarations need to define the
 * simplest LightWave plug-in server.
 *
 *Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of NewTek, Inc.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_RENDER_H
#define LWSDK_RENDER_H

#include <lwtypes.h>
#include <lwhandler.h>
#include <lwenvel.h>
#include <lwmeshes.h>
#include <lwviewportinfo.h>

typedef int     LWItemType;
#define LWI_OBJECT   0
#define LWI_LIGHT    1
#define LWI_CAMERA   2
#define LWI_BONE     3

typedef int      LWItemParam;
#define LWIP_POSITION    1
#define LWIP_RIGHT       2
#define LWIP_UP          3
#define LWIP_FORWARD     4
#define LWIP_ROTATION    5
#define LWIP_SCALING     6
#define LWIP_PIVOT       7
#define LWIP_W_POSITION  8
#define LWIP_W_RIGHT     9
#define LWIP_W_UP        10
#define LWIP_W_FORWARD   11
#define LWIP_PIVOT_ROT   12

typedef void* LWItemInstancerID;
typedef void* LWItemInstanceID;
typedef void *LWMemChunk;
typedef void* LWRenderData;

struct st_LWShaderAccess;
struct st_LWBxDFSample;

typedef double       LWRayCastFunc (const LWDVector position,
                     const LWDVector direction);

typedef double       LWRayTraceFunc (const LWDVector position,
                     const LWDVector direction,
                     LWDVector color);

typedef double       LWRayShadeFunc (const LWDVector position,
                     const LWDVector direction,
                     struct st_LWShaderAccess *shader_access);

#define RNMODE_SPHERE                  0       // Closest polygon within a sphere
#define RNMODE_HEMISPHERE              1       // Closest polygon within a hemisphere
#define RNMODE_RADIOSITY               2       // Radiosity sample test
#define RNMODE_INTERSECTION            3       // Intersection edge test

#define RNFLAG_FIRST_SOLUTION          (1<<8)  // Stop when any closer point is found
#define RNFLAG_NO_ORIGIN_OBJECT        (1<<9)  // Don't include the origin object
#define RNFLAG_RADIOSITY_TRANSPARENCY  (1<<10) // Include polygons transparent to radiosity

typedef int          LWRayNearestFunc (const LWDVector position,
                     const LWDVector direction,
                     unsigned int flags,
                     double *distance);

typedef int      LWIlluminateFunc (LWItemID light, const LWDVector position,
                       LWDVector direction, LWDVector color);

typedef int      LWIlluminateNormalFunc (LWItemID light, LWRenderData rd, const LWDVector position,
                       LWDVector direction, const LWDVector normal_vec, LWDVector color);

typedef int      LightSampleFunc( void *data, LWItemID light, const LWDVector dir, const double color[4] );
typedef double   LWIlluminateSampleFunc( LWItemID light, const LWDVector pos, LWDVector dir,
                                     LightSampleFunc *sampler, void *data );

typedef double   LWIlluminateSampleNormalFunc( LWItemID light, LWRenderData rd, const LWDVector pos,
                                     const LWDVector normal_vec, LightSampleFunc *sampler, void *data );

typedef void     LWBufferGetValFunc( LWRenderData rd, const char *type, int num, double *data );
typedef void     LWBufferSetValFunc( LWRenderData rd, const char *type, int num, double *data );

#define RTMODE_DISTANCE         0
#define RTMODE_REFLECTION       1
#define RTMODE_REFRACTION       2
#define RTMODE_DISSOLVE         3
#define RTMODE_SHADOW           4
#define RTMODE_OCCLUSION        5
#define RTMODE_MASK             0x000000FF

#define RTFLAG_MCONLY           0x00010000      // Force the ray to only use MC radiosity
#define RTFLAG_NOTLIGHT         0x00020000      // Prevent ray from hitting light primitives
#define RTFLAG_PREPROCESS       0x00040000      // Ray is part of the pre-processing pass.
#define RTFLAG_NEAREST          0x00080000      // Find the nearest point in a sphere or hemisphere (distance rays only)
#define RTFLAG_RN_SPHERE        0x00100000      // Not used currently - to be removed.
#define RTFLAG_RN_EDGE          0x00200000      // Not used currently - to be removed.
#define RTFLAG_FIRSTHIT         0x00400000      // Find the first opaque intersection point between tMin and tMax
#define RTFLAG_VOLUMETRIC       0x00800000      // Ray is used for volumetrics
#define RTFLAG_NOTOBJECT        0x01000000      // Prevent the ray from hitting the originating object
#define RTFLAG_RADIOSITY        0x02000000      // Ray is used for radiosity
#define RTFLAG_CAMERA           0x04000000      // The ray originated from the camera
#define RTFLAG_BACKSHADE        0x08000000      // Allow the ray to hit the back sides of polygons
#define RTFLAG_CAUSTICS         0x10000000      // Ray is used for radiosity (when caustics are being evaluated)
#define RTFLAG_OBJECT           0x20000000      // Force the ray to only hit the originating object
#define RTFLAG_SAMPLEDRAY       0x40000000      // A parent ray has already been sampled
#define RTFLAG_NOPOLYGON        0x80000000      // Allow the ray to hit the originating polygon
#define RTFLAG_MASK             0xFFFFFF00

typedef double       LWRayTraceModeFunc (const LWDVector position, const LWDVector direction,
                                         LWDVector color, const double eta, const int rtmode );

typedef void* LWRayData;

#define LWVOLUMESPOT_OPACITY (1 << 1)           // The volumetric is being sampled by an opacity/shadow ray
#define LWVOLUMESPOT_SAMPLED (1 << 2)           // Same as RTFLAG_SAMPLEDRAY

#define RTELEMENT_PRIMITIVE             0x00
#define RTELEMENT_PRIMITIVE_LIGHT       0x01
#define RTELEMENT_POLYGON               0x10
#define RTELEMENT_OUTLINE               0x20
#define RTELEMENT_EDGE                  0x30
#define RTELEMENT_EDGE_SILHOUETTE       0x31
#define RTELEMENT_EDGE_UNSHARED         0x32
#define RTELEMENT_EDGE_SHARP_CREASE     0x33
#define RTELEMENT_EDGE_SURFACE_BORDER   0x34
#define RTELEMENT_EDGE_OTHER            0x35
#define RTELEMENT_EDGE_PATCH_BORDER     0x36
#define RTELEMENT_EDGE_INTERSECTION     0x37
#define RTELEMENT_LINE                  0x40
#define RTELEMENT_PARTIGON              0x50
#define RTELEMENT_POINT                 0x60

typedef struct st_LWVolumeSpot
{
    const struct st_LWRay*  ray;
    LWDVector               P;
    LWDVector               dPdx;
    LWDVector               dPdy;
    unsigned int            flags;
} LWVolumeSpot;

typedef struct st_LWRenderGlobals
{
    double          rayCutoff;                  // Ray cutoff tolerance.

    unsigned int    frameWidth, frameHeight;    // Dimensions of the frame being rendered.

    unsigned int    renderOpts;                 // Render options. Such as; LWROPT_SHADOWTRACE, LWROPT_REFLECTTRACE, etc.

    unsigned int    maxRecursions;              // Ray recursion limit.
    unsigned int    maxTransparency;            // Transparency recursion limit.
    unsigned int    maxReflection;              // Reflection recursion limit.
    unsigned int    maxRefraction;              // Refraction recursion limit.
    unsigned int    maxDiffuseBounces;          // Diffuse GI bounce limit.

    unsigned int    reflectionSamples;          // Number of reflection samples.
    unsigned int    refractionSamples;          // Number of refraction samples.
    unsigned int    scatteringSamples;          // Number of scattering samples.
    unsigned int    lightSamples;               // Maximum number of light samples per evaluation.

    unsigned int    maxIlluminations;           // The maximum number of light samples.

    unsigned int    numThreads;                 // Number of render threads.
} LWRenderGlobals;

typedef void* LWPrimitiveEvaluatorID;
typedef void* LWSurfaceEvaluatorID;
typedef void* LWRenderStateID;

typedef struct st_LWRay
{
    const LWRenderGlobals*          renderGlobals;
    const struct st_LWRenderState*  renderState;
    LWRenderStateID                 renderStateID;  // Render state ID

    const struct st_LWRay           *parentRay;     // The parent ray. When a ray is released, all of its child rays are released.
    const struct st_LWShadingGeometry  *parentIntersection; // The parent ray's intersection point.

    LWBTDFVolume    volume;

    LWDVector       P;              // The start position of the ray.
    LWDVector       dPdx, dPdy;     // The start position differentials.

    LWDVector       I;              // The direction of the ray.
    LWDVector       dIdx, dIdy;     // The direction differentials.

    LWDVector       weight;         // The weight of the ray.

    LWTime          fracTime;       // The sub-frame motion blur time (0..1 between shutter open and shutter closed).

    double          tMin, tMax;     // The min and max distance of the ray.

    double          eta;            // Outgoing refraction index.

    unsigned int    flags;          // The ray flags (currently uses the RTFLAG_ defines above).
    unsigned int    shadingFlags;   // The shading flags. Defaults to LWSURFACE_ALL.
} LWRay;

typedef void* LWRenderBufferSetID;
typedef void* LWRenderBufferID;

#define LWRENDERBUFFER_NULL ((LWRenderBufferID)-1)

// The LWShadingFuncs struct.
typedef struct st_LWShadingFuncs
{
    LWInstance  inst;
    LWSurfaceID surfaceID;  // The surface being shaded
    LWMemChunk  memory;

    // Surface evaluation
    LWBSDF      bsdf;
    double      (*shade)(const LWRay*, struct st_LWShadingGeometry*, LWDVector result); // Shade the surface. The function should evaluate the BSDF.

    int         (*setupBSDF)(const LWRay*, struct st_LWShadingGeometry*); // Needs to be called before F or sampleF are called. The function should evaluate the BSDF.
    double      (*F)(const LWRay*, struct st_LWShadingGeometry*, const LWDVector wo, const LWDVector wi, unsigned int flags, LWDVector f);
    void        (*sampleF)(const LWRay*, struct st_LWShadingGeometry*, const LWDVector sample, struct st_LWBxDFSample*, unsigned int bxdfFlags);

    // Visible light evaluation
    void        (*Le)(const LWRay*, struct st_LWShadingGeometry*, LWDVector);
} LWShadingFuncs;

// The LWShading struct.
typedef struct st_LWShading
{
    LWDVector I; // World space incident ray direction (points out from the surface)
    LWDVector dIdx; // World space partial derivative of the incident ray (points out from the surface)
    LWDVector dIdy; // World space partial derivative of the incident ray (points out from the surface)
    LWDVector dNdx; // World space partial derivative of the normal along x (dNdu * dudx + dNdv[i] * dvdx)
    LWDVector dNdy; // World space partial derivative of the normal along y (dNdu * dudy + dNdv[i] * dvdy)
} LWShading;

// The LWShadingGeometry struct.
typedef struct st_LWShadingGeometry
{
    LWShadingFuncs shadingFuncs; // Shading functions set by the primitive

    void *ownerData; // Should only be set and used by the owner of the Node Editor.
    const LWRay* ray; // The ray which hit the surface (can be null for displacements, etc)
    LWPrimitiveEvaluatorID primEvalID; // Primitive plugin evaluator
    LWItemID itemID; // The item
    LWMeshID meshID; // The mesh
    LWItemInstanceID instanceID; // Instance ID for the old instancing system

    struct st_LWVertexIndexes *vertexIndexes;  // For legacy support of animated UV maps for LWTextureID evaluation.

    LWPntID vrts[3]; // 3 closest vertices for the polygon
    LWPolID polygon; // The polygon

    LWDMatrix4 toLocal; // Transformation matrix to local space
    LWDMatrix4 toWorld; // Transformation matrix to world space

    LWShading shading; // Shading variables

    LWDVector P; // World position
    LWDVector dPdx; // World position partial derivative along x
    LWDVector dPdy; // World position partial derivative along y
    LWDVector dPdu; // World position partial derivative along u
    LWDVector dPdv; // World position partial derivative along v
    LWDVector Np; // Perturbed normal
    LWDVector Ns; // Unperturbed normal
    LWDVector Ng; // Geometric normal
    LWDVector Nb; // Undeformed base geometric normal
    LWDVector dNdu; // Partial derivative of the normal along u
    LWDVector dNdv; // Partial derivative of the normal along v
    LWDVector bary; // Barycentric coordinates
    LWDVector oP; // Local space position
    LWDVector vertsWPos[3]; // World positions of the 3 closest vertices

    double u; // u value for the primitive
    double v; // v value for the primitive
    double dudx; // Partial derivative of u along x
    double dvdx; // Partial derivative of v along x
    double dudy; // Partial derivative of u along y
    double dvdy; // Partial derivative of v along y
    double dist; // The distance of the from the ray to the intersection
    double epsilon;

    unsigned int primID; // Primitive ID set by the primitive plugin
    unsigned int side; // The side of the surface/polygon the ray intersected
    unsigned int polNum; // The polygon number of the polygon
    unsigned int vrtsNum[3]; // The indices of the 3 closest vertices
    unsigned int flags; // LWRT_ flags set by the primitive
    unsigned int element; // Element set by the primitive

    LWID ownerID; // Should only be set and used by the owner of the Node Editor.
} LWShadingGeometry;

typedef struct st_LWRayTraceData
{
    // send
    LWDVector   P;          // The start position of the ray.
    LWDVector   dPdX;       // The start position differentials.
    LWDVector   dPdY;
    LWDVector   I;          // The direction of the ray.
    LWDVector   dIdX;       // The direction differentials.
    LWDVector   dIdY;

    double      tMin;       // The minimum distance of the ray (Not implemented)
    double      tMax;       // The maximum distance of the ray

    LWDVector   backdrop;   // Fill this if RTFLAG_BACKDROP is set.
    LWDVector   weight;     // The weight of the ray.
    double      eta;
    int         flags;

    // receive
    LWRayData   ray;    // This data needs to be filled by LWRayTraceData before calling LWRayTraceShade.
    double      len;
    LWDVector   result; // The result is filled with the backdrop color at LWRayTraceData if the ray doesn't hit anything.
} LWRayTraceData;

typedef void         LWRayTraceDataFunc(LWRenderData, LWRayTraceData*);
typedef void         LWRayTraceShadeFunc(LWRayTraceData*);
typedef void*        RandomFloatData;
typedef float        LWRandomFloatFunc(RandomFloatData);

typedef void*        LWSamplerState;
typedef float        LWGetSamplerRangeFunc(LWSamplerState, unsigned int*, unsigned int*);
typedef void         LWGetSample2DFunc(LWSamplerState, unsigned int, float*); // float* must point to 2 floats.
typedef float        LWGetSample1DFunc(LWSamplerState, unsigned int);

typedef void* LWLightEvaluatorID;

typedef struct st_LWRenderLight
{
    LWItemID            id;             // The item ID of the light.
    LWLightEvaluatorID  evaluator;      // The light evaluator
    LWRenderBufferSetID bufferSetID;    // The buffer set

    // Currently lights handle the color interpolation manually. @TODO modify the light interface to use this color instead!!!
    LWDVector           color;          // Color including intensity
    LWDVector           shadowColor;    // Shadow color
    double              intensity;      // Intensity
    double              volIntensity;   // Volumetric intensity

    unsigned int        lightFlags;    // Light Flags. Same as returned by LWLightInfo
    unsigned int        pluginFlags;   // Plugin Flags. Returned by the light plugin handler flags function
    unsigned int        shadowType;     // Shadow type. Same as returned by LWLightInfo
    unsigned int        numVolSamples;  // Number of volumetric samples
} LWRenderLight;

typedef struct st_LWRenderState
{
    struct st_LWRay*                    (*cameraRay)(LWRenderStateID , unsigned int thread);                                                                        // Get the camera ray for the selected thread
    struct st_LWRay*                    (*newRay)(LWRenderStateID);                                                                                                 // Allocate a new ray and make it a child of its parent ray
    void                                (*releaseRay)(LWRay*);                                                                                                      // Release the ray and all of its child rays

    unsigned int                        (*numLights)(LWRenderStateID);                                                                                              // Number of active lights
    const LWRenderLight*                (*renderLight)(LWRenderStateID, unsigned int lightIndex);                                                                   // Return a LWRenderLight
    struct st_LWLightEvaluationMemory*  (*lightMemory)(LWRenderStateID);                                                                                            // Light memory. Needs to be allocated per recursion.

    void*                               (*allocateMemory)(LWRenderStateID, size_t size);                                                                            // Allocate memory dynamically during rendering. Typically implemented as a memory pool.

    unsigned int                        (*numPrimitives)(LWRenderStateID);                                                                                          // Number of active primitives
    LWPrimitiveEvaluatorID              (*primitiveEvaluator)(LWRenderStateID, unsigned int primitiveIndex);                                                        // Return a primitive evaluator

    LWShadingGeometry*                  (*intersect)(const LWRay*);                                                                                                 // Intersection function
    int                                 (*shadow)(const LWRay*, const LWRenderLight*, LWDVector);                                                                   // Shadow function including volume transmittance
    int                                 (*opacity)(const LWRay*, LWDVector);                                                                                        // Opacity function (skips the shadow ray tests and shadow offset)
    double                              (*distance)(const LWRay*);                                                                                                  // Distance function
    double                              (*rayTrace)(const LWRay*, LWDVector color);                                                                                 // Ray trace function
    void                                (*backdrop)(const LWRay*, LWDVector color);                                                                                 // Backdrop function
    int                                 (*indirectBSDF)(const LWRay*, LWShadingGeometry*, LWBSDF, LWDVector);                                                       // Global illumination function for BSDFs
    int                                 (*indirectRay)(const LWRay*, LWDVector);                                                                                    // Global illumination function for a ray
    int                                 (*integrateRay)(const LWRay*, double tMin, double tMax, LWDVector result, LWDVector transmittance, LWItemID item);          // Volume integrator evaluation function.
    int                                 (*integrateRayTransmittance)(const LWRay*, double tMin, double tMax, LWDVector transmittance);                              // Volume integrator evaluation function. Transmittance only
    int                                 (*integrateBTDF)(const LWRay*, double tMin, double tMax, const LWBTDFVolume, LWDVector result, LWDVector transmittance);    // Volume integrator evaluation function for BTDFs
    int                                 (*integrateBTDFTransmittance)(const LWRay*, double tMin, double tMax, const LWBTDFVolume, LWDVector transmittance);         // Volume integrator evaluation function for BTDFs. Transmittance only

    unsigned int*                       (*totalRays)(LWRenderStateID);                                                                                              // Total number of rays
    unsigned int                        (*reflectionRays)(LWRenderStateID);                                                                                         // Total number of reflection rays
    unsigned int                        (*refractionRays)(LWRenderStateID);                                                                                         // Total number of refraction rays
    unsigned int                        (*radiosityBounces)(LWRenderStateID);                                                                                       // Total number of radiosity bounces
    unsigned int                        (*volumetricBounces)(LWRenderStateID);                                                                                      // Total number of volumetric bounces

    void                                (*pixelXY)(LWRenderStateID, unsigned int* x, unsigned int* y);                                                              // X and Y coordinate of the pixel being rendered.
    void                                (*subPixelXY)(LWRenderStateID, double* x, double* y);                                                                       // Sub-pixel X and Y coordinates being rendered.

    LWFrame                             (*frame)(LWRenderStateID);                                                                                                  // Current frame number being rendered.
    LWTime                              (*fracTime)(LWRenderStateID);                                                                                               // The sub-frame motion blur time (0..1 between shutter open and shutter closed).

    unsigned int                        (*surfaceEvaluationFlags)(LWRenderStateID);                                                                                 // Surface evaluation flags.
    unsigned int                        (*threadIndex)(LWRenderStateID);                                                                                            // The index number of the thread rendering.

    unsigned int*                       (*numSamples)(LWRenderStateID);                                                                                             // Number of samples being rendered.
    unsigned int*                       (*sampleIndex)(LWRenderStateID);                                                                                            // Current sample index being rendered.
    unsigned int*                       (*samplePass)(LWRenderStateID);                                                                                             // Current adaptive sampling pass being rendered.
    int                                 (*sceneBounds)(LWRenderStateID, LWDVector min, LWDVector max);                                                              // Scene bounds. Including volumetric and primitives with motion.

    LWRenderBufferSetID                 (*bufferSet)(LWRenderStateID);                                                                                              // The main render buffer set.
    LWRenderBufferID                    (*renderBuffer)(LWRenderStateID, LWRenderBufferSetID, unsigned int hash);                                                   // Lookup a render buffer from the render buffer set.
    void                                (*writeBuffer)(LWRenderStateID, LWRenderBufferID, double*, double depth);                                                   // Write value into the buffer.

    int                                 (*stopRendering)(LWRenderStateID);                                                                                          // Returns 1 when the rendering is aborted and should stop

    LWIlluminateFunc                    *legacy_illuminate;                                                                                                         // Render functions to support some legacy features
    LWIlluminateSampleFunc              *legacy_illuminateSample;
    LWRayTraceFunc                      *legacy_rayTrace;
    LWRayCastFunc                       *legacy_rayCast;
    LWRayShadeFunc                      *legacy_rayShade;
    LWRayTraceModeFunc                  *legacy_rayTraceMode;
    LWIlluminateNormalFunc              *legacy_illuminateNormal;
    LWIlluminateSampleNormalFunc        *legacy_illuminateSampleNormal;
    LWRayTraceDataFunc                  *legacy_rayTraceData;
    LWRayTraceShadeFunc                 *legacy_rayTraceShade;
    LWRenderData                        legacy_render;
    LWBufferGetValFunc                  *legacy_getVal;
    LWBufferSetValFunc                  *legacy_setVal;
    LWSamplerState                      legacy_samplerState;
    LWGetSamplerRangeFunc               *legacy_getSamplerRange;
    LWGetSample2DFunc                   *legacy_getSample2D;
    LWGetSample1DFunc                   *legacy_getSample1D;
} LWRenderState;

#define LWRT_SHADOW             (1<<0)  // This flag is on if the incoming ray is a shadow ray
#define LWRT_PREVIEW            (1<<1)  // This flag is on if the spot is being rendered in a preview context, like the Viper.
#define LWRT_POLYSIDE           (1<<2)  // This flag is on if the incoming ray hit the back side of the polygon
#define LWRT_SAMPLEDRAY         (1<<3)  // This flag is on if the incoming ray was cast for sampling soft reflections, etc.
#define LWRT_PREPROCESS         (1<<4)  // This flag is on if the incoming ray is a radiosity preprocess ray.
#define LWRT_EXITING            (1<<5)  // This flag is on if the ray is going to exit to an undefined surface - Ie. vacuum/air, which needs to be handled by the material/shader.
#define LWRT_DOUBLESIDED        (1<<6)  // This flag is on if the surface is double sided.
#define LWRT_FRAME_PREPROCESS   (1<<7)  // This flag is on if the incoming ray is a frame preprocess ray.
#define LWRT_RADIOSITY          (1<<8)  // This flag is on if the incoming ray is a radiosity ray.
#define LWRT_CAMERA             (1<<9)  // This flag is on if the incoming ray is a camera ray.

#define LWITEM_RADIOSITY        ((LWItemID) 0x21000000)
#define LWITEM_CAUSTICS         ((LWItemID) 0x22000000)
#define LWITEM_PREVIEW_OBJECT   ((LWItemID) 0x10007FFF)
#define LWITEM_PREVIEW_LIGHT    ((LWItemID) 0x21FFFFFF)

/*
 * Animation item handler extensions.
 */

typedef struct st_LWNameChangeData {
    const char  *oldName /* language encoded */;
    const char  *newName /* language encoded */;
    LWItemID    id;
} LWNameChangeData;

typedef struct st_LWItemFuncs {
    const LWItemID *    (*useItems) (LWInstance);
    void        (*changeID) (LWInstance, const LWItemID *);
    void        (*changeName) (LWInstance, const LWNameChangeData *);
} LWItemFuncs;

typedef struct st_LWItemHandler {
    LWInstanceFuncs  *inst;
    LWItemFuncs  *item;
} LWItemHandler;

#define LWITEM_ALL  ((LWItemID) ~0)


/*
 * Render handler extensions.
 */
typedef struct st_LWRenderFuncs {
    LWError     (*init)    (LWInstance, int);
    void        (*cleanup) (LWInstance);
    LWError     (*newTime) (LWInstance, LWFrame, LWTime);
} LWRenderFuncs;

#define LWINIT_PREVIEW   0
#define LWINIT_RENDER    1

typedef struct st_LWRenderHandler {
    LWInstanceFuncs  *inst;
    LWItemFuncs      *item;
    LWRenderFuncs    *rend;
} LWRenderHandler;


/*
 * Globals.
 */
#define LWLISTINFO_GLOBAL   "LW List Info"

#define SPLICE_HEAD     (LWItemID)0
#define SPLICE_TAIL     (LWItemID)0xffffffff

enum
{
    LI_Failed = -1,     // returned only by index()
    LI_Success,
    LI_InvalidItemType,
    LI_ItemTypeMismatch,
    LI_InvalidPair,
    LI_InvalidBoneParent,
    LI_OutOfBounds,
    LI_MixedTypes,
    LI_MixedBones,
    LI_InvalidInsertionPoint,
    LI_CyclicInsertionPoint
};

typedef struct st_LWListInfo {
    int         (*index)(LWItemID);
    int         (*reorder)(LWItemID*);
    int         (*swap)(LWItemID*);
    int         (*splice)(LWItemID,LWItemID*);
} LWListInfo;

#define LWITEMINFO_GLOBAL   "LW Item Info 8"

typedef struct st_LWItemInfo {
    LWItemID        (*first)        (LWItemType, LWItemID);
    LWItemID        (*next)         (LWItemID);
    LWItemID        (*firstChild)   (LWItemID parent);
    LWItemID        (*nextChild)    (LWItemID parent, LWItemID prevChild);
    LWItemID        (*parent)       (LWItemID);
    LWItemID        (*target)       (LWItemID);
    LWItemID        (*goal)         (LWItemID);
    LWItemType      (*type)         (LWItemID);
    const char * /* language encoded */ (*name)         (LWItemID);
    void            (*param)        (LWItemID, LWItemParam, LWTime, LWDVector);
    unsigned int    (*limits)       (LWItemID, LWItemParam, LWDVector min, LWDVector max);
    const char * /* language encoded */ (*getTag)       (LWItemID, int);
    void            (*setTag)       (LWItemID, int, const char *tag /* language encoded */);
    LWChanGroupID   (*chanGroup)    (LWItemID);
    const char * /* language encoded */ (*server)       (LWItemID, const char *class_name /* ascii encoded */, int);
    unsigned int    (*serverFlags)  (LWItemID, const char *server_name /* ascii encoded? */, int);
    void            (*controller)   (LWItemID, LWItemParam, int type[3]);
    unsigned int    (*flags)        (LWItemID);
    LWTime          (*lookAhead)    (LWItemID);
    double          (*goalStrength) (LWItemID, LWTime);
    void            (*stiffness)    (LWItemID, LWItemParam, LWDVector);
    unsigned int    (*axisLocks)    (LWItemID, LWItemParam);
    unsigned int    (*maxLookSteps) (LWItemID);
    double          (*reliableDistance)     (LWItemID);
    unsigned int    (*goalObjective)        (LWItemID);
    double          (*ikfkBlending)         (LWItemID, LWTime);
    unsigned int    (*ikInitialState)       (LWItemID);
    LWFrame         (*ikInitialStateFrame)  (LWItemID, LWTime);
    unsigned int    (*ikSoft)               (LWItemID, LWTime, unsigned int *distanceType, double *min, double *max);
    LWItemID        (*pole)                 (LWItemID);
    LWItemID        (*sameItem)             (LWItemID, LWItemParam);
    double          (*sameItemBlend)        (LWItemID, LWItemParam, LWTime);
    unsigned int    (*sameItemBlendMethod)  (LWItemID, LWItemParam);
    unsigned int    (*sameItemFlags)        (LWItemID, LWItemParam);
    unsigned int    (*selected)             (LWItemID);
    void            (*follow)               (LWItemID, LWItemParam, int follow[3], LWTextureID txtr[3]);
    void            (*controllerTrans)      (LWItemID, LWItemParam, double mul[3], double add[3]);
    LWItemID        (*splineItem)           (LWItemID);
    unsigned int    (*splineFit)            (LWItemID);
    double          (*splineT)              (LWItemID, LWTime);
} LWItemInfo;

#define LWVECF_0    (1<<0)
#define LWVECF_1    (1<<1)
#define LWVECF_2    (1<<2)

#define LWSRVF_DISABLED (1<<0)
#define LWSRVF_HIDDEN   (1<<1)

#define LWMOTCTL_KEYFRAMES          0
#define LWMOTCTL_TARGETING          1
#define LWMOTCTL_ALIGN_TO_VELOCITY  2
#define LWMOTCTL_IK                 3
#define LWMOTCTL_ALIGN_TO_PATH      4
#define LWMOTCTL_POLE               5
#define LWMOTCTL_SAMEITEM           6
#define LWMOTCTL_SPLINE             7

#define LWITEMF_ACTIVE          (1<<0)
#define LWITEMF_UNAFFECT_BY_IK  (1<<1)
#define LWITEMF_FULLTIME_IK     (1<<2)
#define LWITEMF_GOAL_ORIENT     (1<<3)
#define LWITEMF_REACH_GOAL      (1<<4)
#define LWITEMF_USE_IKCHAINVALS (1<<5)

#define LWMOTGOAL_POSITION  0
#define LWMOTGOAL_POINT     1
#define LWMOTGOAL_ONYZPLANE 2

#define LWMOTIK_INIT_FIRSTKEY       0
#define LWMOTIK_INIT_MOSTRECENTKEY  1
#define LWMOTIK_INIT_CURRENTCHAN    2
#define LWMOTIK_INIT_CUSTOMFRAME    3

#define LWMOTIK_SOFT_OFF            0
#define LWMOTIK_SOFT_EXPONENTIAL    1
#define LWMOTIK_SOFT_LINEAR         2
#define LWMOTIK_SOFT_CLAMP          3

#define LWMOTIK_SOFT_CHAINCHORD     0
#define LWMOTIK_SOFT_CHAINLENGTH    1
#define LWMOTIK_SOFT_CUSTOMDISTANCE 2

#define LWMOTCTL_SAMEITEM_INTERPOLATE 0
#define LWMOTCTL_SAMEITEM_COMPENSATE  1
#define LWMOTCTL_SAMEITEM_WORLD       (1<<7)

#define LWSPLINEFIT_LINEARDISTANCE  0
#define LWSPLINEFIT_ALONGSPLINE     1
#define LWSPLINEFIT_SPLINELENGTH    2


#define LWOBJECTINFO_GLOBAL "LW Object Info 12"

typedef struct st_LWObjectInfo {
    const char* /* language encoded */ (*filename)(LWItemID objectitem);
    int             (*numPoints)(LWItemID objectitem);
    int             (*numPolygons)(LWItemID objectitem);
    unsigned int    (*shadowOpts)(LWItemID objectitem);
    double          (*dissolve)(LWItemID objectitem, LWTime in_time);
    LWMeshInfoID    (*meshInfo)(LWItemID objectitem, int frozen_flag);
    unsigned int    (*flags)(LWItemID objectitem);
    double          (*fog)(LWItemID objectitem, LWTime in_time);
    LWTextureID     (*dispMap)(LWItemID objectitem);
    LWTextureID     (*clipMap)(LWItemID objectitem);
    void            (*patchLevel)(LWItemID objectitem, int *for_display, int *for_render);
    void            (*metaballRes)(LWItemID objectitem, double *for_display, double *for_render);
    LWItemID        (*boneSource)(LWItemID objectitem);
    LWItemID        (*morphTarget)(LWItemID objectitem);
    double          (*morphAmount)(LWItemID objectitem, LWTime in_time);
    unsigned int    (*edgeOpts)(LWItemID objectitem);
    void            (*edgeColor)(LWItemID objectitem, LWTime in_time, LWDVector color);
    int             (*subdivOrder)(LWItemID objectitem);
    double          (*polygonSize)(LWItemID objectitem, LWTime in_time); /* deprecated */
    int             (*excluded)(LWItemID objectitem, LWItemID lightitem);
    void            (*matteColor)(LWItemID objectitem, LWTime in_time, LWDVector color);
    double          (*thickness)(LWItemID objectitem, LWTime in_time, int type);
    double          (*edgeZScale)(LWItemID objectitem, LWTime in_time);
    double          (*shrinkEdgesNominalDistance)(LWItemID objectitem, LWTime in_time); /* added for version 5: obtains nominal distance value used when shrink edges is enabled. */
    double          (*maxDissolveDistance)(LWItemID objectitem); /* added for version 6 */
    double          (*bumpDistance)(LWItemID objectitem); /* added for version 6 */
    unsigned int    (*getGroupIndex)(LWItemID objecitem); /* added for version 7 */
    int             (*dispMapOrder)(LWItemID objectitem);
    int             (*bumpOrder)(LWItemID objectitem);
    NodeEditorID    (*getNodeEditor)(LWItemID objectitem);
    int             (*nodeOrder)(LWItemID objectitem);
    void            (*bounds)(LWItemID objectitem, LWDVector min, LWDVector max); /* added for version 8 */
    double          (*shadowoffsetdistance)(LWItemID);
    LWItemInstancerID (*instancer)(LWItemID);               /* added for version 9 */
    NodeEditorID    (*getEdgeNodeEditor)(LWItemID objectitem);
    int             (*layerLoaded)(LWItemID objectitem);    /* added for version 10 */
    float           (*getAlphaValue)(LWItemID objectitem);  /* added for version 11 */
    int             (*giMode)(LWItemID objectitem);         /* added for version 2018. */
    int             (*bruteForceRays)(LWItemID objectitem);
    int             (*primaryRays)(LWItemID objectitem);
    int             (*secondaryRays)(LWItemID objectitem);
    int             (*missingSampleRays)(LWItemID objectitem);
    double          (*angularTolerance)(LWItemID objectitem);
    double          (*minimumPixelSpacing)(LWItemID objectitem);
    double          (*maximumPixelSpacing)(LWItemID objectitem);
} LWObjectInfo;

#define LWOSHAD_SELF        (1<<0)
#define LWOSHAD_CAST        (1<<1)
#define LWOSHAD_RECEIVE     (1<<2)

#define LWOBJF_UNSEEN_BY_CAMERA (1<<0)
#define LWOBJF_UNSEEN_BY_RAYS   (1<<1)
#define LWOBJF_UNAFFECT_BY_FOG  (1<<2)
#define LWOBJF_MORPH_MTSE   (1<<3)
#define LWOBJF_MORPH_SURFACES   (1<<4)
#define LWOBJF_MATTE        (1<<5)
#define LWOBJF_ALPHA_CONSTANT_COLOR  (1<<7) /* This is set when object alpha is set to 'Constant Color' */
#define LWOBJF_DISTANCEDISSOLVE (1<<8) /* Added in version 6 */
#define LWOBJF_BUMPENABLE (1<<9) /* Added in version 6 */
#define LWOBJF_FASTERBONES (1<<10) /* Added in version 6 */
#define LWOBJF_USEMORPHEDPOSITIONS (1<<11) /* Added in version 6 */
#define LWOBJF_CONTAINS1POINTPOLYGONS (1<<12) /* Added in version 6 */
#define LWOBJF_CONTAINSPATCHES (1<<13) /* Added in version 6 */
#define LWOBJF_CONTAINSMETABALLS (1<<14) /* Added in version 6 */
#define LWOBJF_CONTAINSPARTIGONS (1<<15) /* Added in version 6 */
#define LWOBJF_CONTAINSCUSTOMPOLYGONS (1<<16) /* Added in version 6 */
#define LWOBJF_NODEDISPENABLED (1<<17) /* Added in version 7 */
#define LWOBJF_NODEFORMERSAPPLIED (1<<18) /* Added in version 7 */
#define LWOBJF_UNSEEN_BY_RADIOSITY (1<<19)
#define LWOBJF_NODEEDGEENABLED (1<<20) /* Added in version 9 */

#define LWEDGEF_SILHOUETTE   (1<<0)
#define LWEDGEF_UNSHARED     (1<<1)
#define LWEDGEF_CREASE       (1<<2)
#define LWEDGEF_SURFACE      (1<<3)
#define LWEDGEF_OTHER        (1<<4)
#define LWEDGEF_PATCH        (1<<5)
#define LWEDGEF_INTERSECTION (1<<6)
#define LWEDGEF_SHRINK_DIST  (1<<8)

#define LWTHICK_SILHOUETTE      0
#define LWTHICK_UNSHARED        1
#define LWTHICK_CREASE          2
#define LWTHICK_SURFACE         3
#define LWTHICK_OTHER           4
#define LWTHICK_LINE            5
#define LWTHICK_PARTICLE_HEAD   6
#define LWTHICK_PARTICLE_TAIL   7
#define LWTHICK_PATCH           8
#define LWTHICK_INTERSECTION    9

#define LWBONEINFO_GLOBAL   "LW Bone Info 6"

typedef struct st_LWBoneInfo {
    unsigned int    (*flags)      (LWItemID);
    void            (*restParam)  (LWItemID, LWItemParam, LWDVector);
    double          (*restLength) (LWItemID);
    void            (*limits)     (LWItemID, double *inner, double *outer);
    const char * /* language encoded */ (*weightMap)  (LWItemID);
    double          (*strength)   (LWItemID);
    int             (*falloff)    (LWItemID);
    void            (*jointComp)  (LWItemID, double *self, double *parent);
    void            (*muscleFlex) (LWItemID, double *self, double *parent);
    int             (*type)       (LWItemID);
    double          (*twist)      (LWItemID);
    int             (*transform)  (LWItemID, int transform, int relativeto, LWFMatrix3 m, LWFVector pos, LWFVector end);
    void            (*muscleBulge)(LWItemID, double *self, double *parent);
    void            (*muscleBulgeMap)   (LWItemID, LWTextureID *self, LWTextureID *parent);
    void            (*displacementMap)  (LWItemID, LWTextureID *self, LWTextureID *parent);
    int             (*deformation)(LWItemID, int relativeto,
                                   LWFMatrix3 restm, LWFVector restp, LWFMatrix3 finalm, LWFVector finalp,
                                   LWFMatrix3 defm, LWFVector defp,
                                   LWFMatrix3 rotc, LWFMatrix3 rots, LWFMatrix3 rotp, LWFVector rota);
} LWBoneInfo;

#define LWBONEF_ACTIVE              (1<<0)
#define LWBONEF_LIMITED_RANGE       (1<<1)
#define LWBONEF_SCALE_STRENGTH      (1<<2)
#define LWBONEF_WEIGHT_MAP_ONLY     (1<<3)
#define LWBONEF_WEIGHT_NORM         (1<<4)
#define LWBONEF_JOINT_COMP          (1<<5)
#define LWBONEF_JOINT_COMP_PAR      (1<<6)
#define LWBONEF_MUSCLE_FLEX         (1<<7)
#define LWBONEF_MUSCLE_FLEX_PAR     (1<<8)
#define LWBONEF_TWIST               (1<<9)
#define LWBONEF_MUSCLE_BULGE        (1<<10)
#define LWBONEF_MUSCLE_BULGE_PAR    (1<<11)

#define LWBONETYPE_ZAXIS        0
#define LWBONETYPE_JOINT        1

#define LWBONETRANS_REST        0
#define LWBONETRANS_FINAL       1

#define LWBONETRANS_OBJECT      0
#define LWBONETRANS_PARENT      1
#define LWBONETRANS_WORLD       2


#define LWLIGHTINFO_GLOBAL  "LW Light Info 6"

typedef struct st_LWLightInfo {
    void            (*ambient)      (LWTime, LWDVector color);
    int             (*type)         (LWItemID);
    void            (*color)        (LWItemID, LWTime, LWDVector color);
    int             (*shadowType)   (LWItemID);
    void            (*coneAngles)   (LWItemID, LWTime, double *radius, double *edge);
    unsigned int    (*flags)    (LWItemID);
    double          (*range)        (LWItemID, LWTime);
    int             (*falloff)      (LWItemID);
    LWImageID       (*projImage)    (LWItemID);
    int             (*shadMapSize)  (LWItemID);
    double          (*shadMapAngle) (LWItemID, LWTime);
    double          (*shadMapFuzz)  (LWItemID, LWTime);
    int             (*quality)      (LWItemID, LWTime);
    void            (*rawColor)     (LWItemID, LWTime, LWDVector color);
    double          (*intensity)    (LWItemID, LWTime);
    void            (*shadowColor)  (LWItemID, LWTime, LWDVector color);
    double          (*ambientIntensity)  (LWTime);
    void            (*ambientRaw)   (LWTime, LWDVector color);
    unsigned int    (*numVolumetricSamples)(LWItemID, LWTime);
    double          (*volumetricIntensity)(LWItemID, LWTime);
    NodeEditorID    (*nodeEditor)(LWItemID);
    int             (*nodesEnabled) (LWItemID);
    const char*     (*bufferGroup)  (LWItemID, const char* group);
} LWLightInfo;

#define LWLIGHT_DISTANT     0
#define LWLIGHT_POINT       1
#define LWLIGHT_SPOT        2
#define LWLIGHT_LINEAR      3
#define LWLIGHT_AREA        4
#define LWLIGHT_PHOTOMETRIC 5
#define LWLIGHT_CUSTOM      100

#define LWLSHAD_OFF      0
#define LWLSHAD_RAYTRACE 1
#define LWLSHAD_MAP      2 // deprecated

#define LWLFL_LIMITED_RANGE     (1<<0) // deprecated
#define LWLFL_NO_DIFFUSE        (1<<1)
#define LWLFL_NO_SPECULAR       (1<<2)
#define LWLFL_NO_CAUSTICS       (1<<3)
#define LWLFL_LENS_FLARE        (1<<4)
#define LWLFL_VOLUMETRIC        (1<<5)
#define LWLFL_NO_OPENGL         (1<<6)
#define LWLFL_FIT_CONE          (1<<7)
#define LWLFL_CACHE_SHAD_MAP    (1<<8) // deprecated
#define LWLFL_VISIBLE_TO_CAMERA (1<<9)
#define LWLFL_CAST_SHADOWS      (1<<10)

#define LWLFALL_OFF                 0
#define LWLFALL_ON                  1


#define LWCAMERAINFO_GLOBAL "LW Camera Info 10"

typedef struct st_LWCameraRay
{
    LWDVector rayPos;
    LWDVector rayDir;
    LWDVector filmNorm;
    LWDVector filter[3];
    double dFPX, dFPY;
    LWDVector rayPosDFPX;
    LWDVector rayPosDFPY;
    LWDVector rayDirDFPX;
    LWDVector rayDirDFPY;
    double convergenceDist;
} LWCameraRay;

typedef struct st_LWFrameInfo
{
    LWFrame frame;
    LWTime start;
    LWTime duration;
    double secondsPerFrame;
    unsigned int motionSteps;
} LWFrameInfo;

typedef struct st_LWComponentTimingInfo
{
    unsigned int count;
    double offset;
    double duration;
    double stride;
} LWComponentTimingInfo;

typedef struct st_LWFrameTimingInfo
{
    LWComponentTimingInfo frame;
    LWComponentTimingInfo fields;
    LWComponentTimingInfo steps;
    LWComponentTimingInfo scanlines;
    LWComponentTimingInfo pixels;
} LWFrameTimingInfo;

/* Stereoscopic eyes. */

typedef enum
{
    LWCAMEYE_CENTER = 0,/*!< Stereoscopic center eye. */
    LWCAMEYE_LEFT,      /*!< Stereoscopic left eye.   */
    LWCAMEYE_RIGHT,     /*!< Stereoscopic right eye.  */
    LWCAMEYE_sizeof
} LWCameraEye;

typedef enum
{
    LWCAMOVERSCAN_OFF = 0,
    LWCAMOVERSCAN_PIXELS,
    LWCAMOVERSCAN_FRACTION,
    LWCAMOVERSCAN_sizeof
} LWCameraOverscan;

typedef struct st_LWCameraInfo {
    double      (*zoomFactor)    (LWItemID, LWTime);
    double      (*focalLength)   (LWItemID, LWTime);
    double      (*focalDistance) (LWItemID, LWTime);
    double      (*fStop)         (LWItemID, LWTime);
    double      (*blurLength)    (LWItemID, LWTime);
    void        (*fovAngles)     (LWItemID, LWTime, double *horizontal, double *vertical);
    unsigned int    (*flags)     (LWItemID);
    void        (*resolution)    (LWItemID, int *width, int *height);
    double      (*pixelAspect)   (LWItemID, LWTime);
    double      (*separation)    (LWItemID, LWTime);
    void        (*regionLimits)  (LWItemID, int *out_x0, int *out_y0, int *out_x1, int *out_y1);
    void        (*maskLimits)    (LWItemID, int *out_x0, int *out_y0, int *out_x1, int *out_y1); // defunct
    void        (*maskColor)     (LWItemID, LWDVector color); // defunct
    unsigned int (*motionBlur)   (LWItemID); /* added for version 3: retrieve motion blur setting for a camera item, 1=normal, 2=dither */
    unsigned int (*fieldRendering)(LWItemID); /* added for version 3: retrieve field rendering state. */
    int         (*irisPos)        (LWItemID, LWTime, int pass, float *ix, float *iy);

    int         (*usingGlobalResolution)(LWItemID); // defunct
    int         (*usingGlobalBlur)(LWItemID); // defunct
    int         (*usingGlobalMask)(LWItemID); // defunct

    unsigned int (*motionBlurPasses)  (LWItemID, LWTime);
    double       (*shutterEfficiency) (LWItemID, LWTime);
    unsigned int (*noiseSampler)      (LWItemID);
    void         (*filmSize)          (LWItemID, double *width, double *height);
    unsigned int (*frameTiming)       (LWItemID, LWFrame, LWFrameTimingInfo *timingInfo);
    unsigned int (*antiAliasing)      (LWItemID, LWTime);
    double       (*overSampling)      (LWItemID, LWTime);
    void         (*diaphragm)         (LWItemID, LWTime, int *sides, double *rotation);
    double       (*convergencePoint)  (LWItemID, LWTime);
    int          (*usingConvergencePoint)(LWItemID);
    double       (*convergenceToeIn)  (LWItemID, LWTime);

    void         (*eyeTransform)      (LWItemID, LWTime, LWCameraEye eye, float mot[9], LWFVector convergence_point, float *convergence_distance, LWFMatrix3 delta_rotm, float delta_shear[2], LWFVector delta_pos);
    double       (*rollingShutter)    (LWItemID, LWTime); /* only for photoreal blur */
    double       (*shutterOpen)       (LWItemID, LWTime); /* only when a blur is enabled */
    unsigned int (*minSamples)        (LWItemID, LWTime); /* only valid for non-classic cameras */
    unsigned int (*maxSamples)        (LWItemID, LWTime); /* only valid for non-classic cameras */

    LWCameraEye  (*trackedEye)        (LWItemID);

    LWCameraOverscan    (*overscanMode)         (LWItemID);
    LWCameraOverscan    (*overscanSize)         (LWItemID, double *width, double *height);
    double              (*resolutionmultiplier) (LWItemID);
    int                 (*overscanPixels)       (LWItemID, double *width, double *height);
} LWCameraInfo;


#define LWCAMF_STEREO           (1<<0)  /* Indicates stereo rendering enabled */
#define LWCAMF_LIMITED_REGION   (1<<1)  /* This indicates that some form of limited region is enabled */
#define LWCAMF_MASK             (1<<2)  // defunct
#define LWCAMF_DOF              (1<<3)  /* Indicates depth of field enabled */
#define LWCAMF_PARTICLE_BLUR    (1<<4)
#define LWCAMF_LR_NOBORDERS     (1<<5)  /* This indicates that limited region is being used without borders when limited region is enabled */
#define LWCAMF_FIELD            (1<<6)  /* In Field Rendering, This Indicates Which Field Is Being Processed. */
#define LWCAMF_USECAMTYPE       (1<<7)  /* Indicates that the camera uses a non-classical camera type. (rev.4+) */
#define LWCAMF_SUPPORTS_DOF     (1<<8)  /* Indicates that the camera supports Depth Of Field rendering */
#define LWCAMF_EYE              (1<<9)  /* In Stereo Rendering, This Indicates Which Eye Is Being Processed. Set for left eye, cleared for right eye */
#define LWCAMF_SUPPORTS_STEREO  (1<<10) /* Indicates that the camera supports stereo rendering */
#define LWCAMF_SUPPORTS_PROJECT (1<<11) /* Indicates that the camera supports projecting world position to film position */

#define LWCAMMB_OFF 0
#define LWCAMMB_NORMAL 1
#define LWCAMMB_DITHERED 2
#define LWCAMMB_PHOTOREAL 3

#define LWCAMFIELDR_OFF 0
#define LWCAMFIELDR_EVENFIRST 1
#define LWCAMFIELDR_ODDFIRST 2


#define LWCAMERAEVALUATIONFUNCS_GLOBAL "LW Camera Evaluation Funcs 4"

typedef void* LWCameraEvaluatorID;

typedef struct st_LWCameraEvaluationFuncs
{
    LWCameraEvaluatorID (*create) (LWItemID camera);
    void    (*destroy)          (LWCameraEvaluatorID);
    LWError (*init)             (LWCameraEvaluatorID, int mode);
    void    (*cleanUp)          (LWCameraEvaluatorID);
    LWError (*newTime)          (LWCameraEvaluatorID, LWFrame frame, LWTime in_time);
    int     (*preview)          (LWCameraEvaluatorID, double lpx, double lpy, LWCameraEye eye, LWDMatrix4 projection);
    LWError (*newFrame)         (LWCameraEvaluatorID);
    int     (*evaluate)         (LWCameraEvaluatorID, double fpx, double fpy, double lpx, double lpy, double fractime, LWCameraEye eye, LWCameraRay* camray);
    int     (*project)          (LWCameraEvaluatorID, double lpx, double lpy, LWDVector worldpos, double fractime, LWCameraEye eye, LWDVector fp);
    int     (*sampledProject)   (LWCameraEvaluatorID, double lpx, double lpy, LWDVector worldpos, double fractime, LWCameraEye eye, double* fpx, double* fpy,
                                                      unsigned int first_sample_res[2], unsigned int sub_sample_res[2],
                                                      unsigned int iteration_limit, double filmpos_limit[2]);
} LWCameraEvaluationFuncs;


enum    // for use with LWSceneInfo->loadInProgress
{
    SI_NoLoad = 0,
    SI_LoadScene,
    SI_LoadFromScene
};

enum {
    LWRT_BackDrop = 0,  /* Throws rays in a random distribution, if a miss, used backdrop color. */
    LWRT_MonteCarlo,    /* Throws rays in a random distribution. */
    LWRT_FinalGather,   /* Final Gather implementation. */
    LWRT_Sizeof
};

typedef enum en_lwrenderingmode {
  lwrm_None = 0,        /*!< not currently rendering. */
  lwrm_SceneBake,       /*!< a scene frame baking.    */
  lwrm_Scene,           /*!< a scene.                 */
  lwrm_FrameBake,       /*!< a single frame baking.   */
  lwrm_Frame,           /*!< a single frame.          */
  lwrm_Inspire,         /*!< a scene but with dimensions greater than 800 x 600 for Inspire. */
  lwrm_SelectedObject,  /*!< selected objects only.   */
  lwrm_sizeof
} LWRENDERINGMODE;

typedef enum
{
    LWANIMPASS_MAIN = 0,
    LWANIMPASS_PRELIMINARY,
    LWANIMPASS_BLUR,
    LWANIMPASS_RESTORE,
    LWANIMPASS_DYNAMICS
} LWAnimPass;

enum
{
    LWRANGETYPE_SINGLE = 0,
    LWRANGETYPE_ARBITRARY,
    LWRANGETYPE_KEYFRAME
};

#define LWSCENEINFO_GLOBAL "LW Scene Info 11"

typedef struct st_LWSceneInfo {
    const char *name /* language encoded */;
    const char *filename /* language encoded */;
    int numPoints;
    int numPolygons;
    int renderType;
    int renderOpts;
    LWFrame frameStart;
    LWFrame frameEnd;
    LWFrame frameStep;
    double framesPerSecond;
    int frameWidth;
    int frameHeight;
    double pixelAspect;
    int minSamplesPerPixel;
    int maxSamplesPerPixel;
    int limitedRegion[4];   /* x0, y0, x1, y1 */
    int recursionDepth;
    LWItemID (*renderCamera) (LWTime);
    int numThreads;
    const char *animFilename /* language encoded */;
    const char *RGBPrefix /* language encoded */;
    const char *alphaPrefix /* language encoded */;
    int antialiasing; /* added for version 4: current antialiasing setting 0=off, 1-lowest, increasing for higher values (currently up to 4=extreme)*/
        /* added 12/29/2004: 100 to 114; use PLD (with level 1 being 100 and level 15 being 114) */
    int enhancedAA; /* added for version 4: 1=enabled, 0=disabled. Only valid when antialiasing is 1 thru 4*/
    int adaptiveSampling; /* added for version 4: 1=enabled, 0=disabled */
    float adaptiveThreshold; /* added for version 4: only valid when adaptiveSampling is enabled */
    int filter; /* added for version 4: bit 0: 1=soft filter, 0=no filter */
        /* added 12/29/2004: bits 1thru5 indicate reconstruction filter.
            1-3=box(std,sharp,soft); 4-6=Gaussian(std,sharp,soft); 7-9=Mitchell(std,sharp,soft); 10-12=Lanczos(std,sharp,soft) */
    int dblSidedAreaLights;
    int loadInProgress;         /* version 5: 0=no scene loading; 1=scene load is in progress; 2=load-from-scene in progress */
    int         radiosityType;
    int         radiosityIndirectBounceCount;
    int         radiosityRaysPerEvaluation1;
    int         radiosityRaysPerEvaluation2;
    double      radiosityIntensity;
    double      radiosityAngularTolerance;
    double      radiosityMinimumPixelSpacing;
    double      radiosityMaximumPixelSpacing;
    double      radiosityMultiplier;
    const char *radiosityFileName /* language encoded */;
    /* Added for LW9.6 */
    double      causticsIntensity;
    int         causticsAccuracy;
    int         causticsSoftness;
    int         radiosityFlags;
    double      dynamicsRangeMin;
    double      dynamicsRangeMax;
    double      saturation;

    int         pixelFilterForceMT;
    int         has_multithreaded_filters;      /* All active pixel filters are capable of multi-threading. */
    int         useZminmax;
    double      zmin;
    double      zmax;
    int         rendermode;                     /* LWRENDERINGMODE enum. */
    int         calculateallnormals;            /* Forces the calculation of normals, in layout or while rendering. */
    /* Added for LW9.7 */
    int         enableLenseFlares;
    double      lenseFlareIntensity;
    /* Added for LW10.1 */
    int        *buffersInUse;                   /* Pointer to the flags array of the buffers currently in use. */
    int         animationPass;                  /* Current phase of animation calculation (LWANIMPASS_). */

    /* Added for LW11 */
    float           raycutoff;
    unsigned int    shadingsamples;             // Deprecated
    unsigned int    lightsamples;
    float           lightthreshold;

    /* Added for LW2015/v10 */
    int             rangeType;
    const char      *rangeFrames;
    LWItemID        rangeObject;
    int             rangeChannel;

    /* Added for LW2018 */
    int             transparencyDepth;
    int             reflectionDepth;
    int             refractionDepth;
    unsigned int    reflectionsamples;
    unsigned int    refractionsamples;
    unsigned int    scatteringsamples;
    float           mipmapMultiplier;

} LWSceneInfo;

/* SceneInfo.renderType */
#define LWRTYPE_WIRE         0
#define LWRTYPE_QUICK        1
#define LWRTYPE_REALISTIC    2

/* SceneInfo.renderOpts */
#define LWROPT_SHADOWTRACE (1<<0)
#define LWROPT_REFLECTTRACE (1<<1)
#define LWROPT_REFRACTTRACE (1<<2)
#define LWROPT_FIELDS (1<<3)
#define LWROPT_EVENFIELDS (1<<4)
#define LWROPT_MOTIONBLUR (1<<5)
#define LWROPT_DEPTHOFFIELD (1<<6)
#define LWROPT_LIMITEDREGION (1<<7)
#define LWROPT_PARTICLEBLUR (1<<8)
#define LWROPT_ENHANCEDAA (1<<9)
#define LWROPT_SAVEANIM (1<<10)
#define LWROPT_SAVERGB (1<<11)
#define LWROPT_SAVEALPHA (1<<12)
#define LWROPT_ZBUFFERAA (1<<13) /* added 12/29/2004 */
#define LWROPT_RTTRANSPARENCIES (1<<14) /* added 11/01/2006 */
#define LWROPT_RADIOSITY (1<<15)
#define LWROPT_CAUSTICS (1<<16)
#define LWROPT_OCCLUSION (1<<17)
#define LWROPT_RENDERLINES (1<<18)
#define LWROPT_INTERPOLATED (1<<19)
#define LWROPT_BLURBACKGROUND (1<<20)
#define LWROPT_USETRANSPARENCY (1<<21)
#define LWROPT_VOLUMETRICRADIOSITY  (1<<22)
#define LWROPT_USEAMBIENT           (1<<23)
#define LWROPT_DIRECTIONALRAYS      (1<<24)
#define LWROPT_LIMITDYNAMICRANGE    (1<<25)
#define LWROPT_CACHERADIOSITY       (1<<26)
#define LWROPT_USEGRADIENTS         (1<<27)
#define LWROPT_USEBEHINDTEST        (1<<28)
#define LWROPT_CAUSTICSCACHE        (1<<29)
#define LWROPT_RENDERINSTANCES      (1<<30)
#define LWROPT_UNPREMULTIPLIEDALPHA (1<<31)

/* SceneInfo.radiosityFlags */
#define LWRDFLG_SHOW_NODES                        (1 << 0)
#define LWRDFLG_SHOW_CELLS                        (1 << 1)
#define LWRDFLG_SHOW_COLOR_CELLS                  (1 << 2)
#define LWRDFLG_SHOW_SAMPLES                      (1 << 3)
#define LWRDFLG_SHOW_MISSING_PREPROCESS_SAMPLES   (1 << 4)
#define LWRDFLG_SHOW_MISSING_RENDER_SAMPLES       (1 << 5)
#define LWRDFLG_SHOW_SECOND_BOUNCE                (1 << 6)
#define LWRDFLG_SHOW_BEHIND                       (1 << 7)
#define LWRDFLG_USE_BUMPS                         (1 << 31)

#define LWTIMEINFO_GLOBAL   "LW Time Info"

typedef struct st_LWTimeInfo {
    LWTime       time;
    LWFrame      frame;
} LWTimeInfo;


#define LWCOMPINFO_GLOBAL   "LW Compositing Info"

typedef struct st_LWCompInfo {
    LWImageID    bg;
    LWImageID    fg;
    LWImageID    fgAlpha;
} LWCompInfo;


#define LWBACKDROPINFO_GLOBAL   "LW Backdrop Info 3"

typedef struct st_LWBackdropInfo {
    void        (*backdrop) (LWTime, const LWDVector I, const LWDVector dIdx, const LWDVector dIdy, LWDVector color);
    int         type;
    void        (*color)    (LWTime, LWDVector zenith, LWDVector sky, LWDVector ground, LWDVector nadir);
    void        (*squeeze)  (LWTime, double *sky, double *ground);
} LWBackdropInfo;

#define LWBACK_SOLID        0
#define LWBACK_GRADIENT     1


#define LWFOGINFO_GLOBAL    "LW Fog Info"

typedef struct st_LWFogInfo {
    int       type;
    unsigned int      flags;
    double      (*minDist) (LWTime);
    double      (*maxDist) (LWTime);
    double      (*minAmt)  (LWTime);
    double      (*maxAmt)  (LWTime);
    void        (*color)   (LWTime, double col[3]);
} LWFogInfo;

#define LWFOG_NONE          0
#define LWFOG_LINEAR        1
#define LWFOG_NONLINEAR1    2
#define LWFOG_NONLINEAR2    3
#define LWFOG_REALISTIC     4

#define LWFOGF_BACKGROUND   (1<<0)


#define LWINTERFACEINFO_GLOBAL  "LW Interface Info 7"

typedef struct st_LWInterfaceInfo {
    LWTime            curTime;
    const LWItemID   *selItems;
    unsigned int    (*itemFlags)            (LWItemID item);
    LWFrame           previewStart;
    LWFrame           previewEnd;
    LWFrame           previewStep;
    int               dynaUpdate;
    void            (*schemaPos)            (LWItemID item, double *x, double *y);
    int             (*itemVis)              (LWItemID item);
    unsigned int      displayFlags;
    unsigned int      generalFlags;
    int               boxThreshold;
    int             (*itemColor)            (LWItemID item);
    void            (*setItemColorIndex)    (LWItemID item, int color);
    void            (*setItemColorCustom)   (LWItemID item, double *color);
    int               alertLevel;
    int               autoKeyCreate;
    void            (*defaultItemTypeColor) ( LWItemType itemtype, float *color, int set);
    void            (*itemColorRgba)        (LWItemID item, unsigned int state, float rgba[4]);
    float           (*itemIconScale)        (LWItemID item);
} LWInterfaceInfo;

#define LWITEMF_SELECTED    (1<<0)
#define LWITEMF_SHOWCHILDREN    (1<<1)
#define LWITEMF_SHOWCHANNELS    (1<<2)
#define LWITEMF_LOCKED      (1<<3)

#define LWDYNUP_OFF         0
#define LWDYNUP_DELAYED     1
#define LWDYNUP_INTERACTIVE 2

#define LWIVIS_HIDDEN       0
#define LWIVIS_VISIBLE      1

#define LWAKC_OFF           0
#define LWAKC_MODIFIED      1
#define LWAKC_ALL           2
#define LWAKC_EXISTING      3

#define LWDISPF_MOTIONPATHS (1<<0)
#define LWDISPF_HANDLES     (1<<1)
#define LWDISPF_IKCHAINS    (1<<2)
#define LWDISPF_CAGES       (1<<3)
#define LWDISPF_SAFEAREAS   (1<<4)
#define LWDISPF_FIELDCHART  (1<<5)

#define LWGENF_HIDETOOLBAR          (1<<0)
#define LWGENF_RIGHTTOOLBAR         (1<<1)
#define LWGENF_PARENTINPLACE        (1<<2)
#define LWGENF_FRACTIONALFRAME      (1<<3)
#define LWGENF_KEYSINSLIDER         (1<<4)
#define LWGENF_PLAYEXACTRATE        (1<<5)
#define LWGENF_AUTOKEY              (1<<6)
#define LWGENF_IKENABLE             (1<<7)
#define LWGENF_LMBITEMSELECT        (1<<8)
#define LWGENF_AUTOCONF             (1<<9)
#define LWGENF_DOUBLECLICKBONEMODE  (1<<10)
#define LWGENF_MCENABLE             (1<<11)

#define LWALERT_BEGINNER    0
#define LWALERT_INTERMEDIATE    1
#define LWALERT_EXPERT      2

#define LWITEMCOL_CURRENT   0
#define LWITEMCOL_NORMAL    1
#define LWITEMCOL_SELECTED  2

#define LWGLOBALPOOL_RENDER_GLOBAL  "Global Render Memory"
#define LWGLOBALPOOL_GLOBAL         "Global Memory"

typedef struct st_LWGlobalPool {
    LWMemChunk  (*first) (void);
    LWMemChunk  (*next)  (LWMemChunk);
    const char * /* ascii encoded */ (*ID)    (LWMemChunk);
    int         (*size)  (LWMemChunk);

    LWMemChunk  (*find)  (const char *in_ID /* ascii encoded */);
    LWMemChunk  (*create)(const char *in_ID /* ascii encoded */, int in_size);
} LWGlobalPool;

#endif