/*
 * LWSDK Header File
 *
 * LWLIGHT.H -- LightWave Lighting
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_LIGHT_H
#define LWSDK_LIGHT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwtypes.h>
#include <lwhandler.h>
#include <lwrender.h>
#include <lwbxdf.h>

#define LWLIGHT_HCLASS "LightHandler"
#define LWLIGHT_ICLASS "LightInterface"
#define LWLIGHT_GCLASS "LightGizmo"
#define LWLIGHT_VERSION 4

// Light plugins can return an array of standard lights
// to create a preview. The array is terminated with a
// preview light type of LWPREVIEWLIGHT_END.
typedef struct st_LWPreviewLight
{
    int previewtype;        // One of the LWPREVIEWLIGHT_ values below

    LWDVector worldPos;     // The light to world transformation matrix
    LWDVector toWorld[3];

    LWDVector color;        // Color and intensity of the light
    double intensity;

    double range;           // Falloff
    int falloff;

    double coneAngle;       // Cone for spot lights
    double coneEdge;
} LWPreviewLight;

// Preview light types
#define LWPREVIEWLIGHT_DISTANT 0
#define LWPREVIEWLIGHT_POINT   1
#define LWPREVIEWLIGHT_SPOT    2
#define LWPREVIEWLIGHT_AMBIENT 3

#define LWPREVIEWLIGHT_END     -1

// Light sampling fills in an array of samples
typedef struct st_LWLightSample
{
    LWDVector   direction; // Direction towards the sample from the spot
    LWDVector   color;     // Color, including intensity
    double      distance;  // Distance to sample from the spot; FLT_MAX if infinite
    double      pdf;       // Probability distribution function for the sample
} LWLightSample;

// Emission evaluation fills in an array of emissions
typedef struct st_LWLightEmission
{
    LWDVector   color;     // Color, including intensity
    double      distance;  // Distance to sample from the spot; FLT_MAX if infinite
    double      pdf;       // Probability distribution function for the sample
} LWLightEmission;

// Emission sample evaluation fills in an array of emission samples
typedef struct st_LWLightEmissionSample
{
    LWRay*    ray;
    LWDVector normal;
    double    posPdf;
    double    dirPdf;
} LWLightEmissionSample;

typedef struct st_LWLightEvaluationMemory
{
    LWMemChunk                      memory;         // Light evaluation memory
    struct st_LWLightSample*        samples;        // Memory for LWLightSample array
    struct st_LWLightEmission*      emissions;      // Memory for LWLightEmission array
    struct st_LWBxDFSample*         bxdfSamples;    // Memory for LWBxDFSample array
    LWDVector*                      randomSamples;  // Memory for random samples array
} LWLightEvaluationMemory;

// Light handler flags
#define LWLIGHTTYPEF_NO_FALLOFF           (1 << 0)    // Do not show standard falloff gadgets
#define LWLIGHTTYPEF_INFINITE             (1 << 1)    // If the light is an infinite light, such as Distant or Environment light.
#define LWLIGHTTYPEF_NO_EMISSION          (1 << 2)    // The light does not implement the "emission" function.
#define LWLIGHTTYPEF_AMBIENT              (1 << 3)    // If the light is an ambient light.

// Light evaluation flags
#define LWLIGHTEVAL_VOL_LIGHT             (1 << 0)    // Evaluate is being called when evaluating volumetric lights.

// Light evaluation Return flags
#define LWLIGHTSAMPLE_DELTA_LIGHT         (1 << 0)    // The light is described by a delta distribution. (Point lights, etc)
#define LWLIGHTSAMPLE_NO_SAMPLE           (1 << 1)    // Set if the light should not be sampled.

typedef struct st_LWLightHandler
{
    LWInstanceFuncs*        inst;
    LWItemFuncs*            item;
    LWRenderFuncs*          rend;

    unsigned int            (*flags)(LWInstance);
    LWError                 (*newFrame)(LWInstance, const LWFrameInfo* frameinfo, unsigned int* maxIlluminations);
    const LWPreviewLight*   (*preview)(LWInstance);

    size_t                  (*memorySize)(LWInstance);

    unsigned int            (*evaluate)(LWInstance, LWMemChunk memory, const LWRay*, unsigned int flags, const LWDVector p, unsigned int* samples);
    unsigned int            (*sample)(LWInstance, const LWMemChunk memory, const LWRay*, const LWDVector p,
                                    unsigned int maxSamples, const LWDVector samples[], LWLightSample lightsamples[]);
    void                    (*emission)(LWInstance, const LWMemChunk memory, const LWRay*, const LWDVector p,
                                        unsigned int maxSamples, const LWBxDFSample bxdfsamples[], LWLightEmission emissions[]);
    int                     (*localBounds)(LWInstance, LWDVector min, LWDVector max);
    int                     (*worldBounds)(LWInstance, LWDVector min, LWDVector max);
    int                     (*intersect)(LWInstance, const LWRay*, LWShadingGeometry* is);
    int                     (*power)(LWInstance, LWDVector power);

    int                     (*volumeSegment)(LWInstance, const LWMemChunk memory, const LWRay*, unsigned int flags, double tMin, double tMax, double* segmin, double* segmax);
    double                  (*volumeSample)(LWInstance, const LWMemChunk memory, const LWRay*, double segmin, double segmax, double ru, double *dist);
    double                  (*volumeSamplePdf)(LWInstance, const LWMemChunk memory, const LWRay*, double segmin, double segmax, double dist);

    int                     (*sampleEmission)(LWInstance, const double rndsample[4], LWTime fracTime, LWLightEmissionSample* le);
    unsigned int            (*ambient)(LWInstance, const LWRay*, unsigned int flags, const LWDVector p, LWDVector ambientcolor);
} LWLightHandler;

#define LWLIGHTEVALUATIONFUNCS_GLOBAL "LW Light Evaluation Funcs 3"

typedef struct st_LWLightEvaluationFuncs
{
    LWLightEvaluatorID  (*create)(LWItemID light);
    void                (*destroy)(LWLightEvaluatorID);
    LWCStringUTF8       (*serverName)(LWLightEvaluatorID);
    unsigned int        (*flags)(LWLightEvaluatorID);
    LWError             (*init)(LWLightEvaluatorID, int mode);
    void                (*cleanUp)(LWLightEvaluatorID);
    LWError             (*newTime)(LWLightEvaluatorID, LWFrame frame, LWTime time);
    LWError             (*newFrame)(LWLightEvaluatorID, const LWFrameInfo* frameinfo, unsigned int* maxIlluminations);

    size_t              (*memorySize)(LWLightEvaluatorID);

    unsigned int        (*evaluate)(LWLightEvaluatorID, LWMemChunk memory, const LWRay*, unsigned int flags, const LWDVector p, unsigned int* samples);
    unsigned int        (*sample)(LWLightEvaluatorID, const LWMemChunk memory, const LWRay*, const LWDVector p,
                                  unsigned int maxSamples, const LWDVector samples[], LWLightSample lightsamples[]);
    void                (*emission)(LWLightEvaluatorID, const LWMemChunk memory, const LWRay*, const LWDVector p,
                                    unsigned int nmaxSamples, const LWBxDFSample bxdfsamples[], LWLightEmission emissions[]);
    int                 (*localBounds)(LWLightEvaluatorID, LWDVector min, LWDVector max);
    int                 (*worldBounds)(LWLightEvaluatorID, LWDVector min, LWDVector max);
    int                 (*intersect)(LWLightEvaluatorID, const LWRay*, LWShadingGeometry* is);
    int                 (*power)(LWLightEvaluatorID, LWDVector power);

    int                 (*volumeSegment)(LWLightEvaluatorID, const LWMemChunk memory, const LWRay*, unsigned int flags, double tMin, double tMax, double* segmin, double* segmax);
    double              (*volumeSample)(LWLightEvaluatorID, const LWMemChunk memory, const LWRay*, double segmin, double segmax, double ru, double *dist);
    double              (*volumeSamplePdf)(LWLightEvaluatorID, const LWMemChunk memory, const LWRay*, double segmin, double segmax, double dist);

    int                 (*sampleEmission)(LWLightEvaluatorID, const double rndsample[4], LWTime fracTime, LWLightEmissionSample* le);
    unsigned int        (*ambient)(LWLightEvaluatorID, const LWRay*, unsigned int flags, const LWDVector p, LWDVector ambientcolor);
} LWLightEvaluationFuncs;

#ifdef __cplusplus
}
#endif

#endif