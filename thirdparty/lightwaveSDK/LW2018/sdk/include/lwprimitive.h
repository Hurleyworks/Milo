/*
 * LWSDK Header File
 * lwprimitive.h -- LightWave Primitive
 *
  *Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
  *
  *This file contains confidential and proprietary information of NewTek, Inc.,
  *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_PRIMITIVE_H
#define LWSDK_PRIMITIVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwhandler.h>
#include <lwrender.h>

#define LWPRIMITIVE_HCLASS "PrimitiveHandler"
#define LWPRIMITIVE_ICLASS "PrimitiveInterface"
#define LWPRIMITIVE_GCLASS "PrimitiveGizmo"
#define LWPRIMITIVE_ACLASS "PrimitiveAttribute"

#define LWPRIMITIVE_VERSION 1

#define LWPRIMITIVEFLAG_HAS_SOLIDS   (1 << 0)
#define LWPRIMITIVEFLAG_HAS_VOLUMES  (1 << 1)

typedef enum
{
    LWPRIMITIVETYPE_SOLID = 0,
    LWPRIMITIVETYPE_VOLUME
} LWPrimitiveType;

typedef enum
{
    LWPRIMITIVECOORDSYS_LOCAL = 0,
    LWPRIMITIVECOORDSYS_WORLD
} LWPrimitiveCoordinateSys;

typedef struct st_LWPrimitiveInstance
{
    LWDMatrix4 matrix;
    LWDMatrix4 inverse;
} LWPrimitiveInstance;

typedef void* LWVolumeSegmentFuncData;
typedef struct st_LWVolumeSegmentFunc
{
    LWVolumeSegmentFuncData data;
    void(*addSegment)(LWVolumeSegmentFuncData, double start, double end, double stepSize);
} LWVolumeSegmentFunc;

typedef struct st_LWPrimitiveHandler
{
    LWInstanceFuncs *inst;
    LWItemFuncs     *item;
    LWRenderFuncs   *rend;

    unsigned int    (*flags)(LWInstance);
    LWError         (*newFrame)(LWInstance, const LWFrameInfo* frameinfo);
    int             (*bounds)(LWInstance, LWPrimitiveType, LWDVector min, LWDVector max, LWPrimitiveCoordinateSys* coordSys);

    // Solids
    size_t          (*memorySize)(LWInstance);
    int             (*nearestPoint)(LWInstance, const LWPrimitiveInstance* pinst, const LWRay* ray, LWShadingGeometry* is);
    int             (*intersect)(LWInstance, const LWPrimitiveInstance* pinst, const LWRay* ray, LWShadingGeometry* is);
    double          (*edgePixelRadius)(LWInstance);
    double          (*edgeWorldRadius)(LWInstance);
    LWSurfaceID     (*surface)(LWInstance, const LWShadingGeometry* is);
    double          (*area)(LWInstance, const LWPrimitiveInstance* pinst, LWTime fracTime);
    int             (*sample)(LWInstance, const LWPrimitiveInstance* pinst, LWTime fracTime, const LWDVector randomSample, LWDVector p, LWDVector n);
    double          (*pdf)(LWInstance, const LWPrimitiveInstance* pinst, const LWRay* ray, const LWShadingGeometry* is);
    void            (*opacity)(LWInstance, const LWPrimitiveInstance* pinst, const LWRay* ray, const LWShadingGeometry* is, LWDVector opa);

    // Volumes
    int             (*volumeSegment)(LWInstance, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, double tMin, double tMax, LWVolumeSegmentFunc* segfunc);
    int             (*volumeExtinction)(LWInstance, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, double length, double u, LWDVector extinction);
    int             (*volumeSample)(LWInstance, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, LWDVector emission, LWDVector scattering, LWDVector absorption);
    double          (*volumePhase)(LWInstance, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, const LWDVector wi, const LWDVector wo);
    double          (*volumeSamplePhase)(LWInstance, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, const LWDVector sample, LWDVector wo);
} LWPrimitiveHandler;

#define LWPRIMITIVEEVALUATIONFUNCS_GLOBAL "LW Primitive Evaluation Funcs"

typedef struct st_LWPrimitiveEvaluationFuncs
{
    LWPrimitiveEvaluatorID  (*create)(LWItemID item);
    void                    (*destroy)(LWPrimitiveEvaluatorID);
    const char*             (*serverName)(LWPrimitiveEvaluatorID);
    LWItemID                (*itemID)(LWPrimitiveEvaluatorID);

    LWError                 (*init)(LWPrimitiveEvaluatorID, int mode);
    void                    (*cleanUp)(LWPrimitiveEvaluatorID);
    LWError                 (*newTime)(LWPrimitiveEvaluatorID, LWFrame frame, LWTime in_time);

    unsigned int            (*flags)(LWPrimitiveEvaluatorID);
    LWError                 (*newFrame)(LWPrimitiveEvaluatorID, const LWFrameInfo* frameinfo);
    int                     (*bounds)(LWPrimitiveEvaluatorID, LWPrimitiveType, LWDVector min, LWDVector max, LWPrimitiveCoordinateSys* coordSys);

    // Solids
    size_t                  (*memorySize)(LWPrimitiveEvaluatorID);
    int                     (*nearestPoint)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWRay* ray, LWShadingGeometry* is);
    int                     (*intersect)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWRay* ray, LWShadingGeometry* is);
    double                  (*edgePixelRadius)(LWPrimitiveEvaluatorID);
    double                  (*edgeWorldRadius)(LWPrimitiveEvaluatorID);
    LWSurfaceID             (*surface)(LWPrimitiveEvaluatorID, const LWShadingGeometry* is);
    double                  (*area)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, LWTime fracTime);
    int                     (*sample)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, LWTime fracTime, const LWDVector randomSample, LWDVector p, LWDVector n);
    double                  (*pdf)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWRay* ray, const LWShadingGeometry* is);
    void                    (*opacity)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWRay* ray, const LWShadingGeometry* is, LWDVector opa);

    // Volumes
    int                     (*volumeSegment)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, double tMin, double tMax, LWVolumeSegmentFunc* segfunc);
    int                     (*volumeExtinction)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, double length, double u, LWDVector extinction);
    int                     (*volumeSample)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, LWDVector emission, LWDVector scattering, LWDVector absorption);
    double                  (*volumePhase)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, const LWDVector wi, const LWDVector wo);
    double                  (*volumeSamplePhase)(LWPrimitiveEvaluatorID, const LWPrimitiveInstance* pinst, const LWVolumeSpot*, const LWDVector sample, LWDVector wo);

    const LWItemID*         (*usedItems)(LWPrimitiveEvaluatorID);
} LWPrimitiveEvaluationFuncs;

#ifdef __cplusplus
}
#endif

#endif //LWSDK_PRIMITIVE_H