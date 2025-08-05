/*
* LWSDK Header File
*
* LWSURFACESHADER.H -- LightWave Surface Shader.
*
*Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
*
*This file contains confidential and proprietary information of NewTek, Inc.,
*and is subject to the terms of the LightWave End User License Agreement (EULA).
*/
#ifndef LWSDK_SURFSHAD_H
#define LWSDK_SURFSHAD_H

#include <lwserver.h>
#include <lwrender.h>
#include <lwbxdf.h>
#include <lwaovs.h>

#define LWSURFSHAD_HCLASS           "SurfaceShaderHandler"
#define LWSURFSHAD_ICLASS           "SurfaceShaderInterface"
#define LWSURFSHAD_GCLASS           "SurfaceShaderGizmo"
#define LWSURFSHAD_ACLASS           "SurfaceShaderAttribute"
#define LWSURFSHADCLASS_VERSION     1

#define LWSURFACE_REFLECTION    LWBXDF_REFLECTION
#define LWSURFACE_TRANSMISSION  LWBXDF_TRANSMISSION
#define LWSURFACE_DIFFUSE       LWBXDF_DIFFUSE
#define LWSURFACE_SPECULAR      LWBXDF_SPECULAR
#define LWSURFACE_GLOSSY        LWBXDF_GLOSSY
#define LWSURFACE_EMISSION      (1 << 16)
#define LWSURFACE_SSS           (1 << 17)
#define LWSURFACE_RADIOSITY     (1 << 18)

#define LWSURFACE_ALL           (LWSURFACE_REFLECTION | LWSURFACE_TRANSMISSION | LWSURFACE_DIFFUSE | LWSURFACE_SPECULAR | LWSURFACE_GLOSSY | LWSURFACE_EMISSION | LWSURFACE_SSS | LWSURFACE_RADIOSITY)

typedef struct st_LWSurfaceShaderHandler {
    LWInstanceFuncs *inst;
    LWItemFuncs     *item;
    LWRenderFuncs   *rend;

    double          (*shade)(LWInstance, const LWRay*, LWShadingGeometry*, LWBSDF bsdf, LWDVector result);
    double          (*F)(LWInstance, const LWRay*, LWShadingGeometry*, LWBSDF bsdf, const LWDVector wo, const LWDVector wi, unsigned int flags, LWDVector f);
    void            (*sampleF)(LWInstance, const LWRay*, LWShadingGeometry*, LWBSDF bsdf, const LWDVector sample, LWBxDFSample* so, unsigned int bxdfFlags);
} LWSurfaceShaderHandler;

#define LWSURFACEEVALUATIONFUNCS_GLOBAL "LW Surface Evaluation Funcs"

typedef struct st_LWSurfaceEvaluationFuncs
{
    LWSurfaceEvaluatorID    (*create)(LWSurfaceID surf);
    void                    (*destroy)(LWSurfaceEvaluatorID);
    LWError                 (*init)(LWSurfaceEvaluatorID, int mode);
    void                    (*cleanUp)(LWSurfaceEvaluatorID);
    LWError                 (*newTime)(LWSurfaceEvaluatorID, LWFrame frame, LWTime time);

    double                  (*shade)(LWSurfaceEvaluatorID, const LWRay*, LWShadingGeometry*, LWBSDF bsdf, LWDVector result);
    double                  (*F)(LWSurfaceEvaluatorID, const LWRay*, LWShadingGeometry*, LWBSDF bsdf, const LWDVector wo, const LWDVector wi, unsigned int flags, LWDVector f);
    void                    (*sampleF)(LWSurfaceEvaluatorID, const LWRay*, LWShadingGeometry*, LWBSDF bsdf, const LWDVector sample, LWBxDFSample* so, unsigned int bxdfFlags);
} LWSurfaceEvaluationFuncs;

#endif