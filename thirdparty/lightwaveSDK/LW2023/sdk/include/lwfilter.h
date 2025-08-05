/*
 * LWSDK Header File
 *
 * LWFILTER.H -- LightWave Image and Pixel Filters
 *
 * This header contains the basic declarations need to define the
 * simplest LightWave plug-in server.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_FILTER_H
#define LWSDK_FILTER_H

#include <lwtypes.h>
#include <lwmonitor.h>
#include <lwrender.h>

#define LWIMAGEFILTER_HCLASS    "ImageFilterHandler"
#define LWIMAGEFILTER_ICLASS    "ImageFilterInterface"
#define LWIMAGEFILTER_GCLASS    "ImageFilterGizmo"
#define LWIMAGEFILTER_VERSION   7

#define LWPIXELFILTER_HCLASS    "PixelFilterHandler"
#define LWPIXELFILTER_ICLASS    "PixelFilterInterface"
#define LWPIXELFILTER_GCLASS    "PixelFilterGizmo"
#define LWPIXELFILTER_VERSION   10

//
// Buffer types, used with getLine(), getVal(), setVal()
//

#define LWBUFFERPATH_SEPARATOR              '/'

#define LWGROUPBUFFER_MAIN                  "Main"

#define LWBUFFER_SPECIAL                    "Special"
#define LWBUFFER_LUMINOUS                   "Luminous"
#define LWBUFFER_DIFFUSE                    "Diffuse"
#define LWBUFFER_SPECULAR                   "Specular"
#define LWBUFFER_MIRROR                     "Mirror"
#define LWBUFFER_TRANSPARENCY               "Trans"
#define LWBUFFER_RAW_RGB                    "Raw_RGB"
#define LWBUFFER_SHADING                    "Shading"
#define LWBUFFER_SHADOW                     "Shadow"
#define LWBUFFER_GEOMETRY                   "Geometry"
#define LWBUFFER_DEPTH                      "Depth"
#define LWBUFFER_COVERAGE                   "Coverage"
#define LWBUFFER_DIFFUSE_SHADING            "Diffuse_Shading"
#define LWBUFFER_SPECULAR_SHADING           "Specular_Shading"
#define LWBUFFER_MOTION_XY                  "Motion"
#define LWBUFFER_MOTION_X                   "Motion_X"
#define LWBUFFER_MOTION_Y                   "Motion_Y"
#define LWBUFFER_REFLECTION_RGB             "Reflection"
#define LWBUFFER_DIFFUSE_RGB                "Diffuse_RGB"
#define LWBUFFER_SPECULAR_RGB               "Specular_RGB"
#define LWBUFFER_BACKDROP_RGB               "Backdrop"
#define LWBUFFER_PREEFFECT_RGB              "Pre_Effect"
#define LWBUFFER_FINAL_RENDER_RGBA          "Final_Render"
#define LWBUFFER_ALPHA                      "Alpha"
#define LWBUFFER_REFRACTION_RGBA            "Refraction"
#define LWBUFFER_NORMALS_XYZ                "Normals"
#define LWBUFFER_SURFACE_ID                 "Surface_ID"
#define LWBUFFER_OBJECT_ID                  "Object_ID"
#define LWBUFFER_RADIOSITY_RGB              "Radiosity"
#define LWBUFFER_AMBIENT_OCCLUSION_RGB      "Ambient_Occlusion"
#define LWBUFFER_UV_TANGENTSPACE_T_XYZ      "UV_Tangent_Space_T_XYZ"
#define LWBUFFER_UV_TANGENTSPACE_B_XYZ      "UV_Tangent_Space_B_XYZ"
#define LWBUFFER_UV_TANGENTSPACE_N_XYZ      "UV_Tangent_Space_N_XYZ"
#define LWBUFFER_CAMERA_TANGENTSPACE_XYZ    "Camera_Tangent_Space_XYZ"
#define LWBUFFER_EDGES_RGBA                 "Edges"
#define LWBUFFER_FINALRENDER_CC_RGBA        "Final_Render_CC"
#define LWBUFFER_SSS_DIRECT_RGB             "SSS_Direct"
#define LWBUFFER_SSS_INDIRECT_RGB           "SSS_Indirect"
#define LWBUFFER_VOLUMETRIC_DIRECT_RGB      "Volumetric_Direct"
#define LWBUFFER_VOLUMETRIC_LIGHT_RGB       "Volumetric_Light"
#define LWBUFFER_VOLUMETRIC_INDIRECT_RGB    "Volumetric_Indirect"
#define LWBUFFER_VOLUMETRIC_EMISSION_RGB    "Volumetric_Emission"
#define LWBUFFER_VOLUMETRIC_TRANSMIT_RGB    "Volumetric_Transmittance"
#define LWBUFFER_TRANSLUCENCY_RGB           "Translucency"
#define LWBUFFER_TRANSPARENCY_RGB           "Transparency"
#define LWBUFFER_LUMINOSITY_RGB             "Luminosity"
#define LWBUFFER_WORLD_POSITION_XYZ         "World_Position"
#define LWBUFFER_OBJECT_POSITION_XYZ        "Object_Position"
#define LWBUFFER_TEXTURE_POSITION_XYZ       "Texture_Position"
#define LWBUFFER_UV                         "UV"
#define LWBUFFER_DPDU_XYZ                   "dPdu"
#define LWBUFFER_DPDV_XYZ                   "dPdv"
#define LWBUFFER_LENS_FLARE_RGBA            "Lens_Flare"
#define LWBUFFER_DIFFUSE_DIRECT_RGB         "Diffuse_Direct"
#define LWBUFFER_DIFFUSE_INDIRECT_RGB       "Diffuse_Indirect"
#define LWBUFFER_SPECULAR_DIRECT_RGB        "Specular_Direct"
#define LWBUFFER_SPECULAR_INDIRECT_RGB      "Specular_Indirect"
#define LWBUFFER_SPECULAR_RHO               "Specular_Rho"
#define LWBUFFER_SPECULAR_BY_ID_RGBA        "Specular_By_ID"
#define LWBUFFER_TRANSLUCENCY_DIRECT_RGB    "Translucency_Direct"
#define LWBUFFER_TRANSLUCENCY_INDIRECT_RGB  "Translucency_Indirect"
#define LWBUFFER_FOG_RGB                    "Fog"
#define LWBUFFER_LEGACY_VOLUMETRIC_RGB      "Legacy_Volumetric"

/* LW Filter Info Global. */

#define LWFILTERINFO_GLOBAL "LW Filter Info Global"

typedef struct st_LWFilterGlobal {
    LWCStringUTF8 *(*buffersAvailable)();
    LWCStringUTF8 *(*buffersInUse)();
    int            (*isBufferInUse)( LWCStringUTF8 name );
} LWFilterGlobal;

/* Render flags. */

enum {
    LWPFF_MULTITHREADED = 1<<0,
    LWPFF_EVERYPIXEL    = 1<<1,
    LWPFF_BEFOREVOLUME  = 1<<2,
    LWPFF_RAYTRACE      = 1<<3
};

typedef struct st_LWFilterAccess {
        int               width, height;
        LWFrame           frame;
        LWTime            start, end;
        int             (*layers  )( LWCStringASCII type );
        float *         (*getLine )( LWCStringASCII type, int layer, int y);
        void            (*setRGB  )(int x, int y, const LWFVector rgb);
        void            (*setAlpha)(int x, int y, float alpha);
        LWMonitor        *monitor;
} LWFilterAccess;

/* Note: Flags now returns a NULL-terminated array of string entries. */
/* static LWCStringUTF8 flags[] = { LWBUFFER_SPECIAL, LWBUFFER_LUMINOUS, 0 }; */

typedef struct st_LWImageFilterHandler {
        LWInstanceFuncs  *inst;
        LWItemFuncs      *item;
        LWError         (*process) (LWInstance, const LWFilterAccess *);
        LWCStringASCII *(*flags)   (LWInstance);
} LWImageFilterHandler;


typedef struct st_LWPixelAccess {
        double                        sx, sy;
        int                         (*layers )(LWCStringASCII type);
        void                        (*getVal )(LWCStringASCII type, int num, float *);
        void                        (*setRGBA)(const float[4]);
        void                        (*setVal )(LWCStringASCII type, int num, float *);
        LWIlluminateFunc             *illuminate;
        LWRayTraceFunc               *rayTrace;
        LWRayCastFunc                *rayCast;
        LWRayShadeFunc               *rayShade;
        LWRayTraceModeFunc           *rayTraceMode;
        LWIlluminateSampleFunc       *illuminateSample;
        RandomFloatData               randomData;
        LWRandomFloatFunc            *randomFloat;
        LWIlluminateNormalFunc       *illuminateNormal;
        LWIlluminateSampleNormalFunc *illuminateSampleNormal;
        LWRayTraceDataFunc           *rayTraceData;
        LWRayTraceShadeFunc          *rayTraceShade;
        LWRenderData                  render;
        LWSamplerState                samplerState;
        LWGetSamplerRangeFunc        *getSamplerRange;
        LWGetSample2DFunc            *getSample2D;
        LWGetSample1DFunc            *getSample1D;

        const LWRay* ray;
} LWPixelAccess;

typedef struct st_LWPixelFilterHandler {
        LWInstanceFuncs  *inst;
        LWItemFuncs      *item;
        LWRenderFuncs    *rend;
        void            (*evaluate)    (LWInstance, const LWPixelAccess *);
        LWCStringUTF8  *(*flags)       (LWInstance);
        unsigned int    (*renderFlags) (LWInstance);
} LWPixelFilterHandler;


typedef unsigned int LWFilterContext;


#define LWFCF_PREPROCESS  (1<<0) /* Filter applied in image editor or as pre process */


#endif