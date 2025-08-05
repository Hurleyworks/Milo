/*
 * LWSDK Header File
 *
 * LWSURF.H -- LightWave Surfaces
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_SURF_H
#define LWSDK_SURF_H

typedef void* LWSurfLibID;

#include <lwtypes.h>
#include <lwrender.h>
#include <lwtxtr.h>
#include <lwenvel.h>
#include <lwimage.h>
#include <lwnodes.h>
#include <lwbxdf.h>
#include <lwaovs.h>

#define LWSURFACEFUNCS_GLOBAL "Surface Functions 6"

#define SURF_COLR   "BaseColor"
#define SURF_LUMI   "Luminosity"
#define SURF_DIFF   "Diffuse"
#define SURF_SPEC   "Specularity"
#define SURF_REFL   "Reflectivity"
#define SURF_TRAN   "Transparency"
#define SURF_TRNL   "Translucency"
#define SURF_RIND   "IOR"
#define SURF_BUMP   "Bump"
#define SURF_GLOS   "Glossiness"
#define SURF_BUF1   "SpecialBuffer1"
#define SURF_BUF2   "SpecialBuffer2"
#define SURF_BUF3   "SpecialBuffer3"
#define SURF_BUF4   "SpecialBuffer4"
#define SURF_SHRP   "DiffuseSharpness"
#define SURF_BDRP   "BumpDropoff"
#define SURF_RMUL   "RadiosityMultiply"
#define SURF_SMAN   "SmoothingAngle"
#define SURF_RSAN   "ReflectionSeamAngle"
#define SURF_TSAN   "RefractionSeamAngle"
#define SURF_RBLR   "ReflectionBlurring"
#define SURF_TBLR   "RefractionBlurring"
#define SURF_CLRF   "ColorFilter"
#define SURF_CLRH   "ColorHighlights"
#define SURF_ADTR   "AdditiveTransparency"
#define SURF_AVAL   "AlphaValue"
#define SURF_GVAL   "GlowValue"
#define SURF_LCOL   "LineColor"
#define SURF_LSIZ   "LineSize"
#define SURF_ALPH   "AlphaOptions"
#define SURF_RFOP   "ReflectionOptions"
#define SURF_TROP   "RefractionOptions"
#define SURF_SIDE   "Sidedness"
#define SURF_NVSK   "ExcludeFromVStack"
#define SURF_GLOW   "Glow"
#define SURF_LINE   "RenderOutlines"
#define SURF_RIMG   "ReflectionImage"
#define SURF_TIMG   "RefractionImage"
#define SURF_VCOL   "VertexColoring"
#define SURF_NORM   "VertexNormal"
#define SURF_CMAP   "ClipMap"
#define SURF_REFLSAM "ReflectionSamples"
#define SURF_REFRSAM "RefractionSamples"
#define SURF_OPAQ   "Opaque"
#define SURF_GSIZ   "GlowSize"

#define SURF_BSDF_INPUT 0 // The index for the NodeInputID of the bsdf (bsdf)
#define SURF_NORMAL_INPUT 1 // The index for the NodeInputID of the normal (vector)
#define SURF_BUMP_INPUT 2 // The index for the NodeInputID of the normal (vector)
#define SURF_DISPLACEMENT_INPUT 3 // The index for the NodeInputID for the displacement (scalar)
#define SURF_CLIPMAP_INPUT 4 // The index for the NodeInputID for the clipmap (scalar)

typedef struct st_LWSurfaceCustomAOV
{
    NodeInputID input;
    LWAOVID aovID;
} LWSurfaceCustomAOV;

typedef struct st_LWSurfaceFuncs
{
    LWSurfaceID   (*create)(LWCStringUTF8 objName,LWCStringUTF8 surfName);
    LWSurfaceID   (*first)(void);
    LWSurfaceID   (*next)(LWSurfaceID surf);
    LWSurfaceID*  (*byName)(LWCStringUTF8 name,LWCStringUTF8 objName);
    LWSurfaceID*  (*byObject)(LWCStringUTF8 name);
    LWCStringUTF8 (*name)(LWSurfaceID surf);
    LWCStringUTF8 (*sceneObject)(LWSurfaceID surf);

    int           (*getInt)(LWSurfaceID surf,LWCStringUTF8 channel);
    double*       (*getFlt)(LWSurfaceID surf,LWCStringUTF8 channel);
    LWEnvelopeID  (*getEnv)(LWSurfaceID surf,LWCStringUTF8 channel);
    LWTextureID   (*getTex)(LWSurfaceID surf,LWCStringUTF8 channel);
    LWImageID     (*getImg)(LWSurfaceID surf,LWCStringUTF8 channel);

    LWChanGroupID (*chanGrp)(LWSurfaceID surf);
    LWCStringUTF8 (*getColorVMap)(LWSurfaceID surf);
    void          (*setColorVMap)(LWSurfaceID surf,LWCStringUTF8 vmapName,int type);

    LWSurfLibID   (*createLib)(void);
    void          (*destroyLib)(LWSurfLibID lib);
    void          (*copyLib)(LWSurfLibID to, LWSurfLibID from);
    LWSurfLibID   (*objectLib)(LWCStringUTF8 objname);
    LWSurfLibID   (*loadLib)(LWCStringUTF8 name);
    LWError       (*saveLib)(LWSurfLibID lib, LWCStringUTF8 name);
    int           (*renameLib)(LWSurfLibID lib, LWCStringUTF8 name);
    LWCStringUTF8 (*nameLib)(LWSurfLibID lib);
    int           (*slibCount)(LWSurfLibID lib);
    LWSurfaceID   (*slibByIndex)(LWSurfLibID lib,int idx);
    NodeEditorID  (*getNodeEditor)( LWSurfaceID surf);

    int           (*setInt)(LWSurfaceID,LWCStringUTF8, int);
    int           (*setFlt)(LWSurfaceID,LWCStringUTF8, double *);
    int           (*setEnv)(LWSurfaceID,LWCStringUTF8, LWEnvelopeID);
    int           (*setTex)(LWSurfaceID,LWCStringUTF8, LWTextureID);
    int           (*setImg)(LWSurfaceID,LWCStringUTF8, LWImageID);
    int           (*rename)(LWSurfaceID,LWCStringUTF8);
    int           (*copy)(LWSurfaceID,LWSurfaceID);

    void          (*copyLibByName)  (LWSurfLibID to, LWSurfLibID from);
    LWCStringUTF8 (*getNormalVMap)  (LWSurfaceID surf);
    LWCStringUTF8 (*server)         (LWSurfaceID surf, int);
    unsigned int  (*serverFlags)    (LWSurfaceID surf, int);
    LWInstance    (*serverInstance) (LWSurfaceID surf, int);

    LWSurfaceID   (*createInLib)(LWSurfLibID, LWCStringUTF8 surfName);
    void          (*destroyInLib)(LWSurfLibID, LWSurfaceID);
    int           (*surfaceSave)(LWSurfLibID, LWSurfaceID, const LWSaveState *);
    LWSurfaceID   (*surfaceLoad)(LWSurfLibID, const LWLoadState *);

    unsigned int                (*numCustomAOVs)(LWSurfaceID);
    const LWSurfaceCustomAOV*   (*customAOV)(LWSurfaceID, unsigned int);

    LWCStringUTF8 (*getMaterial)(LWSurfaceID);
    int           (*setMaterial)(LWSurfaceID, LWCStringUTF8);
    LWCStringUTF8 (*getShadingModel)(LWSurfaceID);
    int           (*setShadingModel)(LWSurfaceID, LWCStringUTF8);
    unsigned int  (*getIndexID)(LWSurfaceID);
    void          (*getOGLMaterial)(LWSurfaceID, LWNodeOGLMaterial *);
} LWSurfaceFuncs;

#endif