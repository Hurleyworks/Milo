/*
* LWSDK Header File
*
* LWBXDF.H -- LightWave BxDF functions.
*
*Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
*
*This file contains confidential and proprietary information of NewTek, Inc.,
*and is subject to the terms of the LightWave End User License Agreement (EULA).
*/

#ifndef LWSDK_BXDF_H
#define LWSDK_BXDF_H

#include <lwrender.h>
#include <lwaovs.h>

#define LWBSDFFUNCS_GLOBAL "BSDF Functions"

//BxDF flags
#define LWBXDF_REFLECTION   (1 << 1)
#define LWBXDF_TRANSMISSION (1 << 2)
#define LWBXDF_DIFFUSE      (1 << 3)
#define LWBXDF_SPECULAR     (1 << 4)
#define LWBXDF_GLOSSY       (1 << 5)

#define LWBXDF_ALL          (LWBXDF_REFLECTION | LWBXDF_TRANSMISSION | LWBXDF_DIFFUSE | LWBXDF_SPECULAR | LWBXDF_GLOSSY)
#define LW_MAX_BXDFS        32 // Maximum number of BxDFs per BSDF

typedef enum {
    LWBTDFVOLUMETYPE_HOMOGENEOUS = 0,
    LWBTDFVOLUMETYPE_HETEROGENEOUS
} LWBTDFVolumeType;

typedef enum
{
    LWBxDFAOVLabelType_Label = 0,
    LWBxDFAOVLabelType_Indirect
} LWBxDFAOVLabelType;

typedef enum
{
    LWBSSRDFAOVLabelType_Label = 0,
    LWBSSRDFAOVLabelType_Indirect
} LWBSSRDFAOVLabelType;

typedef struct st_LWBxDFSample
{
    LWDVector       I;
    LWDVector       dIdx;
    LWDVector       dIdy;
    LWDVector       f;

    double          pdf;
    double          eta;

    unsigned int    flags;
} LWBxDFSample;

// The BSDF system works in shading space. All inputs, such as wo, wi, LWBxDFAccess::I should be transformed into shader space
// using LWShadingGeometry::LWShading::shadingSpace. Partial derivatives, such as LWBxDFAccess::dIdx and LWBxDFAccess::dIdy should remain in world space.
// Output variables, such as LWBxDFSample is in world space.

typedef void    LWBTDFVolume_Extinction(const LWMemChunk memory, const LWVolumeSpot*, LWDVector extinction);
typedef void    LWBTDFVolume_Sample(const LWMemChunk memory, const LWVolumeSpot*, LWDVector emission, LWDVector scattering, LWDVector absorption);
typedef double  LWBTDFVolume_Phase(const LWMemChunk memory, const LWVolumeSpot*, const LWDVector wi, const LWDVector wo);
typedef double  LWBTDFVolume_SamplePhase(const LWMemChunk memory, const LWVolumeSpot*, const LWDVector sample, LWDVector wo);

typedef double  LWBxDF_F(const LWMemChunk memory, const LWDVector wo, const LWDVector wi, LWDVector f);
typedef void    LWBxDF_SampleF(const LWMemChunk memory, const LWShading* sa, const LWDVector sample, LWBxDFSample* so);

// !!! All of the functions in this global are considered UNSAFE, I.e. The validity of the input parameters are not tested in the implementation. !!!
typedef struct st_LWBSDFFuncs
{
    LWBSDF              (*createBSDF)(void);
    void                (*destroyBSDF)(LWBSDF bsdf);
    void                (*resetBSDF)(LWBSDF bsdf);

    LWBxDF              (*createBxDF)(LWBxDF_F* f, LWBxDF_SampleF* sampleF, size_t memorySize, unsigned int flags, const char* label);
    void                (*destroyBxDF)(LWBxDF);

    LWBSSRDF            (*createBSSRDF)(const char* label);
    void                (*destroyBSSRDF)(LWBSSRDF);

    LWBTDFVolume        (*createBTDFVolume)(LWBTDFVolume_Extinction* extinctionF,
                                            LWBTDFVolume_Sample* sampleF,
                                            LWBTDFVolume_Phase* phaseF,
                                            LWBTDFVolume_SamplePhase* samplePhaseF,
                                            size_t memorySize);

    void                (*destroyBTDFVolume)(LWBTDFVolume);

    // Add a BxDF to the BSDF. Returns the LWBxDF that was added, 0 if failed.
    LWBxDF              (*addBxDF)(LWBSDF bsdf, LWBxDF bxdf, LWUserData data, const LWDVector weight);
    // Only use addVolume for BxDFs returned by addBxDF and BxDFs which are BTDFs.
    int                 (*addVolume)(LWBSDF bsdf, LWBxDF bxdf, LWBTDFVolume btdfVolume, LWUserData data, LWBTDFVolumeType type, double stepSize);

    unsigned int        (*numBxDF)(LWBSDF bsdf, unsigned int flags);
    LWBxDF              (*getBxDF)(LWBSDF bsdf, unsigned int index, unsigned int flags);

    double              (*F)(LWBSDF bsdf, const LWDVector wo, const LWDVector wi, unsigned int flags, LWDVector f);
    LWBxDF              (*sampleF)(LWBSDF bsdf, const LWShading* sa, const LWDVector sample, LWBxDFSample* so, unsigned int flags);

    // Only use the below functions for LWBxDFs acquired via getBxDF or returned by sampleF. BxDFs acquired using LWBxDFFuncs will not work.
    double              (*BxDF_F)(LWBxDF bxdf, const LWDVector wo, const LWDVector wi, LWDVector f);
    void                (*BxDF_SampleF)(LWBxDF bxdf, const LWShading* sa, const LWDVector sample, LWBxDFSample* so);

    void                (*BxDF_GetWeight)(LWBxDF bxdf, LWDVector weight);
    void                (*BxDF_SetWeight)(LWBxDF bxdf, const LWDVector weight);

    LWAOVID             (*BxDF_AOVID)(LWBxDF bxdf, LWBxDFAOVLabelType type);
    unsigned int        (*BxDF_Flags)(LWBxDF bxdf);

    LWBTDFVolume        (*BTDF_GetVolume)(LWBxDF btdf);
    LWBTDFVolumeType    (*BTDF_VolumeType)(LWBTDFVolume btdfVolume);
    double              (*BTDF_VolumeStepSize)(LWBTDFVolume btdfVolume);
    void                (*BTDF_VolumeExtinction)(LWBTDFVolume btdfVolume, const LWVolumeSpot*, LWDVector extinction);
    void                (*BTDF_VolumeSample)(LWBTDFVolume btdfVolume, const LWVolumeSpot*, LWDVector emission, LWDVector scattering, LWDVector absorption);
    double              (*BTDF_VolumePhase)(LWBTDFVolume btdfVolume, const LWVolumeSpot*, const LWDVector wi, const LWDVector wo);
    double              (*BTDF_VolumeSamplePhase)(LWBTDFVolume btdfVolume, const LWVolumeSpot*, const LWDVector sample, LWDVector wo);

    // Emission functions
    void                (*getEmission)(LWBSDF bsdf, LWDVector emission);
    void                (*setEmission)(LWBSDF bsdf, const LWDVector emission);

    // BSSRDF functions.
    int                 (*addBSSRDF)(LWBSDF bsdf, LWBSSRDF bssrdf, const LWDVector weight, const LWDVector scattering, double distance);
    unsigned int        (*numBSSRDF)(LWBSDF bsdf);

    void                (*BSSRDF_GetParameters)(LWBSDF bsdf, unsigned int index, LWDVector weight, LWDVector scattering, double* distance);
    void                (*BSSRDF_SetParameters)(LWBSDF bsdf, unsigned int index, const LWDVector weight, const LWDVector scattering, double distance);

    LWBSSRDF            (*getBSSRDF)(LWBSDF bsdf, unsigned int index);
    LWAOVID             (*BSSRDF_AOVID)(LWBSSRDF bssrdf, LWBSSRDFAOVLabelType type);
} LWBSDFFuncs;

#endif