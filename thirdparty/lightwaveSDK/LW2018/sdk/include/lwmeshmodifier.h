/*
 * LWSDK Header File
 * lwmeshmodifier.h -- LightWave Mesh Modifiers
 *
 *Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of NewTek, Inc.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_MESHMODIFIER_H
#define LWSDK_MESHMODIFIER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwrender.h>
#include <lwmeshes.h>



#define LWMESHMODIFIER_HCLASS   "MeshModifierHandler"
#define LWMESHMODIFIER_ICLASS   "MeshModifierInterface"
#define LWMESHMODIFIER_GCLASS   "MeshModifierGizmo"
#define LWMESHMODIFIER_ACLASS   "MeshModifierAttrui"
#define LWMESHMODIFIER_VERSION  1


typedef struct st_LWMeshModifierAccess {
    LWMutableMeshID mesh;

    unsigned int status;

    LWDMatrix9   dirToWorld;            // worldPos = dirToWorld * localPos + offsetToWorld
    LWDVector    offsetToWorld;
    LWDMatrix9   dirFromWorld;          // localPos = dirFromWorld * worldPos + offsetFromWorld
    LWDVector    offsetFromWorld;
    LWDMatrix9   normToWorld;           // worldNormal = normalize(normToWorld * localNormal)
    LWDMatrix9   normFromWorld;         // localNormal = normalize(normFromWorld * worldNormal)
} LWMeshModifierAccess;

typedef struct st_LWMeshModifierHandler {
    LWInstanceFuncs *inst;
    LWItemFuncs     *item;
    LWRenderFuncs   *rend;
    unsigned int   (*flags)    (LWInstance);
    unsigned int   (*begin)    (LWInstance, LWMeshModifierAccess *);
    unsigned int   (*evaluate) (LWInstance, LWMeshModifierAccess *);
    void           (*end)      (LWInstance, LWMeshModifierAccess *);
} LWMeshModifierHandler;



#define LWMESHDEFORMER_HCLASS   "MeshDeformerHandler"
#define LWMESHDEFORMER_ICLASS   "MeshDeformerInterface"
#define LWMESHDEFORMER_GCLASS   "MeshDeformerGizmo"
#define LWMESHDEFORMER_ACLASS   "MeshDeformerAttrui"
#define LWMESHDEFORMER_VERSION  1

typedef struct st_LWMeshDeformerAccess {
    LWDeformableMeshID mesh;

    unsigned int status;

    LWDMatrix9   dirToWorld;            // worldPos = dirToWorld * localPos + offsetToWorld
    LWDVector    offsetToWorld;
    LWDMatrix9   dirFromWorld;          // localPos = dirFromWorld * worldPos + offsetFromWorld
    LWDVector    offsetFromWorld;
    LWDMatrix9   normToWorld;           // worldNormal = normalize(normToWorld * localNormal)
    LWDMatrix9   normFromWorld;         // localNormal = normalize(normFromWorld * worldNormal)

    size_t       startIndex;
    size_t       endIndex;

    const float *oPos;
    float       *fPos;
    size_t       posStride;             // Number of floats between start of consecutive indices
} LWMeshDeformerAccess;

typedef struct st_LWMeshDeformerHandler {
    LWInstanceFuncs *inst;
    LWItemFuncs     *item;
    LWRenderFuncs   *rend;
    unsigned int   (*flags)    (LWInstance);
    unsigned int   (*begin)    (LWInstance, LWMeshDeformerAccess *);
    unsigned int   (*evaluate) (LWInstance, LWMeshDeformerAccess *);
    void           (*end)      (LWInstance, LWMeshDeformerAccess *);
} LWMeshDeformerHandler;



// flags() and begin()
#define LWMESHMODIFIERF_SKIP        (1U << 0)   // Do not evaluate
#define LWMESHMODIFIERF_NO_MT       (1U << 1)   // No multithreaded evaluation

// evaluate() and status
#define LWMESHMODIFIER_UNCHANGED    0           // No changes made to mesh
#define LWMESHMODIFIER_VERTSMOVED   (1U << 0)   // Modifier made changes to vertex positions
#define LWMESHMODIFIER_GEOCHANGED   (1U << 1)   // Modifier made changes to the geometry
#define LWMESHMODIFIER_INTERRUPT    (1U << 31)  // Evaluation was interrupted


#ifdef __cplusplus
}
#endif

#endif