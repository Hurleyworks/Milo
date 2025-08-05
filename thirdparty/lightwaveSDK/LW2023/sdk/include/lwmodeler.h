/*
 * LWSDK Header File
 *
 * LWMODELER.H -- LightWave Modeler Global State
 *
 * This header contains declarations for the global services and
 * internal states of the Modeler host application.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_MODELER_H
#define LWSDK_MODELER_H

#include <lwtypes.h>
#include <lwmeshtypes.h>

typedef int EltOpLayer;
#define OPLYR_PRIMARY    0
#define OPLYR_FG         1
#define OPLYR_BG         2
#define OPLYR_SELECT     3
#define OPLYR_ALL        4
#define OPLYR_EMPTY      5
#define OPLYR_NONEMPTY   6

typedef int EltOpSelect;
#define OPSEL_GLOBAL     0
#define OPSEL_USER       1
#define OPSEL_DIRECT     2

#define LWAC_MOUSE 0
#define LWAC_ORIGIN 1
#define LWAC_SELECTION 2
#define LWAC_PIVOT 3
#define LWAC_SELECTION_CLUSTER 4


#define LWSTATEQUERYFUNCS_GLOBAL    "LWM: State Query 5"

typedef struct st_LWStateQueryFuncs
{
    /// @return The number of data layers for the current object.
    int (*numLayers)(void);

    /// @return The bits for the data layers included in the EltOpLayer selection.
    /// If bit i of the mask is set, then layer i + 1 of the current object belongs
    /// to the set defined by the oplayer argument. This function is provided primarily
    /// for backward compatibility. New code should use the layerList function, which is
    /// designed for multiple objects and an unlimited number of layers.
    unsigned int (*layerMask)(EltOpLayer oplayer);

    /// @return The name of the current default surface.
    LWCStringUTF8 (*surface)(void);

    unsigned int (*bbox)(EltOpLayer, double* minmax);

    /// @return A string containing layer numbers for the given EltOpLayer and object.
    /// The layer numbers in the string are separated by spaces, with the highest
    /// numbered layer listed first. The object name is its filename, or NULL for the
    /// current object.
    LWCStringASCII (*layerList)(EltOpLayer oplayer, LWCStringUTF8 objname);

    /// @return The filename of the current object. If the geometry in the current layers
    /// hasn't been saved to a file yet, this returns the reference name (the name that would
    /// be returned by the Object Functions refName function). If no object has been loaded
    /// into Modeler, this returns NULL.
    LWCStringUTF8 (*object)(void);

    /// @return The state of a user interface setting.
    int (*mode)(int setting);

    LWCStringUTF8 (*vmap)(int index, LWID* lwid);

    /// @return The layer number of the layer containing the mesh, or 0 if there is no such layer.
    /// The layers are numbered from 1.
    int (*meshLayer)(LWMeshID mesh);

    /// @return The mesh in the layer given by the layer number, for the current object.
    /// The layers are numbered from 1.
    LWMeshID (*layerMesh)(int layernum);

} LWStateQueryFuncs;

#define LWM_MODE_SELECTION 0
#define LWM_MODE_SYMMETRY 1
#define LWM_MODE_ACTION_CENTER 2

#define LWM_VMAP_WEIGHT 0
#define LWM_VMAP_TEXTURE 1
#define LWM_VMAP_MORPH 2

#define LWM_VMAP_PICK 3
#define LWM_VMAP_RGB  4 //Vertex color information, stored as red, green and blue in the range of 0.0 to 1.0.
#define LWM_VMAP_RGBA 5 //Vertex color with transparency, stored as red, green, blue and alpha.


#define LWFONTLISTFUNCS_GLOBAL "LWM: Font List 2"

typedef struct st_LWFontListFuncs
{
    int             (*count) (void);
    int             (*index) (LWCStringUTF8 name);
    LWCStringUTF8   (*name)  (int in_index);
    int             (*load)  (LWCStringUTF8 filename);
    void            (*clear) (int in_index);

} LWFontListFuncs;

#endif