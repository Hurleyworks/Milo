/*
 * LWSDK Header File
 *
 * LWNODES.H -- LightWave Shader Nodes.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_SURFNODES_H
#define LWSDK_SURFNODES_H

#include <lwtypes.h>
#include <lwserver.h>
#include <lwrender.h>
#include <lwtxtr.h>
#include <lwinstancing.h>
#include <lwbase.h>
#include <lwaovs.h>

#define LWNODE_HCLASS           "NodeHandler"
#define LWNODE_ICLASS           "NodeInterface"
#define LWNODE_GCLASS           "NodeGizmo"
#define LWNODE_ACLASS           "NodeAttribute"
#define LWNODECLASS_VERSION     5

#define LWNODEFUNCS_GLOBAL          "NodeFuncs 3"
#define LWNODEINPUTFUNCS_GLOBAL     "NodeInputFuncs 5"
#define LWNODEOUTPUTFUNCS_GLOBAL    "NodeOutputFuncs 4"
#define LWNODEUTILITYFUNCS_GLOBAL   "NodeUtilityFuncs 3"
#define LWNODEDRAWFUNCS_GLOBAL      "NodeDrawFuncs 2"

// Predefined server strings for SRVTAG_NODECONTEXT server tag.
#define NCON_ALL  "All"      // Node is available in all contexts.
#define NCON_MESH "Mesh"     // Node is available in contexts where a physical mesh is provided.
#define NCON_SPOT "Spot"     // Node is available in contexts where virtual spot geometry is provided.
#define NCON_SURF "Surface"  // Node is available in contexts where virtual or real surface geometry is provided. A surface requires proper normals.
#define NCON_RAYT "Raytrace" // Node is available in contexts where ray-tracing functions are provided.

// The type of the node preview for the nodes editor.
typedef enum {NPT_OFF = 0, NPT_CUSTOM, NPT_RENDER} NodePreviewType;

// Node input/output types.
typedef enum {
    NOT_RGB = 1,        // LWDVector
    NOT_SCALAR,         // double
    NOT_VECTOR,         // LWDVector
    NOT_INTEGER,        // int
    NOT_FUNCTION,       // double
    NOT_PROJECTION,     // LWNodalProjection
    NOT_MATRIX44,       // LWDMatrix4
    NOT_BSDF,           // LWBSDF
    NOT_FRESNEL,        // LWNodalFresnelFunction
    NOT_VOLUME,         // LWNodalVolumeFunction
    NOT_CUSTOM = 1024
} ConnectionType;

typedef void* NodeInputID;
typedef void* NodeOutputID;
typedef void* NodeID;
typedef void* NodeData;
typedef void* NodeValue;

typedef enum {LWPROJECTION_LOCAL = 0, LWPROJECTION_WORLD} LWProjectionSpace;

typedef struct LWNodalProjection_t
{
    LWProjectionSpace   space;

    LWDMatrix4          matrix;

    LWDVector           P;
    LWDVector           dPdx;
    LWDVector           dPdy;
    LWDVector           dPdu;
    LWDVector           dPdv;

    double               u;
    double               v;
    double               dudx;
    double               dvdx;
    double               dudy;
    double               dvdy;
} LWNodalProjection;


// Fresnel function type
typedef void LWNodalFresnelFunc(LWMemChunk memory, double cosTheta, LWDVector fresnel);

typedef struct LWNodalFresnelFunction_t
{
    LWMemChunk          memory;
    LWNodalFresnelFunc* F;
} LWNodalFresnelFunction;


// Volume function type
typedef struct LWNodalVolumeFunction_t
{
    LWMemChunk          memory;
    LWBTDFVolume        volume;
    LWBTDFVolumeType    type;
    double              stepSize;
} LWNodalVolumeFunction;


// Draw Functions.
// Draws on the preview area of the node.
typedef struct LWNodeDrawFuncs_t {
    void   (*drawPixel)(NodeID node, int c, int x, int y);
    void   (*drawRGBPixel)(NodeID node, int r, int g, int b, int x, int y);
    void   (*drawLine)(NodeID node, int c, int x, int y, int x2, int y2);
    void   (*drawBox)(NodeID node, int c, int x, int y, int w, int h);
    void   (*drawRGBBox)(NodeID node, int r, int g, int b, int x, int y, int w, int h);
    int    (*textWidth)( NodeID node, LWCStringUTF8 s );
    int    (*textHeight)( NodeID node, LWCStringUTF8 s );
    void   (*drawText)( NodeID node, LWCStringUTF8 s, int c, int x, int y );
    void   (*blitNode)(NodeID node);
} LWNodeDrawFuncs;

// Input event types.
typedef enum LWNodalEvent_t {
    NIE_CONNECT = 0,        // When an output is connected to this input.
    NIE_DISCONNECT,         // When the output was disconnected from this input.
    NIE_INPUTNODEDESTROY    // When the node connected to this input was destroyed.
} LWNodalEvent;

typedef int NodeInputEvent(void *userData, NodeInputID, LWNodalEvent, ConnectionType);
// Receives the user data, the input the function belongs to, the LWNodalEvent,
//and the type of the output connected to the input this function was called for, if the event was NIE_CONNECT,
// otherwise ConnectionType is 0.

// Node flags. Returned by LWNodeHandler->flags
#define NF_TRANSP           (1 << 0)   // This flag should be set if a material node might have transparency
#define NF_SINGLE_THREADED  (1 << 30)  // Force single-threaded evaluation (for nodal displacement only)

typedef struct LWNodeOGLTextureOutput_t
{
    LWImageID   imageID;
    double      uv_coords[2];
    LWDVector   val;
} LWNodeOGLTextureOutput;

typedef struct LWNodeOGLMaterial_t
{
    LWDVector luminosity_color;
    LWDVector diffuse_color;
    LWDVector specular_color;
    double    specularExponent;
    double    transparency;
    double    diffuse_intensity;
    double    luminosity_intensity;
} LWNodeOGLMaterial;

// Node handler activation.
typedef struct st_LWNodeHandler {
    LWInstanceFuncs *inst;
    LWItemFuncs     *item;
    LWRenderFuncs   *rend;

    // Evaluation function receives the LWShadingGeometry structure.
    // NodeOutputID is the output belonging to this node, and which is being currently asked the value from.
    // NodeValue is the value you need to set with the output functions when setting a value for this evaluation.
    void            (*evaluate)(LWInstance, LWShadingGeometry*, NodeOutputID, NodeValue);

    // customPreview is called when the node has NPT_CUSTOM preview type set.
    void            (*customPreview)(LWInstance, int width, int height);

    unsigned int    (*flags)(LWInstance);

    // OpenGL rendering functions.
    unsigned int    (*materialGL)(LWInstance, LWNodeOGLMaterial*);
} LWNodeHandler;

// Functions for node inputs.
typedef struct  LWNodeInputFuncs_t {
    NodeInputID     (*create)( NodeID, ConnectionType, LWCStringASCII name, NodeInputEvent* ); // Create a new input for the node.
    void            (*destroy)(NodeInputID); // Destroy an input.

    int             (*evaluate)(NodeInputID, LWShadingGeometry*, void* value);
    // Evaluate an input.
    // The value filled with a value received from the node evaluated,
    // according to the type of the input evaluated.

    int             (*check)(NodeInputID); // Check if this input is connected to.
    NodeInputID     (*first)(NodeID); // Get the first input from a node.
    NodeInputID     (*next)(NodeInputID); // Get the next input.
    NodeInputID     (*previous)(NodeInputID); // Get the previous input.
    int             (*numInputs)(NodeID); // Get the number of inputs for this node.
    NodeInputID     (*byIndex)(NodeID, int);
    // Get an input by it's index number.
    // The index number of the first input is 1.

    int             (*getIndex)(NodeInputID);
    // Get the index number of the input.
    // The index number of the first input is 1.

    void            (*disconnect)(NodeInputID); // Disconnect any output from this input.
    NodeID          (*node)(NodeInputID); // Returns the node this input belongs to.
    NodeOutputID    (*connectedOutput)(NodeInputID); // Returns the NodeOutputID this input is connected to.
    LWCStringASCII  (*name)( NodeInputID ); // Returns the name of the NodeInputID.
    void            (*rename)( NodeInputID, LWCStringASCII name ); // Rename a NodeInputID.

    NodeInputID     (*createCustom)(NodeID, ConnectionType, LWCStringASCII name, NodeInputEvent*, LWID, LWID); // Create custom connection type.

    ConnectionType  (*type)(NodeInputID); // Returns the type of the input.
    LWID            (*vendorID)(NodeInputID); // Returns the vendorID for a custom input.
    LWID            (*typeID)(NodeInputID); // Returns the typeID for a custom input.
} LWNodeInputFuncs;

typedef struct LWNodeOutputFuncs_t {
    NodeOutputID    (*create)( NodeID, ConnectionType, LWCStringASCII name ); // Create a new output for the node.
    void            (*destroy)(NodeOutputID); // Destroy an output from the node.

    void            (*setValue)(NodeValue, const void*);
    // Set the value for the output being evaluated.
    // Call from the node evaluation function when rendering.

    void            *(*getValue)(NodeValue); // Get the pointer to the value cast to the input functions evaluate call.
    ConnectionType  (*getType)(NodeValue); // Get the type of the connection the value is coming from.

    NodeOutputID    (*first)(NodeID); // Get the first output from a node.
    NodeOutputID    (*next)(NodeOutputID); // Get the next output.
    NodeOutputID    (*previous)(NodeOutputID); // Get the previous output.
    int             (*numInputs)(NodeID); // Get the number of outputs for this node.

    NodeOutputID    (*byIndex)(NodeID, int);
    // Get an output by it's index number.
    // The index number of the first output is 1.

    int             (*getIndex)(NodeOutputID);
    // Get the index number of the output.
    // The index number of the first output is 1.

    NodeID          (*node)(NodeOutputID);
    // Returns the node this output belongs to.

    LWCStringASCII  (*name)( NodeOutputID ); // Returns the name of the NodeOutputID.
    void            (*rename)( NodeOutputID, LWCStringASCII name ); // Rename a NodeOutputID.

    NodeOutputID    (*createCustom)(NodeID, ConnectionType, LWCStringASCII name, LWID, LWID); // Create a custom connection.

    ConnectionType  (*type)(NodeOutputID); // Returns the type of the output.
    LWID            (*vendorID)(NodeOutputID); // Returns the vendorID for a custom output.
    LWID            (*typeID)(NodeOutputID); // Returns the typeID for a custom output.

    int             (*evaluate)(NodeOutputID, LWShadingGeometry*, NodeValue); // Evaluate an output.
} LWNodeOutputFuncs;

typedef enum {NMI_NODE = 1, NMI_INPUT, NMI_OUTPUT} NodeMenuInputType;

typedef struct LWNodeMenuFuncs_t
{
    void*           userdata;
    int           (*countFn)(NodeMenuInputType, void* inputdata, void* userdata);
    LWCStringUTF8 (*nameFn)(NodeMenuInputType, void* inputdata, void* userdata, int item);
    void          (*eventFn)(NodeMenuInputType, void* inputdata, void* userdata, int item);
} LWNodeMenuFuncs;

typedef struct LWNodeFuncs_t {
    LWCStringUTF8  (*nodeName)( NodeID );
    // Get the name for the node in the editor.
    // Will be the name of the node, with it's index number added to it. Ie. "Texture (1)", etc.

    LWCStringUTF8   (*serverUserName)( NodeID ); // Get the server name for this node.
    LWChanGroupID   (*chanGrp)(NodeID); // Get the channel group for this node.
    void            (*setNodeColor)(NodeID, int[3]); // Set the color for this node.
    void            (*setNodeColor3)(NodeID, int r, int g, int b); // Set the color for this node. using separated R, G and B values.
    void            (*setNodePreviewType)(NodeID, NodePreviewType); // Set the preview type for this node.

    void            (*UpdateNodePreview)(NodeID);
    // Do an immediate interface update for this node.
    // Draws the preview only for this node.

    void            (*SetNodeMenuFuncs)(NodeID, LWNodeMenuFuncs*);
    LWBaseID        (*getBaseID)(NodeID);
} LWNodeFuncs;

// Blending modes.
typedef enum {
    Blend_Normal, Blend_Additive, Blend_Subtractive, Blend_Multiply,
    Blend_Screen, Blend_Darken, Blend_Lighten, Blend_Difference,
    Blend_Negative, Blend_ColorDodge, Blend_ColorBurn, Blend_Red,
    Blend_Green, Blend_Blue
} BlendingMode;

// Node utility functions.
// To help creating new nodes easier, with consistent color blending modes.
typedef struct LWNodeUtilityFuncs_t {
    void    (*Blend)(LWDVector result, LWDVector bg, LWDVector fg, double alpha, BlendingMode); // Function to use the built-in blending modes.

    int     (*NodeAutosize)(NodeID, LWDVector scale, LWDVector position); // Scale and position vectors will be filled with the automatic values.

    void    (*referenceItemMatrix)(LWTime t, LWItemID item, LWDMatrix4 m); // Build an item matrix.

    double  (*cameraProjectionMatrix)(LWTime t, LWItemID camera, LWDMatrix4 m); // Build a camera matrix for front projection mapping. Returns the frame aspect.

    void    (*planarMapping)(const LWDVector pos, const LWDVector rot, const LWDVector scl,
                             int axis, int world, double utiles, double vtiles, double uoffset, double voffset,
                             const LWDMatrix4 refMatrix, const LWShadingGeometry* na, LWNodalProjection* proj);


    void    (*sphericalMapping)(const LWDVector pos, const LWDVector rot,
                                int axis, int world, double utiles, double vtiles, double uoffset, double voffset,
                                const LWDMatrix4 refMatrix, const LWShadingGeometry* na, LWNodalProjection* proj);

    void    (*cylindricalMapping)(const LWDVector pos, const LWDVector rot, const LWDVector scl,
                                  int axis, int world, double utiles, double vtiles, double uoffset, double voffset,
                                  const LWDMatrix4 refMatrix, const LWShadingGeometry* na, LWNodalProjection* proj);

    void    (*cubicMapping)(const LWDVector pos, const LWDVector rot, const LWDVector scl,
                            int world, double utiles, double vtiles, double uoffset, double voffset,
                            const LWDMatrix4 refMatrix, const LWShadingGeometry* na, LWNodalProjection* proj);

    void    (*frontMapping)(double aspect, const LWDMatrix4 camMAtrix, const LWShadingGeometry* na, LWNodalProjection* proj);

    int     (*uvMapping)(LWCStringUTF8 uvMapName, double utiles, double vtiles, double uoffset, double voffset,
                         LWShadingGeometry* na, LWNodalProjection* proj); // Returns true if successful.

} LWNodeUtilityFuncs;

#endif