/*
 * LWSDK Header File
 *
 * LWVIEWPORTOBJECT.H -- LightWave Viewport Object
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_VIEWPORTOBJECT_H
#define LWSDK_VIEWPORTOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwtypes.h>
#include <lwserver.h>
#include <lwgeneric.h>
#include <lwhandler.h>
#include <lwrender.h>
#include <lwmonitor.h>

/* Viewport object window handler. */

#define LWVIEWPORTOBJECT_HCLASS  "ViewportObjectHandler"
#define LWVIEWPORTOBJECT_ICLASS  "ViewportObjectInterface"
#define LWVIEWPORTOBJECT_VERSION 6

/* Rendering Flags. */

typedef enum en_lwviewflags {
    LWVF_a = 0,                 /*!< Buffer has Alpha in it.  */
    LWVF_z,                     /*!< Z Buffer is valid.       */
    LWVF_motion_blur,           /*!< Renders motion blur.     */
    LWVF_sizeof
} LWViewFlags;

#define LWHA_eye_to_render_mask 0x03

typedef enum en_hostsccessflags {
    LWHA_eye_to_render  = 0,    /*!< LWCameraEye.             */
    LWHA_frame_complete = 2,    /*!< The frame is complete or has been aborted. */
    LWHA_frame_refresh,         /*!< Refresh the viewport during the next pass through the main loop. */
    LWHA_smoothing,             /*!< Smoothing shading res-up.*/
    LWHA_sizeof
} LWHostAccessFlags;

/* Surface Preview Background .*/

typedef enum lwen_background {
    LWBG_layout = 0,
    LWBG_checkerBoard,
    LWBG_color
} LW_BACKGROUND;

struct st_LWVPAHostAccess;
typedef struct st_LWViewportObjectAccess {
    int          (*showFrame   )( struct st_LWVPAHostAccess *access, unsigned int width, unsigned int height, float *rgb_buffer, unsigned int modulus_rgb, float *z_buffer, unsigned int modulus_z, unsigned int scale, unsigned int flags, float *minimum, float *maximum );
} LWViewportObjectAccess;

struct LWInputHandling;
typedef struct st_LWViewportHandler {
    LWInstanceFuncs *inst;
    LWItemFuncs     *item;
    int          (*newTime     )( LWInstance, double );                             /*!< Time.                                     */
    int          (*newSize     )( LWInstance, unsigned int, unsigned int );         /*!< Width and Height.                         */
    int          (*redraw      )( LWInstance );                                     /*!< Redraws the window.                       */
    int          (*setViewIndex)( LWInstance, int );                                /*!< View 0 to n.                              */
    int          (*setViewMode )( LWInstance, unsigned int );                       /*!< LVVIEWT.                                  */
    int          (*setAccess   )( LWInstance, LWViewportObjectAccess *, struct st_LWVPAHostAccess * );
    unsigned int (*flags       )( LWInstance );                                     /*!< LWViewFlags.                              */
    int          (*start       )( LWInstance, unsigned int );                       /*!< Start rendering.                          */
    int          (*stop        )( LWInstance );                                     /*!< Done, free memory.                        */
    LWError      (*selected    )( LWInstance, int );                                /*!< Currently selected or unselected.         */
    int          (*preview     )( LWInstance, int );                                /*!< Making preview.                           */
    void         (*inputHandling)( LWInstance, struct LWInputHandling *handler );   /*!< Returns the input handler                 */

    /* Layout pull down menu. */

    int          (*menuCount   )( LWInstance );                                     /*!< Count of entries in the menu.             */
    LWCStringUTF8 (*menuName   )( LWInstance, unsigned int );                       /*!< Name  of menu entry.                      */
    int          (*menuGetPick )( LWInstance, unsigned int );                       /*!< Menu set entry selected.                  */
    int          (*menuSetPick )( LWInstance, unsigned int, int );                  /*!< Menu get entry selected.                  */

    /* Shader preview. */

    int          (*previewBackground)( LWInstance, LW_BACKGROUND backgound, float *color ); /*!< Sets up the background color mode.*/
    int          (*previewObject)( LWInstance, LWItemID );                          /*!< Which object  is the preview object.      */
    int          (*previewLight)(  LWInstance, LWItemID );                          /*!< Which light   is the preview light.       */
    int          (*previewSurface)(LWInstance, LWSurfaceID );                       /*!< Which surface is the preview surface.     */
    int          (*previewMode)(   LWInstance, int );                               /*!< Enables / Disabled preview mode.          */
} LWViewportHandler;

/* Viewport object camera global. */

#define LWVIEWPORTCAMERA "LW Viewport Camera 2"

typedef enum {
    LWVPOC_Disabled             = 0,
    LWVPOC_Enabled_Orthographic = 1,
    LWVPOC_Enabled_Perspective  = 2
} LWViewportObjectCameraEnabled;

typedef struct st_LWViewportObjectCamera {
    double         filmSize;                                                        /*!< Film size for viewport.                   */
    double         focalLength;                                                     /*!< Focal length of viewport.                 */
    double         aspect;                                                          /*!< Pixel aspect ratio of viewport.           */
    double         zNear;                                                           /*!< z Near clipping plane.                    */
    double         zFar;                                                            /*!< z Far  clipping plane.                    */
    LWDMatrix4     modelMatrix;                                                     /*!< OpenGL model view matrix for viewport.    */
    LWDMatrix4     modelMatrixInverted;                                             /*!< Inverse of the model view matrix.         */
    LWDMatrix4     projectionMatrix;                                                /*!< OpenGL projection matrix for viewport.    */
    LWDMatrix4     projectionMatrixInverted;                                        /*!< Inverse of the projection matrix.         */
    LWDMatrix4     viewMatrix;                                                      /*!< Combined model and projection view matrix.*/
    LWDMatrix4     viewMatrixInverted;                                              /*!< Inverse of the view matrix.               */
    float          colorBackground[ 4 ];                                            /*!< Back ground clear color.                  */
    int          (*getRay)( struct st_LWViewportObjectCamera *, double, double, double *, double * ); /*!< Ray tracing position and direction. */
    int            enabled;                                                         /*!< LWViewportObjectCameraEnabled.            */
    LWDVector      origin;                                                          /*!< Camera origin in world coordinates.       */
    int          (*getOrigin)( struct st_LWViewportObjectCamera *, double * );      /*!< Camera world space origin.                */
    double         overscanFactors[ 2 ];                                            /*!< Width and height overscan scaling factors */
} LWViewportObjectCamera;

/* LightWave function global proto type.    */
/* The global returns this function pointer.*/

typedef LWViewportObjectCamera *LWViewportCamera(unsigned int view_index, unsigned int view_mode, int eye_to_render);

/* Viewport object menu global. */

/* Menu Flags. */

typedef enum en_lwviewport_flags {
    LWVPF_openglWireframe = 0,                              /*!< Displays wireframe over viewport object. */
    LWVPF_openglOverlay,                                    /*!< Displays OpenGL overlays.                */
    LWVPF_sizeof
} LWViewportFlags;

#define LWVIEWPORTMENU "LW Viewport Menu"

typedef struct st_LWViewportObjectMenu {
    unsigned int (*getFlags)( unsigned int );               /*!< View 0 to n.                             */
    int          (*setFlags)( unsigned int, unsigned int ); /*!< View 0 to n, LWViewportFlags.            */
} LWViewportObjectMenu;

/* Viewport object monitor. */

#define LWVIEWPORTOBJECTMONITORFUNCS_GLOBAL "Viewport Object Monitor"

typedef struct st_LWViewportObjectMonitorFuncs {
    LWMonitor *(*create) ( LWCStringUTF8 title, int viewport );
    void       (*destroy)( LWMonitor * );
} LWViewportObjectMonitorFuncs;

#ifdef __cplusplus
}
#endif

#endif