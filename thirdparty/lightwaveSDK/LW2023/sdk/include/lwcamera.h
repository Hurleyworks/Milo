/*
 * LWSDK Header File
 *
 * LWCAMERA.H -- LightWave Camera
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_CAMERA_H
#define LWSDK_CAMERA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwhandler.h>
#include <lwrender.h>


#define LWCAMERA_HCLASS "CameraHandler"
#define LWCAMERA_ICLASS "CameraInterface"
#define LWCAMERA_GCLASS "CameraGizmo"
#define LWCAMERA_VERSION 4

typedef struct st_LWCameraAccess
{
    LWItemID cameraID;
    LWDVector worldPos;
    LWDVector toWorld[3];
    LWDVector toCamera[3];
    double filmSize[2];
} LWCameraAccess;

typedef struct st_LWCameraHandler
{
    LWInstanceFuncs* inst;
    LWItemFuncs* item;
    LWRenderFuncs* rend;
    int             (*preview)  (LWInstance, double lpx, double lpy, LWCameraEye eye, LWDMatrix4 projection, const LWCameraAccess* camaccess);
    LWError         (*newFrame) (LWInstance, const LWFrameInfo* frameinfo, const LWCameraAccess* camaccess);
    int             (*evaluate) (LWInstance, double fpx, double fpy, double lpx, double lpy, double fractime, LWCameraEye eye, LWCameraRay* ray, const LWCameraAccess* camaccess);
    unsigned int    (*flags)    (LWInstance);
    int             (*project)  (LWInstance, double lpx, double lpy, LWDVector worldpos, double fractime, LWCameraEye eye, LWDVector fp, const LWCameraAccess* camaccess);
} LWCameraHandler;

// Camera handler evaluation return values
#define LWCAMTYPEEVAL_NO_RAY        0
#define LWCAMTYPEEVAL_RAY           (1 << 0)
#define LWCAMTYPEEVAL_DO_DEF_INTERP (1 << 1)
#define LWCAMTYPEEVAL_RAY_DELTA     (1 << 2)
#define LWCAMTYPEEVAL_STEREO_RAY    (1 << 3)

// Camera handler flags
#define LWCAMTYPEF_SUPPORTSDOF      (1 << 0)    // Camera supports depth of field
#define LWCAMTYPEF_SUPPORTSSTEREO   (1 << 1)    // No longer used
#define LWCAMTYPEF_NOPROJECT        (1 << 2)    // Camera does not support projection (even though project method is implemented)

#ifdef __cplusplus
}
#endif

#endif