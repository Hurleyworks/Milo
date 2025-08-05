/*
 * LWSDK Header File
 *
 * LWANIMUV.H -- LightWave Animation UV
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_ANIMUV_H
#define LWSDK_ANIMUV_H

#include <lwrender.h>

#define LWANIMUV_HCLASS  "AnimUVHandler"
#define LWANIMUV_ICLASS  "AnimUVInterface"
#define LWANIMUV_GCLASS  "AnimUVGizmo"
#define LWANIMUV_VERSION 6

typedef struct st_LWAnimUVHandler {
    LWInstanceFuncs *inst;
    LWItemFuncs     *item;
    int     (*GetOptions)( LWInstance, char *option_bytes);
    int     (*SetOptions)( LWInstance, char *option_bytes);
    int     (*Begin     )( LWInstance, char *option_bytes, double time, int vertexCount, int wRepeat, int hRepeat, double aspect, int width, int height );
    int     (*Evaluate  )( LWInstance, int vertexIndex, double *uv );
    int     (*End       )( LWInstance );
} LWAnimUVHandler;
#endif
