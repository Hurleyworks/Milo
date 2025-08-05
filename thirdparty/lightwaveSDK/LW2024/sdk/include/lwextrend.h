/*
 * LWSDK Header File
 *
 * LWEXTRENDERER.H -- LightWave Renderer
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_EXTRENDERER_H
#define LWSDK_EXTRENDERER_H

#include <lwtypes.h>
#include <lwserver.h>
#include <lwgeneric.h>
#include <lwhandler.h>
#include <lwrender.h>

#define LWEXTRENDERER_HCLASS  "ExtRendererHandler"
#define LWEXTRENDERER_VERSION 3

typedef int EXTRENDERERIMAGE( void *user, int frame, int eye, LWPixmapID displayimage, LWCStringUTF8 name, LWCStringUTF8 buffer );

typedef struct st_LWExtRendererHandler {
    LWInstanceFuncs *inst;
    LWItemFuncs     *item;
    int            (*options)( LWInstance );
    int            (*render )( LWInstance, int first_frame, int last_frame, int frame_step, EXTRENDERERIMAGE *render_image, void *user, int render_mode );
} LWExtRendererHandler;

#endif