/*
 * LWSDK Header File
 *
 * LWSCENECV.H -- LightWave Scene Converters
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_SCENECV_H
#define LWSDK_SCENECV_H

#include <lwtypes.h>
#include <lwhandler.h>

#define LWSCENECONVERTER_CLASS      "SceneConverter"
#define LWSCENECONVERTER_VERSION    2


typedef struct st_LWSceneConverter {
    LWCStringUTF8 filename;
    LWError       readFailure;
    LWCStringUTF8 tmpScene;
    void        (*deleteTmp) (LWCStringUTF8 tmpScene);
} LWSceneConverter;


#endif
