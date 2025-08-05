/**
 *      @brief lwpreferences.h
 *
 *      @short Contains the preferences global.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 *
 */

#ifndef LWSDK_PREFERENCES_H
#define LWSDK_PREFERENCES_H

#include <lwtypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LWPREFERENCESS_GLOBAL "Preferences Global"

/* LightWave change tracker options. */

typedef enum lwen_changeoptions {
    lwc_Disabled = 0,
    lwc_Simple,
    lwc_Advanced,
    lwc_Sizeof
    } LWCHANGEOPTIONS;


typedef struct st_LWPreferences {
    int     (*defaultStartFrame   )( void );    /*!< Default start frame.        */
    int     (*defaultKeyFrame     )( void );    /*!< Default keyframe.           */
    int     (*initKeyFramePerScene)( void );    /*!< Default keyframe per scene. */
    int     (*defaultSceneLength  )( void );    /*!< Default scene length.       */
    int     (*saveOptions         )( void );    /*!< Change tracker options.     */
    int     (*autoSaveObjectTime  )( void );    /*!< Auto save object.           */
    int     (*autoSaveSceneTime   )( void );    /*!< Auto save scene.            */
} LWPreferences;

#ifdef __cplusplus
}
#endif

#endif