/*
 * LWSDK Header File
 *
 * LWDIALOG.H -- LightWave Standard Dialogs
 *
 * LightWave makes some of its more common requests from the user using
 * standard dialogs.  These dialogs (or "requesters") are used for getting
 * files and paths for saving and loading, and for getting color choices.
 * By default, the standard system dialogs are used, but these can be
 * overridden by plug-ins of the right class.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_DIALOG_H
#define LWSDK_DIALOG_H

#include <lwtypes.h>

/*
 * File dialogs can be configured by servers of this class.
 */
#define LWFILEREQ_CLASS     "FileRequester"
#define LWFILEREQ_VERSION   4

typedef struct st_LWFileReqLocal {
    int         reqType;
    int         result;
    LWCStringUTF8 title;
    LWCStringUTF8 fileType;
    LWMutableCStringUTF8 path;
    LWMutableCStringUTF8 baseName;
    LWMutableCStringUTF8 fullName;
    int         bufLen;
    int       (*pickName)( void );
} LWFileReqLocal;

#define FREQ_LOAD       1
#define FREQ_SAVE       2
#define FREQ_DIRECTORY  3
#define FREQ_MULTILOAD  4


/*
 * Color dialogs can be configured by servers of this class.
 */
#define LWCOLORPICK_CLASS   "ColorPicker"
#define LWCOLORPICK_VERSION 6           /* This version means your need to supply your own color correction. */

typedef void LWHotColorFunc( void *data, float r, float g, float b );

typedef struct st_LWColorPickLocal {
    int             result;
    LWCStringUTF8   title;
    float           red, green, blue;
    void           *data;
    LWHotColorFunc *hotFunc;
    LWCStringUTF8   colorSpace;
} LWColorPickLocal;

#endif