/*
 * LWSDK Header File
 *
 * LWGENERIC.H -- LightWave Generic Commands
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_GENERIC_H
#define LWSDK_GENERIC_H

#include <lwtypes.h>
#include <lwdyna.h>

#define LWLAYOUTGENERIC_CLASS   "LayoutGeneric"
#define LWLAYOUTGENERIC_VERSION 7

typedef struct st_LWLayoutGeneric {
    int             (*saveScene) (LWCStringUTF8 file);
    int             (*loadScene) (LWCStringUTF8 file, LWCStringUTF8 name);

    void             *data;
    LWCommandCode   (*lookup)    (void *, LWCStringUTF8 cmdName);
    int             (*execute)   (void *, LWCommandCode cmd, int argc, const DynaValue *argv, DynaValue *result);
    int             (*evaluate)  (void *, LWCStringUTF8 command);
    LWCStringUTF8   (*commandArguments)(void *);
    int             (*parsedArguments)  (void *, DynaValue **argv);

} LWLayoutGeneric;

#endif
