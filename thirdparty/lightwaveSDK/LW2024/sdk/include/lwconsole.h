/*
 * LWSDK Header File
 *
 * LWCONSOLE.H -- LightWave Console Access
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_PCORECONSOLE_H
#define LWSDK_PCORECONSOLE_H

#include <lwtypes.h>

#define LWPCORECONSOLE_GLOBAL    "LW PCore Console"

typedef struct st_LWPCoreConsole
{
    void        (*info)(LWCStringUTF8 message);
    void        (*error)(LWCStringUTF8 message);
    void        (*clear)();
    void        (*show)();
    void        (*hide)();
    int         (*visible)();
} LWPCoreConsole;

#endif