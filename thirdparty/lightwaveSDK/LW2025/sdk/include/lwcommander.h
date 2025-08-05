/*
 * LWSDK Header File
 *
 * LWCONSOLE.H -- LightWave Python Commander Access
 *
 *Copyright © 2025 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_PCORECOMMANDER_H
#define LWSDK_PCORECOMMANDER_H

#include <lwtypes.h>

#define LWPCORECOMMANDER_GLOBAL "LW PCore Commander"

typedef struct st_LWPCoreCommander
{
    void (*show)();
    void (*hide)();
    int (*visible)();
} LWPCoreCommander;

#endif