/*
 * LWSDK Header File
 *
 * LWCOMMANDPORT.H -- LightWave Command Port
 *
 * This header contains declarations for retrieving information
 * about the state of the application's Command Port
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_COMMANDPORT_H
#define LWSDK_COMMANDPORT_H

#include <lwtypes.h>

#define LWCOMMANDPORT_GLOBAL "LW Command Port"

typedef struct st_LWCommandPort
{
    int         (*currentPort)();       /* 0 = disabled, >0 = port in use */
    int         (*enable)(int port);    /* port range 1025 - 65535*/
    int         (*disable)();
} LWCommandPort;

#endif