/*
 * LWSDK Header File
 *
 * LWGLOBSERV.H -- LightWave Global Server
 *
 * This header contains declarations necessary to define a "Global"
 * class server.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_GLOBSERV_H
#define LWSDK_GLOBSERV_H

#include <lwtypes.h>

#define LWGLOBALSERVICE_CLASS   "Global"
#define LWGLOBALSERVICE_VERSION 1

typedef struct st_LWGlobalService {
    LWCStringASCII  id;
    void           *data;
} LWGlobalService;

#endif
