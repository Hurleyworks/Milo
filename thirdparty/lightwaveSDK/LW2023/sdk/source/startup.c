/*
 * LWSDK Library Source File
 *
 * Default 'Startup' function returns any non-zero value for success.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#include <lwserver.h>

void *Startup( void); // prototype

void *Startup (void)
{
    return (void *) 4;
}