/*
 * LWSDK Header File
 *
 * LWCOMPORT.H -- LightWave Communications Port
 *
 * This header contains declarations necessary to submit commands
 * to running LightWave instances.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_COMPORT_H
#define LWSDK_COMPORT_H

#include <lwtypes.h>        // for LWID_()

// Command Port events will be broadcast on this ComRing notification channel

#define CP_CHANNEL "command-port-event"

// Because only one port can be in use at a time within the same
// domain, and because we need each running instance to respond
// to a Command Port discovery request, we declare a range of
// ports to be used for listening within the same domain.
//
// These values represent the default range.

#define CP_DISCOVERY_START      50155
#define CP_DISCOVERY_END        50165

typedef enum {
    CP_COMMANDSET_LAYOUT = 1,
    CP_COMMANDSET_MODELER,
} CpCommandSet;

#define CP_REQ_MAGIC  LWID_('C', 'R', 'E', 'Q')
#define COMMANDPORT_REQ_VERSION    1

typedef struct _CommandPortReq
{
    // magic value (uniquely identifies the packet type)
    int                 magic;
    // version of structure
    int                 version;
    // response port
    unsigned short      port, dummy;
} CommandPortReq;

#define CP_INFO_MAGIC LWID_('C', 'N', 'F', 'O')
// v2 adds the build value
#define COMMANDPORT_INFO_VERSION    2

typedef struct _CommandPortInfo
{
    // magic value (uniquely identifies the packet type)
    int                 magic;
    // version of structure
    int                 version;
    // version of app
    unsigned short      major, minor, build;
    // 1 = Layout, 2 = Modeler: indicates command set accepted
    unsigned short      app;
    // Command Port port
    unsigned short      port;
    // Command Port alias (e.g., 'Ariel') (ASCII)
    char                port_alias[128];
} CommandPortInfo;

#endif