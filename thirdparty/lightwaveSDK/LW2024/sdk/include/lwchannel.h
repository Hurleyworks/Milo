/*
 * LWSDK Header File
 *
 * LWCHANNEL.H -- LightWave Channel Filters
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_CHANNEL_H
#define LWSDK_CHANNEL_H

#include <lwtypes.h>
#include <lwrender.h>

#define LWCHANNEL_HCLASS    "ChannelHandler"
#define LWCHANNEL_ICLASS    "ChannelInterface"
#define LWCHANNEL_GCLASS    "ChannelGizmo"
#define LWCHANNEL_VERSION   5


typedef struct st_LWChannelAccessPriv LWChannelAccessPriv;

typedef struct st_LWChannelAccess {
    LWChannelAccessPriv *priv;
    LWChannelID    channel;
    LWFrame        frame;
    LWTime         time;
    double         value;
    void          (*getChannel)  (LWChannelAccessPriv *priv, LWChannelID chan, LWTime t, double *value);
    void          (*setChannel)  (LWChannelAccessPriv *priv, LWChannelID chan, double value);
    LWCStringUTF8 (*channelName) (LWChannelAccessPriv *priv, LWChannelID chan);
} LWChannelAccess;

typedef struct st_LWChannelHandler {
    LWInstanceFuncs  *inst;
    LWItemFuncs      *item;
    void            (*evaluate) (LWInstance, const LWChannelAccess *);
    unsigned int    (*flags)    (LWInstance);
} LWChannelHandler;

#endif