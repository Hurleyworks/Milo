/*
 * LWSDK Header File
 *
 * LWCOMRING.H -- LightWave Communications Ring
 *
 * This header contains declarations necessary to engage in a
 * communications ring among plug-ins
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_COMRING_H
#define LWSDK_COMRING_H

#include <lwtypes.h>

#define LWCOMRING_GLOBAL "LW Communication Ring"

typedef void    (*RingEvent)    (void *clientData,void *portData,int eventCode,void *eventData);


typedef struct st_LWComRing
{
    int         (*ringAttach)   (LWCStringASCII topic,LWInstance pidata,RingEvent eventCallback);
    void        (*ringDetach)   (LWCStringASCII topic,LWInstance pidata);
    void        (*ringMessage)  (LWCStringASCII topic,int eventCode,void *eventData);
} LWComRing;

#define LW_PLUGIN_LIMBO_STATES "LW Plugin Limbo States"

#define LW_LIMBO_START 0x00000001
#define LW_LIMBO_END   0x00000002

/* Color space events. */

#define LW_PORT_COLORSPACE "color-space-change"

enum {
    LWCSEV_CHANGE = 0,
    LWCSEV_SIZEOF
};

/**
 *      For applications with multiple non-modal windows, it is often useful for
 *      the host and the client windows to be able to broadcast state information
 *      pertinent to those windows.
 *
 *      If the window data is NULL, then open or close your window.
 *      If the window data is not NULL, then the data is the name of the window that was opened or closed.
 */

#define LW_PORT_WINSTATE   "window-state"

enum {
    LWWSEV_CLOSE = 0,   /* Close window. */
    LWWSEV_OPEN,        /* Open  window. */
    LWWSEV_SIZEOF
};


/**
*      During certain times, such as rendering, disabling the contents of a window is required to prevent
*      changes to the scene/data.
*
*      The event data is NULL, LWWEEV_ENABLE/LWWEEV_DISABLE is sent to request the window enable/disable input.
*/

#define LW_PORT_WINENABLE   "window-enable"

enum {
    LWWEEV_DISABLE = 0, /* Disable window. */
    LWWEEV_ENABLE,      /* Enable window. */
    LWWEEV_SIZEOF
};


/**     Master Events
 *      For applications that need access to master events without needing the overhead of a full Master class
 *      plugin. The data received from the port is the same as the data provided by the event() callback
 *      for Master class plugins, as defined in lwmaster.h
 */

#define LW_PORT_MASTER_EVENT   "master-event"

/**     Command Port
 *      Messages arriving on the application's command port will be forwarded to this ring
 *      topic if they have a meta-data prefix that starts with the character '{' (it is assumed
 *      this prefix will be terminated with '}', but that is not enforced at the application level).
 */

#define LW_PORT_COMMAND_PORT   "command-port"

#endif