/*
 * LWSDK Header File
 *
 * LWBASE.H -- LightWave Attributes Base Functions
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_BASE_H
#define LWSDK_BASE_H

#define LWBASEFUNCS_GLOBAL "LW Base Funcs 2"

#include <lwtypes.h>

typedef struct st_LWBase* LWBaseID;
#ifndef LWAttributeID
typedef struct st_lwattribute *LWAttributeID;
#endif

// The base context is used to help identify the owner of server instances
// We presume that each context will treat all servers of particular class as a single array.
typedef LWID LWBaseContextID;
#define LWBASECONTEXT_CURRENT             LWID_('C','R','N','T') // context is NULL (valid during creation of server instance)
#define LWBASECONTEXT_GLOBAL              LWID_('G','L','B','L') // context is NULL
#define LWBASECONTEXT_SCENEITEM           LWID_('S','I','T','M') // context is LWItemID <lwrender.h> (LWITEM_NULL for scene)
#define LWBASECONTEXT_CLIP                LWID_('C','L','I','P') // context is LWImageID <lwimage.h>
#define LWBASECONTEXT_NODE                LWID_('N','O','D','E') // context is NodeID <lwnodes.h>
#define LWBASECONTEXT_CHANNEL             LWID_('C','H','N','L') // context is LWChannelID <lwenvel.h>
#define LWBASECONTEXT_SURFACE             LWID_('S','U','R','F') // context is LWSurfaceID <lwsurf.h>
#define LWBASECONTEXT_VIEWPORT            LWID_('V','P','R','T') // context is view index (unsigned int (0-based))

// A base context consists of a type and the specific context data, which relates to the owner of the base.
typedef struct st_LWBaseContext {
    LWBaseContextID type;
    const void *context;
} LWBaseContext;

// LWBaseID container reference
typedef struct st_LWBaseContainer *LWBaseContainerRef;

// base events
#define LWBASEEVENT_DID_CREATE   LWID_('D','N','E','W') // a server has been created.  data is LWBaseID
#define LWBASEEVENT_WILL_DESTROY LWID_('W','I','L','D') // a server will be destroyed.  data is LWBaseID
#define LWBASEEVENT_DID_DESTROY  LWID_('D','I','D','D') // a server has been destroyed.  data is LWBaseID (but is stale)
#define LWBASEEVENT_DID_UPDATE   LWID_('D','U','P','D') // a server proprietary data has been updated.  data is LWBaseID
#define LWBASEEVENT_DID_ENABLE   LWID_('D','E','N','A') // a server became enabled.  data is LWBaseID
#define LWBASEEVENT_DID_DISABLE  LWID_('D','D','I','S') // a server became disabled.  data is LWBaseID
#define LWBASEEVENT_DID_MOVE     LWID_('D','M','O','V') // a server slot has changed. data is LWBaseID

typedef void (*LWBaseEventFunc)(void *client_data, LWID event, LWBaseID base);

typedef struct st_LWBaseFuncs
{
    /// create a LWBaseID container from those bases (server instances) that are attached to the owner and have the given server class
    /// A NULL class_name will consider all server classes.
    /// The type determines what 'context' represents.
    LWBaseContainerRef (*baseContainerCreate)(LWBaseContextID type, const void *context, LWCStringASCII class_name);

    /// get the number of base items in the container
	unsigned int (*baseContainerCount)(LWBaseContainerRef container);

    /// get the base from an index (0-based) into the given container
    LWBaseID (*baseContainerAtIndex)(LWBaseContainerRef container, unsigned int index);

    /// destroy an existing base container
	void (*baseContainerDestroy)(LWBaseContainerRef container);

    /// server class (return value local scope)
    LWCStringASCII (*getClassName)(LWBaseID base);

    /// server name (return value local scope)
    LWCStringUTF8 (*getServerName)(LWBaseID base);

    /// description (return value local scope)
    LWCStringUTF8 (*getDescription)(LWBaseID base);

    /// context in which the base exists
    LWBaseContext (*getContext)(LWBaseID base);

    /// file system path to plug-in module (return value local scope)
    LWCStringUTF8 (*getModulePath)(LWBaseID base);

    /// Flags LWBASEF_* for the plugin.
    unsigned int (*getFlags)(LWBaseID base);

    /// current number of references to this server
    unsigned int (*getRefCount)(LWBaseID base);

    /// Access to attributes container (return value local scope)
    LWAttributeID (*getAttributes)(LWBaseID base);

    /// subscribe to LWBASEEVENT_ events.  client_data must be unique.  subscribe will fail (return 0), if the client_data is already being referenced.
    int (*subscribe)(void *client_data, LWBaseEventFunc client_func);

    /// unsubscribe will fail (return 0) if the client_data was not found.
    int (*unsubscribe)(void *client_data);

} LWBaseFuncs;

// base flags
#define LWBASEF_DISABLED         (1 << 0)       // server is disabled
#define LWBASEF_HIDDEN           (1 << 1)       // server is hidden from the user
#define LWBASEF_EVALUATORCOPY    (1 << 7)       // server is a copy managed by an evaluator

#endif // LWSDK_BASE_H