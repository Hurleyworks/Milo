/*
 * LWSDK Header File
 *
 * LWHANDLER.H -- LightWave Instance Handlers
 *
 * This header contains the definition of a basic handler.  A handler
 * is a special kind of server that manages a persistent instance.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_HANDLER_H
#define LWSDK_HANDLER_H

#include <lwbase.h>
#include <lwio.h>
#include <lwxpanel.h>


/* The server base context is passed to the create function for certain handlers.
 * Refer to each handler to better understand what it receives in 'context'.
 * This structure may be passed into 'context' for some create routines (depends on handler).
 */
typedef struct st_LWContext {
    LWBaseContext context;
    LWBaseID baseID;
} LWContext;

/*
 * Handlers of all types have a class name of the form "*Handler", and
 * take a handler struct as the local data to the Activate function.
 * The Activate function fills in the handler fields and returns.
 */
typedef struct st_LWInstanceFuncs {
    void           *priv;
    LWInstance    (*create)  (void *priv, void *context, LWError *);
    void          (*destroy) (LWInstance);
    LWError       (*copy)    (LWInstance, LWInstance from);
    LWError       (*load)    (LWInstance, const LWLoadState *);
    LWError       (*save)    (LWInstance, const LWSaveState *);
    LWCStringUTF8 (*descln)  (LWInstance);
} LWInstanceFuncs;

typedef struct st_LWHandler {
    LWInstanceFuncs  *inst;
} LWHandler;


/*
 * User interfaces for handlers are defined as separate servers with
 * class names of the form "*Interface".  Their Activate functions take
 * an LWInterface structure which they should fill in.  The 'panel'
 * field can be set to point to the non-modal controls for one or more
 * instances.  Otherwise the 'options' function should be set to open a
 * modal dialog for a single instance.  The 'command' function can be
 * set to provide batch commands to operate on instances.
 *
 * Technically this is version 2 of the plugin interface, but for
 * compatibility with legacy UI code, which checks against the handler
 * class API version, all versions less than LWINTERFACE_VERSION will be considered version
 * 1 API functions.  New plugins MUST return AFUNC_BADVERSION when called
 * with a version less than LWINTERFACE_VERSION.
 */
#define LWINTERFACE_VERSION 5

typedef struct st_LWInterface {
    LWInstance    inst;
    LWXPanelID    panel;
    LWError     (*options) (LWInstance);
    LWError     (*command) (LWInstance, LWCStringUTF8 command);
} LWInterface;


/*
 * Because the UI for instances can now operate in a non-modal fashion,
 * handlers need to inform the host when their instance data have changed
 * in a way that will affect the host.  They can do this by getting the
 * "Instance Update" global function.  This can be called at any point
 * with the class name of the instance and the instance pointer itself,
 * and the host will respond to any changes.
 */
#define LWINSTUPDATE_GLOBAL "Instance Update"

typedef void    LWInstUpdate (LWCStringASCII className, LWInstance);


#endif
