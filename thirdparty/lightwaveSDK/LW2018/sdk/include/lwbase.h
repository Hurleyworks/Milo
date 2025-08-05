/*
 * LWSDK Header File
 *
 * LWBASE.H -- LightWave Attributes Base Functions
 *
 *Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of NewTek, Inc.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_BASE_H
#define LWSDK_BASE_H

#define LWBASEFUNCS_GLOBAL "LW Base Funcs"

#include <lwtypes.h>

typedef struct st_LWBase* LWBaseID;
#ifndef LWAttributeID
typedef struct st_lwattribute *LWAttributeID;
#endif

typedef struct st_LWBaseContext
{
    void* context;
    LWBaseID baseID;
} LWBaseContext;

typedef struct st_LWBaseFuncs
{
    /// Attributes accessor
    LWAttributeID (*attributes) (LWBaseID base);

    /// Convenience accessor to attributes for plugin instances.
    /// During plugin creation, passing in nullptr returns the container
    /// for the plugin being created.
    /// After plugin creation a valid LWInstance must be supplied.
    LWAttributeID (*attributesFromInstance) (LWInstance inst);

    /// Flags LWBASEF_* for the plugin.
    unsigned int (*flags) (LWBaseID base);

    /// Returns the LWBaseID associated with an LWInstance.
    /// During plugin creation, passing in nullptr returns the container
    /// for the plugin being created.
    /// After plugin creation a valid LWInstance must be supplied.
    LWBaseID (*baseFromInstance) (LWInstance inst);

    /// Returns the LWInstance associated with an LWBaseID.
    /// Only valid after the plugin has been created.
    LWInstance (*instanceFromBase) (LWBaseID base);

    /// return server class
    const char *(*serverClass)(LWBaseID base);

    /// return server internal name (not localized)
    const char *(*serverName)(LWBaseID base);

} LWBaseFuncs;

#define LWBASEF_DISABLED         (1 << 0)       // Plugin is disabled
#define LWBASEF_HIDDEN           (1 << 1)       // Plugin is hidden from the user
#define LWBASEF_EVALUATORCOPY    (1 << 7)       // Plugin is a copy managed by an evaluator

#endif // LWSDK_BASE_H