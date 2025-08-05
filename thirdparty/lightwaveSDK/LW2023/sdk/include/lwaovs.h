/*
* LWSDK Header File
*
* LWAOVS.H -- LightWave AOVs
*
*Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
*
*This file contains confidential and proprietary information of LightWave Digital, Ltd.,
*and is subject to the terms of the LightWave End User License Agreement (EULA).
*/
#ifndef LWSDK_AOVS_H
#define LWSDK_AOVS_H

#include <lwtypes.h>
#include <stddef.h>

typedef enum {
    LWAOVType_Custom = 0,
    LWAOVType_Float,
    LWAOVType_Int
} LWAOVType;

typedef unsigned int LWAOVID;

#define LWAOVFLAG_DIRECT        (1 << 0)
#define LWAOVFLAG_DISCRETE      (1 << 1)
#define LWAOVFLAG_CUSTOMSURFACE (1 << 2)
#define LWAOVFLAG_ORIGIN_RENDERER (1 << 3)
#define LWAOVFLAG_INTEGER       (1 << 4)
#define LWAOVFLAG_COVERAGE      (1 << 5)

typedef struct st_LWAOVDefinition
{
    LWAOVType       type;
    unsigned int    numComponents;
    size_t          componentSize; // Data size for a single custom type component.
    unsigned int    flags;
} LWAOVDefinition;

#define LWAOVFUNCS_GLOBAL "AOV Functions"

/** Generic attributes change events. */
typedef enum {
    LWAOV_PRE_ADD,
    LWAOV_POST_ADD,
    LWAOV_PRE_REMOVE,
    LWAOV_POST_REMOVE
} LWAOVEvent;

typedef struct st_LWAOVEventData
{
    LWCStringUTF8 name;
    const LWAOVDefinition*  definition;
} LWAOVEventData;

typedef int(*LWAOVEventFunc)(void* userData, void* priv, int event, LWAOVEventData* aovEventData);

typedef struct st_LWAOVFuncs
{
    LWAOVID                 (*registerAOV)(LWCStringUTF8, const LWAOVDefinition*);
    void                    (*unregisterAOV)(LWAOVID);
    unsigned int            (*numAOVs)();
    LWCStringUTF8           (*name)(unsigned int index);
    const LWAOVDefinition*  (*definition)(unsigned int index);
    LWAOVID                 (*nameToID)(LWCStringUTF8);
    void                    (*addCustomAOV)(LWCStringUTF8);
    void                    (*removeCustomAOV)(LWCStringUTF8);
    int                     (*setChangeEvent)(LWAOVEventFunc evntFun, void* userData);
} LWAOVFuncs;

#endif