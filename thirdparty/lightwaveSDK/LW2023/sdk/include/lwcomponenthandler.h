#ifndef LWSDK_COMPONENTHANDLER_H
#define LWSDK_COMPONENTHANDLER_H

#include <lwhandler.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LWCOMPONENT_HCLASS "ComponentHandler"
#define LWCOMPONENT_ICLASS "ComponentInterface"
#define LWCOMPONENT_GCLASS "ComponentGizmo"
#define LWCOMPONENT_ACLASS "ComponentAttribute"
#define LWCOMPONENT_VERSION 1

typedef struct st_LWComponentHandler {

    LWInstanceFuncs  *inst;
    // create context is a LWBaseContext with knowledge of what LWBaseContextID is used

} LWComponentHandler;

#ifdef __cplusplus
}
#endif

#endif // LWSDK_COMPONENTHANDLER_H