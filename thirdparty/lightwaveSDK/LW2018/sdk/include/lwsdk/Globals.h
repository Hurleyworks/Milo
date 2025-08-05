// Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of NewTek, Inc.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_GLOBALS_H
#define LWCPP_GLOBALS_H

#include <lwserver.h>
#include <lwhost.h>
#include <lwrender.h>
#include <lwsurf.h>
#include <lwbase.h>
#include <lwvparm.h>
#include <lwbxdf.h>
#include <lwlight.h>
#include <lwnodeeditor.h>
#include <lwsurfaceshader.h>
#include <lwaovs.h>
#include <string>
#include <lwmtutil.h>
#include <lwcomring.h>
#include <lwio.h>
#include <lwpanel.h>
#include <lwviewportinfo.h>
#include <lwmodeler.h>
#include <lwsurfed.h>
#include <lwgradient.h>
#include <lwprimitive.h>
#include <cassert>
#include <lwtxtred.h>

#define LWSDK_CPP_WRAPPER_VERSION 1

/// @todo Should be moved to an official LWSDK header.
#define LWCOMMANDFUNC_GLOBAL "LW Command Interface"
typedef int LWCommandFunc ( const char *cmd );

namespace lwsdk
{
    extern size_t system_id;

    extern LWSceneInfo* sceneinfo;
    extern LWInterfaceInfo* intinfo;
    extern LWDirInfoFunc* dirinfo;
    extern LWMessageFuncs* msgfuncs;
    extern LWCommandFunc* commandfunc;
    extern LWChannelInfo* chaninfo;
    extern LWItemInfo* iteminfo;
    extern LWObjectInfo* objectinfo;
    extern LWPrimitiveEvaluationFuncs* primevalfuncs;
    extern LWObjectFuncs* objfuncs;
    extern LWCameraInfo* caminfo;
    extern LWSurfaceFuncs* surffuncs;
    extern LWBaseFuncs* basefuncs;
    extern LWNodeInputFuncs* inputfuncs;
    extern LWNodeOutputFuncs* outputfuncs;
    extern LWVParmFuncs* vparmfuncs;
    extern LWTextureFuncs* txtfuncs;
    extern LWTxtrEdFuncs* txtredfuncs;
    extern LWNodeFuncs* nodefuncs;
    extern LWInstUpdate* instupdate;
    extern LWXPanelFuncs* xpanfuncs;
    extern LWEnvelopeFuncs* envfuncs;
    extern LWBSDFFuncs* bsdffuncs;
    extern LWLightEvaluationFuncs* lightevalfuncs;
    extern LWNodeEditorFuncs* nodeedfuncs;
    extern LWSurfaceEvaluationFuncs* surfevalfuncs;
    extern LWAOVFuncs* aovfuncs;
    extern LWNodeUtilityFuncs* nodeutilfuncs;
    extern LWImageList* imagelist;
    extern LWBackdropInfo* backdropinfo;
    extern LWMTUtilFuncs* mtutilfuncs;
    extern LWFileActivateFunc* filereqfunc;
    extern LWComRing* comring;
    extern LWPanelFuncs* panelfuncs;
    extern LWFileIOFuncs* fileiofuncs;
    extern LWViewportInfo* viewportfuncs;
    extern LWLightInfo* lightinfo;
    extern LWImageUtil* imageutil;
    extern LWStateQueryFuncs* statefuncs;
    extern LWFileTypeFunc* filetypefuncs;
    extern LWTimeInfo* timeinfo;
    extern LWSurfEdFuncs* surfedfuncs;
    extern LWGradientFuncs* gradientfuncs;
    extern LWFogInfo* foginfo;

    extern GlobalFunc* globalfunc;

    /// Implicitly called by lwsdk::init_wrapper
    inline void set_global(GlobalFunc* func)
    {
        globalfunc = func;
        system_id = (size_t)globalfunc(LWSYSTEMID_GLOBAL, GFUSE_TRANSIENT);

        size_t app_id = system_id & LWSYS_TYPEBITS;

        // skip for "OTHER" hosts
        if (app_id == LWSYS_OTHER)
            return;

        if (app_id == LWSYS_LAYOUT || app_id == LWSYS_SCREAMERNET)
        {
            sceneinfo = (LWSceneInfo*)(*globalfunc)(LWSCENEINFO_GLOBAL, GFUSE_TRANSIENT);
            iteminfo = (LWItemInfo*)globalfunc(LWITEMINFO_GLOBAL, GFUSE_TRANSIENT);
            objectinfo = (LWObjectInfo*)globalfunc(LWOBJECTINFO_GLOBAL, GFUSE_TRANSIENT);
            primevalfuncs = (LWPrimitiveEvaluationFuncs*)globalfunc(LWPRIMITIVEEVALUATIONFUNCS_GLOBAL, GFUSE_TRANSIENT);
            caminfo = (LWCameraInfo*)globalfunc(LWCAMERAINFO_GLOBAL, GFUSE_TRANSIENT);
            commandfunc = (LWCommandFunc *)globalfunc(LWCOMMANDFUNC_GLOBAL, GFUSE_TRANSIENT);
            lightevalfuncs = (LWLightEvaluationFuncs*)globalfunc(LWLIGHTEVALUATIONFUNCS_GLOBAL, GFUSE_TRANSIENT);
            backdropinfo = (LWBackdropInfo*)globalfunc(LWBACKDROPINFO_GLOBAL, GFUSE_TRANSIENT);
            lightinfo = (LWLightInfo*)globalfunc(LWLIGHTINFO_GLOBAL, GFUSE_TRANSIENT);
            timeinfo = (LWTimeInfo*)globalfunc(LWTIMEINFO_GLOBAL, GFUSE_TRANSIENT);
            foginfo = (LWFogInfo*)globalfunc(LWFOGINFO_GLOBAL, GFUSE_TRANSIENT);

            assert(sceneinfo && iteminfo && objectinfo && primevalfuncs &&
                caminfo && commandfunc && lightevalfuncs &&
                backdropinfo && lightinfo && foginfo);
        }

        // Globals not available in Screamernet
        if (app_id != LWSYS_SCREAMERNET)
        {
            if (app_id == LWSYS_LAYOUT)
            {
                intinfo = (LWInterfaceInfo *)(*globalfunc)(LWINTERFACEINFO_GLOBAL, GFUSE_TRANSIENT);
                assert(intinfo);
            }

            statefuncs = (LWStateQueryFuncs*)globalfunc(LWSTATEQUERYFUNCS_GLOBAL, GFUSE_TRANSIENT);
            xpanfuncs = (LWXPanelFuncs*)globalfunc(LWXPANELFUNCS_GLOBAL, GFUSE_TRANSIENT);
            panelfuncs = (LWPanelFuncs*)globalfunc(LWPANELFUNCS_GLOBAL, GFUSE_TRANSIENT);
            filereqfunc = (LWFileActivateFunc*)globalfunc(LWFILEACTIVATEFUNC_GLOBAL, GFUSE_TRANSIENT);
            viewportfuncs = (LWViewportInfo *)(*globalfunc)(LWVIEWPORTINFO_GLOBAL, GFUSE_TRANSIENT);
            filetypefuncs = (LWFileTypeFunc*)globalfunc(LWFILETYPEFUNC_GLOBAL, GFUSE_TRANSIENT);
            surfedfuncs = (LWSurfEdFuncs*)globalfunc(LWSURFEDFUNCS_GLOBAL, GFUSE_TRANSIENT);
            assert(statefuncs && xpanfuncs && filereqfunc && viewportfuncs && filetypefuncs);// some plugins are created *before* the panel functions are available // && panelfuncs && surfedfuncs);
        }

        // Common globals available in Layout, Modeler and Screamernet
        dirinfo = (LWDirInfoFunc*)globalfunc(LWDIRINFOFUNC_GLOBAL, GFUSE_TRANSIENT);
        msgfuncs = (LWMessageFuncs*)(*globalfunc)(LWMESSAGEFUNCS_GLOBAL, GFUSE_TRANSIENT);
        chaninfo = (LWChannelInfo*)globalfunc(LWCHANNELINFO_GLOBAL, GFUSE_TRANSIENT);
        objfuncs = (LWObjectFuncs *)globalfunc(LWOBJECTFUNCS_GLOBAL, GFUSE_TRANSIENT);
        surffuncs = (LWSurfaceFuncs*)globalfunc(LWSURFACEFUNCS_GLOBAL, GFUSE_TRANSIENT);
        basefuncs = (LWBaseFuncs*)globalfunc(LWBASEFUNCS_GLOBAL, GFUSE_TRANSIENT);
        outputfuncs = (LWNodeOutputFuncs*)globalfunc(LWNODEOUTPUTFUNCS_GLOBAL, GFUSE_TRANSIENT);
        inputfuncs = (LWNodeInputFuncs*)globalfunc(LWNODEINPUTFUNCS_GLOBAL, GFUSE_TRANSIENT);
        vparmfuncs = (LWVParmFuncs*)globalfunc(LWVPARMFUNCS_GLOBAL, GFUSE_TRANSIENT);
        txtfuncs = (LWTextureFuncs*)globalfunc(LWTEXTUREFUNCS_GLOBAL, GFUSE_TRANSIENT);
        txtredfuncs = (LWTxtrEdFuncs*)globalfunc(LWTXTREDFUNCS_GLOBAL, GFUSE_TRANSIENT);
        nodefuncs = (LWNodeFuncs*)globalfunc(LWNODEFUNCS_GLOBAL, GFUSE_TRANSIENT);
        instupdate = (LWInstUpdate*)globalfunc(LWINSTUPDATE_GLOBAL, GFUSE_TRANSIENT);
        envfuncs = (LWEnvelopeFuncs*)globalfunc(LWENVELOPEFUNCS_GLOBAL, GFUSE_TRANSIENT);
        bsdffuncs = (LWBSDFFuncs*)globalfunc(LWBSDFFUNCS_GLOBAL, GFUSE_TRANSIENT);
        nodeedfuncs = (LWNodeEditorFuncs*)globalfunc(LWNODEEDITORFUNCS_GLOBAL, GFUSE_TRANSIENT);
        surfevalfuncs = (LWSurfaceEvaluationFuncs*)globalfunc(LWSURFACEEVALUATIONFUNCS_GLOBAL, GFUSE_TRANSIENT);
        aovfuncs = (LWAOVFuncs*)globalfunc(LWAOVFUNCS_GLOBAL, GFUSE_TRANSIENT);
        nodeutilfuncs = (LWNodeUtilityFuncs*)globalfunc(LWNODEUTILITYFUNCS_GLOBAL, GFUSE_TRANSIENT);
        imagelist = (LWImageList*)globalfunc(LWIMAGELIST_GLOBAL, GFUSE_TRANSIENT);
        fileiofuncs = (LWFileIOFuncs*)globalfunc(LWFILEIOFUNCS_GLOBAL, GFUSE_TRANSIENT);
        comring = (LWComRing*)globalfunc(LWCOMRING_GLOBAL, GFUSE_TRANSIENT);
        mtutilfuncs = (LWMTUtilFuncs*)globalfunc(LWMTUTILFUNCS_GLOBAL, GFUSE_TRANSIENT);
        imageutil = (LWImageUtil*)globalfunc(LWIMAGEUTIL_GLOBAL, GFUSE_TRANSIENT);
        gradientfuncs = (LWGradientFuncs*)globalfunc(LWGRADIENTFUNCS_GLOBAL, GFUSE_TRANSIENT);

        assert(dirinfo
            && msgfuncs
            && chaninfo
            && objfuncs
            && surffuncs
            && basefuncs
            && outputfuncs
            && inputfuncs
            && vparmfuncs
            && nodefuncs
            && instupdate
            && envfuncs
            && bsdffuncs
            && nodeedfuncs
            && comring
            && fileiofuncs
            && surfevalfuncs
            && aovfuncs
            && nodeutilfuncs
            && imagelist
            && mtutilfuncs
            && imageutil
            && gradientfuncs
            );
    }

    inline GlobalFunc* get_global()
    {
        assert(globalfunc && "The LWSDK wrapper was not initialized with a valid global!");
        return globalfunc;
    }

    /// The intinfo, timeinfo and sceneinfo globals need to be re-acquired on every call due to direct data members.
    /// {
    inline void refresh_intinfo()
    {
        // needs to be re-acquired every time to assure the data is up to date.
        size_t app_id = system_id & LWSYS_TYPEBITS;
        if (app_id == LWSYS_LAYOUT)
            intinfo = (LWInterfaceInfo *)(*globalfunc)(LWINTERFACEINFO_GLOBAL, GFUSE_TRANSIENT);
    }

    inline void refresh_timeinfo()
    {
        // needs to be re-acquired every time to assure the data is up to date.
        size_t app_id = system_id & LWSYS_TYPEBITS;
        if (app_id == LWSYS_LAYOUT || app_id == LWSYS_SCREAMERNET)
            timeinfo = (LWTimeInfo *)(*globalfunc)(LWTIMEINFO_GLOBAL, GFUSE_TRANSIENT);
    }

    inline void refresh_sceneinfo()
    {
        size_t app_id = system_id & LWSYS_TYPEBITS;
        if (app_id == LWSYS_LAYOUT || app_id == LWSYS_SCREAMERNET)
            sceneinfo = (LWSceneInfo*)(*globalfunc)(LWSCENEINFO_GLOBAL, GFUSE_TRANSIENT);
    }

    inline void refresh_foginfo()
    {
        size_t app_id = system_id & LWSYS_TYPEBITS;
        if (app_id == LWSYS_LAYOUT || app_id == LWSYS_SCREAMERNET)
            foginfo = (LWFogInfo*)globalfunc(LWFOGINFO_GLOBAL, GFUSE_TRANSIENT);
    }

    /// }

    inline void execute_command(const std::string& cmd)
    {
        commandfunc(cmd.c_str());
    }
}

#define LWSDK_STATIC_GLOBAL_DEFINES \
namespace lwsdk  \
{ \
    GlobalFunc* globalfunc = NULL; \
     \
    size_t system_id = 0; \
    LWDirInfoFunc* dirinfo = NULL; \
    LWMessageFuncs* msgfuncs = NULL; \
    LWCommandFunc* commandfunc = NULL; \
    LWSceneInfo* sceneinfo = NULL; \
    LWInterfaceInfo* intinfo = NULL; \
    LWChannelInfo* chaninfo = NULL; \
    LWItemInfo* iteminfo = NULL; \
    LWObjectInfo* objectinfo = NULL; \
    LWPrimitiveEvaluationFuncs* primevalfuncs = NULL; \
    LWObjectFuncs* objfuncs = NULL; \
    LWCameraInfo* caminfo = NULL; \
    LWSurfaceFuncs* surffuncs = NULL; \
    LWBaseFuncs* basefuncs = NULL; \
    LWNodeInputFuncs* inputfuncs = NULL; \
    LWNodeOutputFuncs* outputfuncs = NULL; \
    LWVParmFuncs* vparmfuncs = NULL; \
    LWTextureFuncs* txtfuncs = NULL; \
    LWTxtrEdFuncs* txtredfuncs = NULL; \
    LWNodeFuncs* nodefuncs = NULL; \
    LWInstUpdate* instupdate = NULL; \
    LWXPanelFuncs* xpanfuncs = NULL; \
    LWEnvelopeFuncs* envfuncs = NULL; \
    LWBSDFFuncs* bsdffuncs = NULL; \
    LWLightEvaluationFuncs* lightevalfuncs = NULL; \
    LWNodeEditorFuncs* nodeedfuncs = NULL; \
    LWSurfaceEvaluationFuncs* surfevalfuncs = NULL; \
    LWAOVFuncs* aovfuncs = NULL; \
    LWNodeUtilityFuncs* nodeutilfuncs = NULL; \
    LWBackdropInfo* backdropinfo = NULL; \
    LWImageList* imagelist = NULL; \
    LWMTUtilFuncs* mtutilfuncs = NULL; \
    LWFileActivateFunc* filereqfunc = NULL; \
    LWComRing* comring = NULL; \
    LWPanelFuncs* panelfuncs = NULL; \
    LWFileIOFuncs* fileiofuncs = NULL; \
    LWViewportInfo* viewportfuncs = NULL; \
    LWLightInfo* lightinfo = NULL; \
    LWImageUtil* imageutil = NULL; \
    LWStateQueryFuncs* statefuncs = NULL; \
    LWFileTypeFunc* filetypefuncs = NULL; \
    LWTimeInfo* timeinfo = NULL; \
    LWSurfEdFuncs* surfedfuncs = NULL; \
    LWGradientFuncs* gradientfuncs = NULL; \
    LWFogInfo* foginfo = NULL; \
} \

#endif // LWCPP_GLOBALS_H