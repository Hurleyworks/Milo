// Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of NewTek, Inc.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_PLUGIN_H
#error "Must include Plugin.h before PluginGizmo.h"
#endif

#ifndef LWCPP_GIZMO_H
#define LWCPP_GIZMO_H

#include <lwgizmo.h>

namespace lwsdk
{
    template<class T>
    class PluginGizmoAdaptor : public PluginAdaptor
    {
    public:
        PluginGizmoAdaptor(const char* gclassName, const char* plugin_name, const char* user_name, ServerTagInfo* additional = NULL)
        {
            PluginAdaptor::register_plugin(gclassName, activate, plugin_name, user_name, additional);
        }

        static int activate(int version, GlobalFunc *global, void *local_, void *serverdata)
        {
            if (version < LWGIZMO_VERSION)
                return AFUNC_BADVERSION;

            LWGizmo* local = (LWGizmo*)local_;
            init_wrapper(global);

            if (local == NULL)
                return AFUNC_BADLOCAL;

            local->gizmo->adjust = adjust;
            local->gizmo->count = count;
            local->gizmo->dirty = dirty;
            local->gizmo->down = down;
            local->gizmo->draw = draw;
            local->gizmo->end = end;
            local->gizmo->handle = handle;
            local->gizmo->help = help;
            local->gizmo->move = move;
            local->gizmo->start = start;
            local->gizmo->up = up;

            return AFUNC_OK;
        }

        //void done)     (LWInstance);
        static void draw(LWInstance inst, LWCustomObjAccess* ca)
        {
            assert(inst);
            ((T*)inst)->draw(ca);
        }
        static const char*    help(LWInstance inst, LWToolEvent* te)
        {
            assert(inst);
            return ((T*)inst)->help(te);
        }
        static int dirty(LWInstance inst)
        {
            assert(inst);
            return ((T*)inst)->dirty();
        }
        static int count(LWInstance inst, LWToolEvent* te)
        {
            assert(inst);
            return ((T*)inst)->count(te);
        }
        static int handle(LWInstance inst, LWToolEvent* te, int i, LWDVector pos)
        {
            assert(inst);
            return ((T*)inst)->handle(te, i, pos);
        }
        static int start(LWInstance inst, LWToolEvent* te)
        {
            assert(inst);
            return ((T*)inst)->start(te);
        }
        static int adjust(LWInstance inst, LWToolEvent* te, int i)
        {
            assert(inst);
            return ((T*)inst)->adjust(te, i);
        }
        static int down(LWInstance inst, LWToolEvent* te)
        {
            assert(inst);
            return ((T*)inst)->down(te);
        }
        static void move(LWInstance inst, LWToolEvent* te)
        {
            assert(inst);
            ((T*)inst)->move(te);
        }
        static void up(LWInstance inst, LWToolEvent* te)
        {
            assert(inst);
            ((T*)inst)->up(te);
        }
        static int end(LWInstance inst, LWToolEvent* te, int i)
        {
            assert(inst);
            return ((T*)inst)->end(te, i);
        }
    };
}

/*
Easy copy paste into your instance header

void draw(LWCustomObjAccess* ca);
const char*    help(LWToolEvent *);
int dirty();
int count(LWToolEvent* te);
int handle(LWToolEvent* te, int i, LWDVector pos);
int start(LWToolEvent* te);
int adjust(LWToolEvent* te, int i);
int down(LWToolEvent* te);
void move(LWToolEvent* te);
void up(LWToolEvent* te);
int end(LWToolEvent* te, int i);
*/

#endif // LWCPP_GIZMO_H