// Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of NewTek, Inc.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_CUSTOM_OBJECT_H
#define LWCPP_CUSTOM_OBJECT_H

#include <lwsdk/Plugin.h>
#include <lwsdk/Wrapper.h>
#include <lwcustobj.h>
#include <cassert>

namespace lwsdk
{
    /// Derive from this class and override the "evaluate" function with your own implementation.
    class CustomObjectPlugin : public PluginBase
    {
    public:
        CustomObjectPlugin(GlobalFunc* global, LWItemID id) : PluginBase(global),
            item_id(id)
        {}

        void evaluate(const LWCustomObjAccess* access)
        {}

        unsigned int flags()
        {
            return 0;
        }

    protected:
        LWItemID item_id;
    };

    template<class T>
    class CustomObjectAdaptor : public PluginRenderAdaptor<T>
    {
    public:

        CustomObjectAdaptor(const char* plugin_name)
        {
            PluginAdaptor::register_plugin(LWCUSTOMOBJ_HCLASS, activate, plugin_name, plugin_name);
        }

        static int activate(int version, GlobalFunc* global, void* local_, void* serverData)
        {
            LWCustomObjHandler* local = (LWCustomObjHandler*)local_;
            init_wrapper(global);

            PluginInstanceAdaptor<T>::activate(version, global, local->inst, serverData);
            PluginItemAdaptor<T>::activate(version, global, local->item, serverData);
            PluginRenderAdaptor<T>::activate(version, global, local->rend, serverData);
            local->inst->create = create;
            local->evaluate     = evaluate;
            local->flags        = flags;
            return AFUNC_OK;
        }

        // CustomObject specific create function
        // Todo: Is the context of *all* plugin classes with item functions an item id?
        // If yes then they could share the same "create" function.
        static LWInstance create(void* data, LWItemID id, LWError* err)
        {
            return new T(PluginInstanceAdaptor<T>::global, id);
        }

        static void evaluate(void *inst, const LWCustomObjAccess* cobj_access)
        {
            assert(inst);
            ((T*)inst)->evaluate(cobj_access);
        }

        static unsigned int flags(void *inst)
        {   assert(inst);
        return ((T*)inst)->flags();
        }
    };
}

#endif // LWCPP_CUSTOM_OBJECT_H