// Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of LightWave Digital, Ltd.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_LAYOUT_COMMAND_H
#define LWCPP_LAYOUT_COMMAND_H

#include <lwsdk/Plugin.h>
#include <lwsdk/Wrapper.h>
#include <lwgeneric.h>
#include <cassert>

namespace lwsdk
{
    /// Derive from this class and override the "execute" function with your own implementation.
    class LayoutGenericPlugin
    {
    public:
        LayoutGenericPlugin(GlobalFunc* global_):
            global(global_)
        {}

        int execute()
        { return 0; }

    protected:
        GlobalFunc* global;
    };

    template<typename T>
    class LayoutGenericAdaptor : PluginAdaptor
    {
    public:
        LayoutGenericAdaptor(const char* plugin_name, const char* user_name)
        {
            PluginAdaptor::register_plugin(LWLAYOUTGENERIC_CLASS, activate, plugin_name, user_name);
        }

        static int activate(int version, GlobalFunc* global, void* local, void* serverData)
        {
            init_wrapper(global);
            T inst(global);
            return inst.execute();
        }
    };
}

#endif // LWCPP_LAYOUT_COMMAND_H