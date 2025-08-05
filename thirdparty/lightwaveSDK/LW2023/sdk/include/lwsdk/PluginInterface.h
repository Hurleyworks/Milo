// Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of LightWave Digital, Ltd.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_INTERFACE_H
#define LWCPP_INTERFACE_H

#include <lwsdk/Wrapper.h>

namespace lwsdk
{
    template<class T>
    class PluginInterfaceAdaptor : public PluginAdaptor
    {
    public:
        PluginInterfaceAdaptor(const char* iclassName, const char* plugin_name, const char* user_name = nullptr)
        {
            PluginAdaptor::register_plugin(iclassName, activate, plugin_name, user_name);
        }

        static int activate(int version, GlobalFunc *global, void *local_, void *serverdata)
        {
            if (version < LWINTERFACE_VERSION)
                return AFUNC_BADVERSION;

            LWInterface* local = (LWInterface*)local_;
            init_wrapper(global);

            if (local == NULL)
                return AFUNC_BADLOCAL;

            // From The LightWave SDK docs:
            // Exactly one of the panel and options fields should be non-NULL
            local->panel = ((T*)local->inst)->ui();
            if (!local->panel) {
                local->options = options;
            }
            else
                local->options = NULL;
            local->command = NULL;

            return AFUNC_OK;
        }

        static LWError options(LWInstance inst)
        {
            return ((T*)inst)->panel_ui();
        }

    };
}

#endif // LWCPP_ATTRIBUTE_H