// Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of NewTek, Inc.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_INTERFACE_H
#define LWCPP_INTERFACE_H

#include <lwsdk/Wrapper.h>

namespace lwsdk
{
    typedef LWError (*PanelUIFunc) (LWInstance);

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

            local->panel = ((T*)local->inst)->ui();
            local->options = NULL; //PanelUIFunc
            local->command = NULL;

            return AFUNC_OK;
        }

        static int activate_options(int version, GlobalFunc *global, void *local_, void *serverdata)
        {
            if (version < LWINTERFACE_VERSION)
                return AFUNC_BADVERSION;

            LWInterface* local = (LWInterface*)local_;
            init_wrapper(global);

            if (local == NULL)
                return AFUNC_BADLOCAL;

            local->panel = NULL;
            local->options = options;
            local->command = NULL;

            return AFUNC_OK;
        }

        static LWError options(LWInstance inst)
        {
            return ((T*)inst)->options();
        }
    };
}

#endif // LWCPP_ATTRIBUTE_H