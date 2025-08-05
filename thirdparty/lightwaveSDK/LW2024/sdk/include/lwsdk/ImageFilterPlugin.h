// Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of LightWave Digital, Ltd.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_IMAGEFILTER_H
#define LWCPP_IMAGEFILTER_H

#include <lwfilter.h>
#include <lwsdk/Plugin.h>
#include <lwsdk/PluginInterface.h>
#include <lwsdk/Wrapper.h>

#include <cassert>

namespace lwsdk
{
    class ImageFilterPlugin : public PluginBase
    {
    public:
        ImageFilterPlugin(GlobalFunc* global, void* flags) : PluginBase(global)
        {}

        LWError process(void* inst, const LWFilterAccess* fa)
        {
            return nullptr;
        }

        const char** flags()
        {
            return 0;
        }

    protected:
        LWItemID item_id;
    };

    template<class T>
    class ImageFilterPluginAdaptor : public PluginItemAdaptor<T>
    {
    public:
        ImageFilterPluginAdaptor(const char* plugin_name, const char* user_name, const ServerTagInfo* additional = NULL)
        {
            PluginAdaptor::register_plugin(LWIMAGEFILTER_HCLASS, activate, plugin_name, user_name,additional);
        }

        static int activate(int version, GlobalFunc* global, void* local_, void* serverData)
        {
            LWImageFilterHandler* local = (LWImageFilterHandler*)local_;
            init_wrapper(global);
            PluginInstanceAdaptor<T>::activate(version, global, local->inst, serverData);
            PluginItemAdaptor<T>::activate(version, global, local->item, serverData);

            local->inst->create         = create;
            local->flags                = flags;
            local->process              = process;

            return AFUNC_OK;
        }

        // Primitive specific create function
        // Todo: Is the context of *all* plugin classes with item functions an item id?
        // If yes then they could share the same "create" function.
        static LWInstance create(void* data, void* ctx, LWError* err)
        {
            try
            {
                return new T(PluginInstanceAdaptor<T>::global, ctx);
            }
            catch (const char* c)
            {
                *err = c;
                return nullptr;
            }
        }

        static const char** flags(void *inst)
        {
            assert(inst);
            return ((T*)inst)->flags();
        }

        static LWError process(void* inst, const LWFilterAccess* fa)
        {
            assert(inst);
            ((T*)inst)->process(fa);
            return NULL;
        }
    };
}

#endif // LWCPP_PRIMITIVE_H