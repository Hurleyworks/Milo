// Copyright © 2025 LightWave Digital, Ltd. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of LightWave Digital, Ltd.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_PIXELFILTER_H
#define LWCPP_PIXELFILTER_H

#include <lwfilter.h>
#include <lwsdk/Plugin.h>
#include <lwsdk/PluginInterface.h>
#include <lwsdk/Wrapper.h>

#include <cassert>

namespace lwsdk
{
class PixelFilterPlugin : public PluginBase
{
public:
    PixelFilterPlugin(GlobalFunc *global, void *flags) : PluginBase(global) {}

    void evaluate(void *inst, const LWPixelAccess *pa) {}

    const char **flags()
    {
        return nullptr;
    }

    unsigned int renderFlags()
    {
        return 0;
    }

    void preprocess(LWPreprocessAccess *, LWRenderGlobals *, LWRenderState *, LWRenderStateID *) {}
};

template <class T> class PixelFilterPluginAdaptor : public PluginRenderAdaptor<T>
{
public:
    PixelFilterPluginAdaptor(const char *plugin_name, const char *user_name, const ServerTagInfo *additional = nullptr)
    {
        PluginAdaptor::register_plugin(LWPIXELFILTER_HCLASS, activate, plugin_name, user_name, additional);
    }

    static int activate(int version, GlobalFunc *global, void *local_, void *serverData)
    {
        LWPixelFilterHandler *local = (LWPixelFilterHandler *)local_;
        init_wrapper(global);
        PluginInstanceAdaptor<T>::activate(version, global, local->inst, serverData);
        PluginItemAdaptor<T>::activate(version, global, local->item, serverData);
        PluginRenderAdaptor<T>::activate(version, global, local->rend, serverData);

        local->inst->create = create;
        local->flags = flags;
        local->renderFlags = renderFlags;
        local->evaluate = evaluate;
        local->preprocess = preprocess;

        return AFUNC_OK;
    }

    // Primitive specific create function
    // Todo: Is the context of *all* plugin classes with item functions an item id?
    // If yes then they could share the same "create" function.
    static LWInstance create(void *data, void *ctx, LWError *err)
    {
        try
        {
            return new T(PluginInstanceAdaptor<T>::global, ctx);
        } catch (const char *c)
        {
            *err = c;
            return nullptr;
        }
    }

    static const char **flags(void *inst)
    {
        assert(inst);
        return ((T *)inst)->flags();
    }

    static unsigned int renderFlags(void *inst)
    {
        assert(inst);
        return ((T *)inst)->renderFlags();
    }

    static void evaluate(void *inst, const LWPixelAccess *pa)
    {
        assert(inst);
        ((T *)inst)->evaluate(pa);
    }

    static void preprocess(LWInstance inst, LWPreprocessAccess *pa, LWRenderGlobals *rg, LWRenderState *rs,
                           LWRenderStateID *rsids)
    {
        assert(inst);
        ((T *)inst)->preprocess(pa, rg, rs, rsids);
    }
};
} // namespace lwsdk

#endif // LWCPP_PRIMITIVE_H