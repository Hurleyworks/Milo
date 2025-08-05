// Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of NewTek, Inc.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_SURFACE_SHADER_H
#define LWCPP_SURFACE_SHADER_H

#include <lwsdk/Plugin.h>
#include <lwsdk/Wrapper.h>
#include <lwbase.h>
#include "lwsurfaceshader.h"

namespace lwsdk
{
    class SurfaceShaderPlugin : public PluginBase
    {
    public:
        SurfaceShaderPlugin(GlobalFunc* global, LWAttributeID attrs_, LWSurfaceID sid) : PluginBase(global),
            m_surface_id(sid),
            m_attrs(attrs_)
        {}

        double shade(const LWRay*, LWShadingGeometry*, LWBSDF, LWDVector) { return 0; }
        double F(const LWRay*, LWShadingGeometry*, LWBSDF, const LWDVector, const LWDVector, unsigned int, LWDVector) { return 0; }
        void sampleF(const LWRay*, LWShadingGeometry*, LWBSDF, const LWDVector, LWBxDFSample*, unsigned int) {}

    protected:
        LWSurfaceID m_surface_id;
        LWAttributeID m_attrs;
    };

    template<class T>
    class SurfaceShaderAdaptor : public PluginRenderAdaptor<T>
    {
    public:

        SurfaceShaderAdaptor(const char* plugin_name, const char* user_name)
        {
            PluginAdaptor::register_plugin(LWSURFSHAD_HCLASS, activate, plugin_name, user_name, nullptr);
        }

        static int activate(int version, GlobalFunc* global, void* local_, void* serverData)
        {
            LWSurfaceShaderHandler* local = (LWSurfaceShaderHandler*)local_;
            init_wrapper(global);

            PluginInstanceAdaptor<T>::activate(version, global, local->inst, serverData);
            PluginItemAdaptor<T>::activate(version, global, local->item, serverData);
            PluginRenderAdaptor<T>::activate(version, global, local->rend, serverData);
            local->inst->create = create;
            local->shade = shade;
            local->F = F;
            local->sampleF = sampleF;
            return AFUNC_OK;
        }

        static LWInstance create(void* data, void* context, LWError* err)
        {
            LWBaseContext* ctx = (LWBaseContext*)context;
            LWAttributeID attrs = lwsdk::basefuncs->attributes(ctx->baseID);
            LWSurfaceID sid = static_cast<LWSurfaceID>(ctx->context);
            return new T(PluginInstanceAdaptor<T>::global, attrs, sid);
        }

        static double shade(LWInstance vinst, const LWRay* ray, LWShadingGeometry* is, LWBSDF bsdf, LWDVector result)
        {
            assert(vinst);
            return ((T*)vinst)->shade(ray, is, bsdf, result);
        }

        static double F(LWInstance vinst, const LWRay* ray, LWShadingGeometry* is, LWBSDF bsdf, const LWDVector wo, const LWDVector wi, unsigned int bxdfFlags, LWDVector f)
        {
            assert(vinst);
            return ((T*)vinst)->F(ray, is, bsdf, wo, wi, bxdfFlags, f);
        }

        static void sampleF(LWInstance vinst, const LWRay* ray, LWShadingGeometry* is, LWBSDF bsdf, const LWDVector sample, LWBxDFSample* so, unsigned int bxdfFlags)
        {
            assert(vinst);
            ((T*)vinst)->sampleF(ray, is, bsdf, sample, so, bxdfFlags);
        }
    };
}
#endif // LWCPP_SURFACE_SHADER_H