// Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of LightWave Digital, Ltd.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_CAMERA_H
#define LWCPP_CAMERA_H

#include "lwcamera.h"
#include <lwsdk/Plugin.h>
#include <lwsdk/PluginInterface.h>
#include <lwsdk/Wrapper.h>

namespace lwsdk
{
    class CameraPlugin : public PluginBase
    {
    public:
        CameraPlugin(GlobalFunc* global, LWBaseID base, LWAttributeID attrs, LWItemID id)
            : PluginBase(global)
            , m_base(base)
            , m_attrs(attrs)
            , m_item_id(id)
        {}

        unsigned int flags()
        {
            return 0;
        }

        int preview(double lpx, double lpy, LWCameraEye eye, LWDMatrix4 projection, const LWCameraAccess* camaccess)
        {
            return 0;
        }

        LWError new_frame(const LWFrameInfo* frameinfo, const LWCameraAccess* camaccess)
        {
            return nullptr;
        }

        int evaluate(double fpx, double fpy, double lpx, double lpy, double fractime, LWCameraEye eye, LWCameraRay* ray, const LWCameraAccess* camaccess)
        {
            return 0;
        }

        int project(double lpx, double lpy, LWDVector worldpos, double fractime, LWCameraEye eye, LWDVector fp, const LWCameraAccess* camaccess)
        {
            return 0;
        }

    protected:
        LWItemID m_item_id;
        LWBaseID m_base;
        LWAttributeID m_attrs;
    };

    template<class T>
    class CameraPluginAdaptor : public PluginRenderAdaptor<T>
    {
        std::string user_name;
        std::string plugin_name;

    public:

        CameraPluginAdaptor(const char* plugin_name_, const char* user_name_, ServerTagInfo* additional = NULL) :
            plugin_name(plugin_name_),
            user_name(user_name_)
        {
            PluginAdaptor::register_plugin(LWCAMERA_HCLASS, activate, plugin_name.c_str(), user_name.c_str(), additional);
        }

        static int activate(int version, GlobalFunc* global, void* local_, void* serverData)
        {
            LWCameraHandler* local = (LWCameraHandler*)local_;
            init_wrapper(global);
            PluginInstanceAdaptor<T>::activate(version, global, local->inst, serverData);
            PluginItemAdaptor<T>::activate(version, global, local->item, serverData);
            PluginRenderAdaptor<T>::activate(version, global, local->rend, serverData);
            local->inst->create = create;

            local->flags = flags;
            local->preview = preview;
            local->newFrame = new_frame;
            local->evaluate = evaluate;
            local->project = project;

            return AFUNC_OK;
        }

        static LWInstance create(void* data, void* context, LWError* err)
        {
            LWBaseID baseid = nullptr;
            LWAttributeID attrs = nullptr;

            // find attributes for current server
            LWBaseContainerRef base_container = basefuncs->baseContainerCreate(LWBASECONTEXT_CURRENT, nullptr, nullptr);
            if(base_container)
            {
                if(basefuncs->baseContainerCount(base_container) > 0)
                {
                    baseid = basefuncs->baseContainerAtIndex(base_container, 0);
                    attrs = basefuncs->getAttributes(baseid);
                }
                basefuncs->baseContainerDestroy(base_container);
            }

            LWItemID id = static_cast<LWItemID>(context);
            return new T(PluginInstanceAdaptor<T>::global, baseid, attrs, id);
        }

        static unsigned int flags(LWInstance vinst)
        {
            assert(vinst);
            return ((T*)vinst)->flags();
        }

        static int preview(LWInstance vinst, double lpx, double lpy, LWCameraEye eye, LWDMatrix4 projection, const LWCameraAccess* camaccess)
        {
            assert(vinst);
            return ((T*)vinst)->preview(lpx, lpy, eye, projection, camaccess);
        }

        static LWError new_frame(LWInstance vinst, const LWFrameInfo* frameinfo, const LWCameraAccess* camaccess)
        {
            assert(vinst);
            return ((T*)vinst)->new_frame(frameinfo, camaccess);
        }

        static int evaluate(LWInstance vinst, double fpx, double fpy, double lpx, double lpy, double fractime, LWCameraEye eye, LWCameraRay* ray, const LWCameraAccess* camaccess)
        {
            assert(vinst);
            return ((T*)vinst)->evaluate(fpx, fpy, lpx, lpy, fractime, eye, ray, camaccess);
        }

        static int project(LWInstance vinst, double lpx, double lpy, LWDVector worldpos, double fractime, LWCameraEye eye, LWDVector fp, const LWCameraAccess* camaccess)
        {
            assert(vinst);
            return ((T*)vinst)->project(lpx, lpy, worldpos, fractime, eye, fp, camaccess);
        }
    };
}

#endif // LWCPP_CAMERA_H