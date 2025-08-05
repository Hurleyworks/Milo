// Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of LightWave Digital, Ltd.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_NODE_H
#define LWCPP_NODE_H

#include <lwnodes.h>
#include <lwsdk/Plugin.h>
#include <lwsdk/PluginInterface.h>
#include <lwsdk/Wrapper.h>

#include <cassert>

namespace lwsdk
{
    class NodePlugin : public PluginBase
    {
    public:
        NodePlugin(GlobalFunc* global, NodeID id) : PluginBase(global),
            node_id(id)
        {}

        NodePlugin& operator=(const NodePlugin& other)
        {
            PluginBase::operator=(other);
            // The node_id should not be copied.
            return *this;
        }

        void evaluate(LWShadingGeometry*, NodeOutputID, NodeValue)
        {}

        unsigned int flags()
        {
            return 0;
        }

        void custom_preview(int width, int height)
        {}

        unsigned int material_gl(LWNodeOGLMaterial* mat)
        {
            return 0;
        }

        LWXPanelID ui()
        {
            return nullptr;
        }

    protected:
        NodeID node_id;
    };

    template<class T>
    class NodePluginAdaptor : public PluginRenderAdaptor<T>
    {
    public:

        NodePluginAdaptor(const char* plugin_name, const char* user_name, const ServerTagInfo* additional = NULL)
        {
            PluginAdaptor::register_plugin(LWNODE_HCLASS, activate, plugin_name, user_name, additional);
        }

        static int activate(int version, GlobalFunc* global, void* local_, void* serverData)
        {
            if (version != LWNODECLASS_VERSION)
                return (AFUNC_BADVERSION);

            LWNodeHandler* local = (LWNodeHandler*)local_;
            init_wrapper(global);
            PluginInstanceAdaptor<T>::activate(version, global, local->inst, serverData);
            PluginItemAdaptor<T>::activate(version, global, local->item, serverData);
            PluginRenderAdaptor<T>::activate(version, global, local->rend, serverData);

            local->inst->create         = create;
            local->evaluate             = evaluate;
            local->customPreview        = customPreview;
            local->materialGL           = materialGL;
            local->flags                = flags;

            return AFUNC_OK;
        }

        // Primitive specific create function
        // Todo: Is the context of *all* plugin classes with item functions an item id?
        // If yes then they could share the same "create" function.
        static LWInstance create(void* data, LWItemID id, LWError* err)
        {
            return new T(PluginInstanceAdaptor<T>::global, id);
        }

        static unsigned int flags(void *inst)
        {
            assert(inst);
            return ((T*)inst)->flags();
        }

        static void evaluate(LWInstance inst, LWShadingGeometry* sg, NodeOutputID oID, NodeValue val)
        {
            assert(inst);
            return ((T*)inst)->evaluate(sg, oID, val);
        }
        static void   customPreview(LWInstance inst, int width, int height)
        {
            assert(inst);
            return ((T*)inst)->custom_preview(width, height);
        }
        static unsigned int materialGL(LWInstance inst, LWNodeOGLMaterial* mat)
        {
            assert(inst);
            return ((T*)inst)->material_gl(mat);
        }

    };
}

#endif // LWCPP_NODE_H