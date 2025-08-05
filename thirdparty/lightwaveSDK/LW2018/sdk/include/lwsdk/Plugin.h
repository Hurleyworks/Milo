// Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of NewTek, Inc.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_PLUGIN_H
#define LWCPP_PLUGIN_H

#include <cassert>
#include <vector>
#include <lwserver.h>
#include <lwhandler.h>
#include <lwrender.h>
#include <lwio.h>

///////////////////////////////////////////////////////////////////////////////
// Defines and Macros to set up exported symbols
extern "C"
{
    // Sufficiently large static server description struct, to be filled by static plugin adaptor instances.
    extern ServerRecord ServerDesc[];
};

#define LW_SERVER_DESC_DEFINE(num_reserved) \
extern "C" \
{ \
    ServerRecord ServerDesc[num_reserved+1] = { \
        { NULL } \
    }; \
}; \
namespace lwsdk \
{ \
    int next_free_serverdesc_entry() \
    { \
        int i = 0; \
        for (; ServerDesc[i].className != NULL; ++i); \
        assert(i < (num_reserved)); \
        ServerDesc[i + 1].className = NULL; \
        return i; \
    } \
}


namespace lwsdk
{
    /// Returns the next free entry in the server description array.
    int next_free_serverdesc_entry();

    /// Base class for all plugins
    class PluginBase
    {
    public:
        PluginBase(GlobalFunc* global_) :
            global(global_)
        {}

        /// Guard against accidental calls to the copy constructor.
        PluginBase(const PluginBase& other) = delete;

        /// Instance Handler Functions
        /// Note: create, copy and destroy have to be implemented using
        /// a C++ constructor, destructor and assignment operator.
        /// {
        LWError load(const LWLoadState* lstate) { return nullptr; }
        LWError save(const LWSaveState *sstate) { return nullptr; }
        const char* describe() { return ""; }
        /// }

        /// Item Handler Functions
        /// {
        const LWItemID* use_items() { return nullptr; }
        void change_id(const LWItemID *ids) {}
        void change_name(const LWNameChangeData* cd) {}
        /// }

        /// Render Handler Functions
        /// {
        LWError init(int mode) { return nullptr; }
        LWError new_time(LWFrame fr, LWTime t) { return nullptr; }
        void cleanup() {}
        /// }

    protected:
        GlobalFunc* global;

        /// A base class destructor should be either public and virtual, or protected and nonvirtual.
        ~PluginBase()
        {}
    };

    class PluginAdaptor
    {
        std::vector<ServerTagInfo> server_tags;
        void add_server_tag(const char* str, unsigned int tag)
        {
            server_tags[server_tags.size() - 1] = { str, tag };
            server_tags.push_back({ nullptr, 0 });
        }
    public:
        PluginAdaptor()
        {
            // initial zero-entry
            server_tags.push_back({ nullptr, 0 });
        }

        /// helper function to register the new plugin
        void register_plugin(const char* class_name, ActivateFunc* activate, const char* plugin_name, const char* user_name, const ServerTagInfo* additional = NULL)
        {
            assert(class_name && activate && plugin_name);
            int i = next_free_serverdesc_entry();

            // add the server record
            ServerDesc[i].className = class_name;
            ServerDesc[i].name = plugin_name;
            ServerDesc[i].activate = activate;

            if (user_name)
                add_server_tag(user_name, SRVTAG_USERNAME);
            if (additional)
                for (int i = 0; additional[i].string; ++i)
                    add_server_tag(additional[i].string, additional[i].tag);

            ServerDesc[i].tagInfo = &server_tags[0];
        }
    };

    /// To avoid the overhead of runtime polymorphism
    /// we are use compile-time polymorphism.
    template<class T>
    class PluginInstanceAdaptor : public PluginAdaptor
    {
    protected:
        static GlobalFunc*  global;
    public:
        static int activate (int version, GlobalFunc* global_, LWInstanceFuncs* inst, void* serverData)
        {
            global = global_;
            // Create is specific to the plugin class to allow construction
            // taking the class-specific "context" argument into account.
            //inst->create  = create;
            inst->destroy = destroy;
            inst->load    = load;
            inst->save    = save;
            inst->copy    = copy;
            inst->descln  = describe;
            return AFUNC_OK;
        }

        static void destroy(void* inst)
        {
            T* to_delete = (T*) inst;
            delete to_delete;
        }

        static LWError load(void* inst, const LWLoadState* lstate)
        {
            assert(inst);
            return ((T*)inst)->load(lstate);
        }

        XCALL_(static LWError)save(void *inst, const LWSaveState *sstate)
        {
            assert(inst);
            return ((T*)inst)->save(sstate);
        }

        XCALL_( static LWError )copy(void *to, void *from)
        {
            assert(to && from);
            (*(T*)to) = (*(T*)from);
            return NULL;
        }

        XCALL_( static const char * )describe(void *inst)
        {
            assert(inst);
            return ((T*)inst)->describe();
        }
    };

    template<class T>
    GlobalFunc* PluginInstanceAdaptor<T>::global = NULL;

    template<class T>
    class PluginItemAdaptor : public PluginInstanceAdaptor<T>
    {
    public:
        static int activate(int version, GlobalFunc* global_, LWItemFuncs* item, void* serverData)
        {
            // item handler functions are not available in Modeler
            if (item)
            {
                item->useItems = use_items;
                item->changeID = change_id;
                item->changeName = change_name;
            }

            return AFUNC_OK;
        }

        XCALL_(static const LWItemID*)use_items(void *inst)
        {
            assert(inst);
            return ((T*)inst)->use_items();
        }

        XCALL_(static void)change_id(void* inst, const LWItemID *ids)
        {
            assert(inst);
            return ((T*)inst)->change_id(ids);
        }

        XCALL_(static void)change_name(LWInstance inst, const LWNameChangeData* cd)
        {
            assert(inst);
            return ((T*)inst)->change_name(cd);
        }
    };

    template<class T>
    class PluginRenderAdaptor : public PluginItemAdaptor<T>
    {
    public:
        static int activate(int version, GlobalFunc* global_, LWRenderFuncs* rend, void* serverData)
        {
            rend->init = init;
            rend->newTime = new_time;
            rend->cleanup = cleanup;
            return AFUNC_OK;
        }

        XCALL_(static LWError)init(void *inst, int mode)
        {
            assert(inst);
            return ((T*)inst)->init(mode);
        }

        XCALL_(static LWError)new_time(void *inst, LWFrame fr, LWTime t)
        {
            assert(inst);
            return ((T*)inst)->new_time(fr, t);
        }

        XCALL_(static void)cleanup(void *inst)
        {
            assert(inst);
            return ((T*)inst)->cleanup();
        }
    };

    /// Helper to register plugins not using the C++ Wrappers through static class instances.
    class PluginRegistrar : public PluginAdaptor
    {
    public:
        PluginRegistrar(const char * class_name, const char* plugin_name, const char* user_name, ActivateFunc* activate, ServerTagInfo* additional = NULL)
        {
            register_plugin(class_name, activate, plugin_name, user_name, additional);
        }
    };
}

#endif // LWCPP_PLUGIN_H