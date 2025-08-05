// Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of LightWave Digital, Ltd.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_LIGHT_H
#define LWCPP_LIGHT_H

#include "lwlight.h"
#include <lwsdk/Plugin.h>
#include <lwsdk/PluginInterface.h>
#include <lwsdk/Wrapper.h>

namespace lwsdk
{
    class LightPlugin : public PluginBase
    {
    public:
        LightPlugin(GlobalFunc* global, LWBaseID base, LWAttributeID attrs_, LWItemID id) : PluginBase(global),
            m_item_id(id),
            m_base(base),
            m_attrs(attrs_)
        {}

        unsigned int flags() { return 0; }
        LWError new_frame(const LWFrameInfo* frameinfo, unsigned int* maxIlluminations) { return nullptr; }
        const LWPreviewLight* preview() { return nullptr; }

        size_t memory_size() { return 0; }
        unsigned int evaluate(LWMemChunk, const LWRay*, unsigned int, const LWDVector, unsigned int*) { return LWLIGHTSAMPLE_NO_SAMPLE; }
        unsigned int sample(const LWMemChunk, const LWRay*, const LWDVector, unsigned int, const LWDVector[], LWLightSample[]) { return 0; }
        void emission(const LWMemChunk, const LWRay*, const LWDVector, unsigned int, const LWBxDFSample[], LWLightEmission[]) {}
        int local_bounds(LWDVector, LWDVector) { return 0; }
        int world_bounds(LWDVector, LWDVector) { return 0; }
        int intersect(const LWRay*, LWShadingGeometry*) { return 0; }
        int power(LWDVector) { return 0; }
        int volume_segment(const LWMemChunk, const LWRay*, unsigned int, double, double, double*, double*) { return 0; }
        double volume_sample(const LWMemChunk, const LWRay*, double, double, double, double*) { return 0; }
        double volume_sample_pdf(const LWMemChunk, const LWRay*, double, double, double) { return 0; }
        int sample_emission(const double rndsample[4], LWTime fracTime, LWLightEmissionSample* le) { return 0; }
        unsigned int ambient(const LWRay*, unsigned int, const LWDVector, LWDVector) { return 0; }

    protected:
        LWItemID m_item_id;
        LWBaseID m_base;
        LWAttributeID m_attrs;
    };

    template<class T>
    class LightPluginAdaptor : public PluginRenderAdaptor<T>
    {
        std::string user_name;
        std::string plugin_name;

    public:

        LightPluginAdaptor(const char* plugin_name_, const char* user_name_, ServerTagInfo* additional = nullptr) :
            user_name(user_name_),
            plugin_name(plugin_name_)
        {
            PluginAdaptor::register_plugin(LWLIGHT_HCLASS, activate, plugin_name.c_str(), user_name.c_str(), additional);
        }

        static int activate(int version, GlobalFunc* global, void* local_, void* serverData)
        {
            LWLightHandler* local = (LWLightHandler*)local_;
            init_wrapper(global);
            PluginInstanceAdaptor<T>::activate(version, global, local->inst, serverData);
            PluginItemAdaptor<T>::activate(version, global, local->item, serverData);
            PluginRenderAdaptor<T>::activate(version, global, local->rend, serverData);
            local->inst->create = create;

            local->flags = flags;
            local->newFrame = new_frame;
            local->preview = preview;
            local->memorySize = memory_size;
            local->evaluate = evaluate;
            local->sample = sample;
            local->emission = emission;
            local->localBounds = local_bounds;
            local->worldBounds = world_bounds;
            local->intersect = intersect;
            local->power = power;
            local->volumeSegment = volume_segment;
            local->volumeSample = volume_sample;
            local->volumeSamplePdf = volume_sample_pdf;
            local->sampleEmission = sample_emission;
            local->ambient = ambient;

            return AFUNC_OK;
        }

        static LWInstance create(void* data, void* context, LWError* err)
        {
            LWContext* ctx = (LWContext*)context;
            LWAttributeID attrs = lwsdk::basefuncs->getAttributes(ctx->baseID);
            LWItemID id = const_cast<LWItemID>(ctx->context.context);
            return new T(PluginInstanceAdaptor<T>::global, ctx->baseID, attrs, id);
        }

        static unsigned int flags(LWInstance vinst)
        {
            assert(vinst);
            return ((T*)vinst)->flags();
        }

        static LWError new_frame(LWInstance vinst, const LWFrameInfo* frameinfo, unsigned int* maxIlluminations)
        {
            assert(vinst);
            return ((T*)vinst)->new_frame(frameinfo, maxIlluminations);
        }

        static const LWPreviewLight* preview(LWInstance vinst)
        {
            assert(vinst);
            return ((T*)vinst)->preview();
        }

        static size_t memory_size(LWInstance vinst)
        {
            assert(vinst);
            return ((T*)vinst)->memory_size();
        }

        static unsigned int evaluate(LWInstance vinst, LWMemChunk memory, const LWRay* ray, unsigned int flags, const LWDVector p, unsigned int* samples)
        {
            assert(vinst);
            return ((T*)vinst)->evaluate(memory, ray, flags, p, samples);
        }

        static unsigned int sample(LWInstance vinst, const LWMemChunk memory, const LWRay* ray, const LWDVector p, unsigned int maxSamples, const LWDVector samples[], LWLightSample lightsamples[])
        {
            assert(vinst);
            return ((T*)vinst)->sample(memory, ray, p, maxSamples, samples, lightsamples);
        }

        static void emission(LWInstance vinst, const LWMemChunk memory, const LWRay* ray, const LWDVector p, unsigned int maxSamples, const LWBxDFSample bxdfsamples[], LWLightEmission emissions[])
        {
            assert(vinst);
            ((T*)vinst)->emission(memory, ray, p, maxSamples, bxdfsamples, emissions);
        }

        static int local_bounds(LWInstance vinst, LWDVector min, LWDVector max)
        {
            assert(vinst);
            return ((T*)vinst)->local_bounds(min, max);
        }

        static int world_bounds(LWInstance vinst, LWDVector min, LWDVector max)
        {
            assert(vinst);
            return ((T*)vinst)->world_bounds(min, max);
        }

        static int intersect(LWInstance vinst, const LWRay* ray, LWShadingGeometry* is)
        {
            assert(vinst);
            return ((T*)vinst)->intersect(ray, is);
        }

        static int power(LWInstance vinst, LWDVector power_)
        {
            assert(vinst);
            return ((T*)vinst)->power(power_);
        }

        static int volume_segment(LWInstance vinst, const LWMemChunk memory, const LWRay* ray, unsigned int flags, double tMin, double tMax, double* segmin, double* segmax)
        {
            assert(vinst);
            return ((T*)vinst)->volume_segment(memory, ray, flags, tMin, tMax, segmin, segmax);
        }

        static double volume_sample(LWInstance vinst, const LWMemChunk memory, const LWRay* ray, double segmin, double segmax, double ru, double *dist)
        {
            assert(vinst);
            return ((T*)vinst)->volume_sample(memory, ray, segmin, segmax, ru, dist);
        }

        static double volume_sample_pdf(LWInstance vinst, const LWMemChunk memory, const LWRay* ray, double segmin, double segmax, double dist)
        {
            assert(vinst);
            return ((T*)vinst)->volume_sample_pdf(memory, ray, segmin, segmax, dist);
        }

        static int sample_emission(LWInstance vinst, const double rndsample[4], LWTime fracTime, LWLightEmissionSample* le)
        {
            assert(vinst);
            return ((T*)vinst)->sample_emission(rndsample, fracTime, le);
        }

        static unsigned int ambient(LWInstance vinst, const LWRay* ray, unsigned int flags, const LWDVector p, LWDVector amb)
        {
            assert(vinst);
            return ((T*)vinst)->ambient(ray, flags, p, amb);
        }
    };
}
#endif // LWCPP_LIGHT_H