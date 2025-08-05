// Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of NewTek, Inc.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_PRIMITIVE_H
#define LWCPP_PRIMITIVE_H

#include <lwprimitive.h>
#include <lwsdk/Plugin.h>
#include <lwsdk/PluginInterface.h>
#include <lwsdk/Wrapper.h>

#include <cassert>

namespace lwsdk
{
    class PrimitivePlugin : public PluginBase
    {
    public:
        PrimitivePlugin(GlobalFunc* global, LWItemID id) : PluginBase(global),
            item_id(id)
        {}

        unsigned int flags()
        {
            return 0;
        }

    protected:
        LWItemID item_id;
    };

    template<class T>
    class PrimitivePluginAdaptor : public PluginRenderAdaptor<T>
    {
    public:

        PrimitivePluginAdaptor(const char* plugin_name, const char* user_name, const ServerTagInfo* additional = NULL)
        {
            PluginAdaptor::register_plugin(LWPRIMITIVE_HCLASS, activate, plugin_name, user_name,additional);
        }

        static int activate(int version, GlobalFunc* global, void* local_, void* serverData)
        {
            LWPrimitiveHandler* local = (LWPrimitiveHandler*)local_;
            init_wrapper(global);
            PluginInstanceAdaptor<T>::activate(version, global, local->inst, serverData);
            PluginItemAdaptor<T>::activate(version, global, local->item, serverData);
            PluginRenderAdaptor<T>::activate(version, global, local->rend, serverData);

            local->inst->create         = create;
            local->flags                = flags;
            local->newFrame             = new_frame;
            local->bounds               = bounds;

            // Solids
            local->memorySize           = memory_size;
            local->intersect            = intersect;
            local->nearestPoint         = nearest_point;
            local->edgePixelRadius      = edge_pixel_radius;
            local->edgeWorldRadius      = edge_world_radius;
            local->surface              = surface;
            local->area                 = area;
            local->sample               = sample;
            local->pdf                  = pdf;

            // Volumes
            local->volumeSegment        = volume_segment;
            local->volumeExtinction     = volume_extinction;
            local->volumeSample         = volume_sample;
            local->volumePhase          = volume_phase;
            local->volumeSamplePhase    = volume_sample_phase;

            return AFUNC_OK;
        }

        // Primitive specific create function
        // Todo: Is the context of *all* plugin classes with item functions an item id?
        // If yes then they could share the same "create" function.
        static LWInstance create(void* data, void* ctx_in, LWError* err)
        {
            LWBaseContext* ctx = (LWBaseContext*)ctx_in;
            return new T(PluginInstanceAdaptor<T>::global, ctx->context);
        }

        static unsigned int flags(void *inst)
        {
            assert(inst);
            return ((T*)inst)->flags();
        }

        static LWError new_frame(void *inst, const LWFrameInfo* frameinfo)
        {
            assert(inst);
            return ((T*)inst)->new_frame(frameinfo);
        }

        static int bounds(void* inst, LWPrimitiveType ptype, LWDVector min, LWDVector max, LWPrimitiveCoordinateSys* coordSys)
        {
            assert(inst);
            return ((T*)inst)->bounds(ptype, min, max, coordSys);
        }

        // Solids
        static size_t memory_size(void* inst)
        {
            assert(inst);
            return ((T*)inst)->memory_size();
        }

        static int intersect(void* inst, const LWPrimitiveInstance* pinst, const LWRay* ray, LWShadingGeometry* is)
        {
            assert(inst);
            return ((T*)inst)->intersect(pinst, ray, is);
        }

        static int nearest_point(void* inst, const LWPrimitiveInstance* pinst, const LWRay* ray, LWShadingGeometry* is)
        {
            assert(inst);
            return ((T*)inst)->nearest_point(pinst, ray, is);
        }

        static double edge_pixel_radius(void* inst)
        {
            assert(inst);
            return ((T*)inst)->edge_pixel_radius();
        }

        static double edge_world_radius(void* inst)
        {
            assert(inst);
            return ((T*)inst)->edge_world_radius();
        }

        static LWSurfaceID surface(void* inst, const LWShadingGeometry* is)
        {
            assert(inst);
            return ((T*)inst)->surface(is);
        }

        static double area(void* inst, const LWPrimitiveInstance* pinst, LWTime fracTime)
        {
            assert(inst);
            return ((T*)inst)->area(pinst, fracTime);
        }

        static int sample(void* inst, const LWPrimitiveInstance* pinst, LWTime fracTime, const LWDVector randomSample, LWDVector p, LWDVector n)
        {
            assert(inst);
            return ((T*)inst)->sample(pinst, fracTime, randomSample, p, n);
        }

        static double pdf(void* inst, const LWPrimitiveInstance* pinst, const LWRay* ray, const LWShadingGeometry* is)
        {
            assert(inst);
            return ((T*)inst)->pdf(pinst, ray, is);
        }

        // Volumes
        //@param ray The ray tested against
        //@param tMax The maximum length of the ray to test
        //@param segmin pointer to the minimum distance
        //@param segmax pointer to the maximum distance
        //@return 1 if there is a segment along this ray
        static int volume_segment(void* inst, const LWPrimitiveInstance* pinst, const LWVolumeSpot* vs, double tMin, double tMax, LWVolumeSegmentFunc* segFunc)
        {
            assert(inst);
            return ((T*)inst)->volume_segment(pinst, vs, tMin, tMax, segFunc);

        }
        static int volume_extinction(void* inst, const LWPrimitiveInstance* pinst, const LWVolumeSpot* vspot, double length, double u, LWDVector extinction)
        {
            assert(inst);
            return ((T*)inst)->volume_extinction(pinst, vspot, length, u, extinction);
        }

        static int volume_sample(void* inst, const LWPrimitiveInstance* pinst, const LWVolumeSpot* vspot, LWDVector emission, LWDVector scattering, LWDVector absorption)
        {
            assert(inst);
            return ((T*)inst)->volume_sample(pinst, vspot, emission, scattering, absorption);
        }
        static double volume_phase(void* inst, const LWPrimitiveInstance* pinst, const LWVolumeSpot* vspot, const LWDVector wi, const LWDVector wo)
        {
            assert(inst);
            return ((T*)inst)->volume_phase(pinst, vspot, wi, wo);
        }
        static double volume_sample_phase(void* inst, const LWPrimitiveInstance* pinst, const LWVolumeSpot* vspot, const LWDVector sample, LWDVector wo)
        {
            assert(inst);
            return ((T*)inst)->volume_sample_phase(pinst, vspot, sample, wo);
        }
    };
}

#endif // LWCPP_PRIMITIVE_H