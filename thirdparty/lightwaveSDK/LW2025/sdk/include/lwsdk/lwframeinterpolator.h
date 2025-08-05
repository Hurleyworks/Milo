//
// Helper code to make it easy to interpolate values
//

#include <lwtypes.h>
#include <lwmath.h>

#include <vector>

using FrameInterpolatorID = void *;
#define FRAMEINTERPOLATOR_PARAMFLAG_NORMALIZE (1 << 0)

namespace FrameInterpolator
{

template <typename T> static bool vnormalize(T *vec)
{
    T x = VLEN(vec);
    if (x <= 0.0)
        return false;
    x = 1.0 / x;
    VSCL(vec, x);
    return true;
}

static GlobalFunc *global = nullptr;
static LWItemInfo *iteminfo = nullptr;
static LWVParmFuncs *vparmf = nullptr;

struct FrameInterpolator
{
    LWFrameInfo frameinfo;
    unsigned int numparams;

    double timeperstep;

    struct Param
    {
        LWDVector val;
    };
    using Params = std::vector<Param>;
    Params params;
    using Flags = std::vector<unsigned int>;
    Flags flags;
};

FrameInterpolatorID Create(GlobalFunc *globalfunc, unsigned int numparams)
{
    if (numparams == 0)
        return nullptr;

    if (!global)
        global = globalfunc;

    if (!iteminfo)
        iteminfo = (LWItemInfo *)global(LWITEMINFO_GLOBAL, GFUSE_TRANSIENT);

    if (!vparmf)
        vparmf = (LWVParmFuncs *)global(LWVPARMFUNCS_GLOBAL, GFUSE_TRANSIENT);

    FrameInterpolator *frameinterpolator = new FrameInterpolator();
    if (frameinterpolator == nullptr)
        return nullptr;

    frameinterpolator->numparams = numparams;
    frameinterpolator->flags.resize(numparams, 0);

    return frameinterpolator;
}

void Destroy(FrameInterpolatorID frameinterpolatorid)
{
    FrameInterpolator *frameinterpolator = (FrameInterpolator *)frameinterpolatorid;

    delete frameinterpolator;
}

void NewFrame(FrameInterpolatorID frameinterpolatorid, const LWFrameInfo *frameinfo)
{
    FrameInterpolator *frameinterpolator = (FrameInterpolator *)frameinterpolatorid;
    if (!frameinterpolator)
        return;

    if (!frameinfo)
        return;

    frameinterpolator->frameinfo = *frameinfo;
    frameinterpolator->timeperstep = 0;
    if (frameinfo->motionSteps > 0)
        frameinterpolator->timeperstep = frameinfo->duration / frameinfo->motionSteps;
    FrameInterpolator::Param zero = {{0.0, 0.0, 0.0}};
    frameinterpolator->params.resize(frameinterpolator->numparams * (frameinfo->motionSteps + 1), zero);
}

void FillParamItemParam(FrameInterpolatorID frameinterpolatorid, unsigned int param, LWItemID itemid,
                        LWItemParam itemparam, unsigned int flags)
{
    FrameInterpolator *frameinterpolator = (FrameInterpolator *)frameinterpolatorid;
    if (!frameinterpolator)
        return;

    if (param >= frameinterpolator->numparams)
        return;

    for (unsigned int t = 0; t <= frameinterpolator->frameinfo.motionSteps; t++)
    {
        LWTime subtime = frameinterpolator->frameinfo.start + t * frameinterpolator->timeperstep;
        iteminfo->param(itemid, itemparam, subtime,
                        frameinterpolator->params[param + t * frameinterpolator->numparams].val);
    }
    frameinterpolator->flags[param] = flags;
}

void FillParamArray(FrameInterpolatorID frameinterpolatorid, unsigned int param, const LWDVector vals[],
                    unsigned int flags)
{
    FrameInterpolator *frameinterpolator = (FrameInterpolator *)frameinterpolatorid;
    if (!frameinterpolator)
        return;

    if (param >= frameinterpolator->numparams)
        return;

    for (unsigned int t = 0; t <= frameinterpolator->frameinfo.motionSteps; t++)
    {
        VCPY(frameinterpolator->params[param + t * frameinterpolator->numparams].val, vals[t]);
    }
    frameinterpolator->flags[param] = flags;
}

void FillParamFunc(FrameInterpolatorID frameinterpolatorid, unsigned int param,
                   void (*func)(unsigned int param, LWTime time, LWDVector val, void *userdata), void *userdata,
                   unsigned int flags)
{
    FrameInterpolator *frameinterpolator = (FrameInterpolator *)frameinterpolatorid;
    if (!frameinterpolator)
        return;

    if (param >= frameinterpolator->numparams)
        return;

    for (unsigned int t = 0; t <= frameinterpolator->frameinfo.motionSteps; t++)
    {
        LWTime subtime = frameinterpolator->frameinfo.start + t * frameinterpolator->timeperstep;
        func(param, subtime, frameinterpolator->params[param + t * frameinterpolator->numparams].val, userdata);
    }
    frameinterpolator->flags[param] = flags;
}

void FillParamVParm(FrameInterpolatorID frameinterpolatorid, unsigned int param, LWVParmID vparm, unsigned int flags)
{
    FrameInterpolator *frameinterpolator = (FrameInterpolator *)frameinterpolatorid;
    if (!frameinterpolator)
        return;

    if (param >= frameinterpolator->numparams)
        return;

    for (unsigned int t = 0; t <= frameinterpolator->frameinfo.motionSteps; t++)
    {
        LWTime subtime = frameinterpolator->frameinfo.start + t * frameinterpolator->timeperstep;
        vparmf->getVal(vparm, subtime, NULL, frameinterpolator->params[param + t * frameinterpolator->numparams].val);
    }
    frameinterpolator->flags[param] = flags;
}

void Interpolate(FrameInterpolatorID frameinterpolatorid, double fractime, LWDVector vals[])
{
    FrameInterpolator *frameinterpolator = (FrameInterpolator *)frameinterpolatorid;
    if (!frameinterpolator)
        return;

    if (frameinterpolator->frameinfo.motionSteps == 0)
    {
        const FrameInterpolator::Param *params = &(frameinterpolator->params[0]);
        for (unsigned int t = 0; t < frameinterpolator->numparams; t++)
        {
            const LWDVector &v = params[t].val;
            VCPY(vals[t], v);
            if (frameinterpolator->flags[t] & FRAMEINTERPOLATOR_PARAMFLAG_NORMALIZE)
                vnormalize(vals[t]);
        }
    }
    else
    {
        double stepd = fractime * frameinterpolator->frameinfo.motionSteps;
        int step = static_cast<int>(floor(stepd));
        double fract = stepd - step;

        if (step < 0)
        {
            step = 0;
            fract = 0;
        }
        else if (step >= (int)frameinterpolator->frameinfo.motionSteps)
        {
            step = frameinterpolator->frameinfo.motionSteps - 1;
            fract = 1;
        }

        const FrameInterpolator::Param *params_lower =
            &(frameinterpolator->params[step * frameinterpolator->numparams]);
        const FrameInterpolator::Param *params_upper =
            &(frameinterpolator->params[(step + 1) * frameinterpolator->numparams]);
        for (unsigned int t = 0; t < frameinterpolator->numparams; t++)
        {
            const LWDVector &v_lower = params_lower[t].val;
            const LWDVector &v_upper = params_upper[t].val;
            VCOMB(vals[t], (1 - fract), v_lower, fract, v_upper);
            if (frameinterpolator->flags[t] & FRAMEINTERPOLATOR_PARAMFLAG_NORMALIZE)
                vnormalize(vals[t]);
        }
    }
}

} /* FrameInterpolator */