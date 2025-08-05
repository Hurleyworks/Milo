#ifndef LWSDK_RSQRT_H
#define LWSDK_RSQRT_H

/**
 *  @file rsqrt.h
 *  @brief reciprocal square root functions.
 *
 *  © Copyright 2009 Newtek, Inc. and its licensors. All rights reserved.
 *  This file contains confidential and proprietary information of Newtek,
 *  Inc., and is subject to the terms of the CORE end user license agreement
 *  (or EULA).
 */

#include <cmath>
#include <cassert>

namespace lwsdk
{
    /// @return 1 / sqrt(x)
	float fast_rsqrt(float x);

    /// @return 1 / sqrt(x)
    float precise_rsqrt(float x);

    // ----------------------------------------------------------------------
    // implementation details
    // ----------------------------------------------------------------------
    inline float fast_rsqrt(float x)
    {
		// Super evil, very fast code based on the fast inverse square
        // root magic number.
        assert(sizeof(unsigned int) == sizeof(float));
        union
        {
            float f;
            unsigned int u;
        } y = {x};
        y.u = 0x5F1FFFF9ul - (y.u >> 1);
        return 0.703952253f * y.f * (2.38924456f - x * y.f * y.f);
    }

	inline float precise_rsqrt(float x)
	{
		// Provides a more precise inverse square root.
		return 1.0f / sqrt(x);
	}
}

#endif