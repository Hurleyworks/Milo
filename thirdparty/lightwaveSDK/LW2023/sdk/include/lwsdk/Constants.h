#ifndef INCLUDE_LWSDK_MATH_CONSTANTS_H
#define INCLUDE_LWSDK_MATH_CONSTANTS_H

/**
 *  @file constants.h
 *  @brief Math constants.
 *
 *  © Copyright 2009 Newtek, Inc. and its licensors. All rights reserved.
 *  This file contains confidential and proprietary information of Newtek,
 *  Inc., and is subject to the terms of the CORE end user license agreement
 *  (or EULA).
 */

#include <lwsdk/Vector.h>

namespace lwsdk
{
#ifndef PI
    const double PI         = 3.14159265358979323846;
#endif
    const double PId        = PI;
    const float  PIf        = static_cast<float>(PI);

#ifndef HALFPI
    const double HALFPI     = 3.14159265358979323846 / 2.0;
#endif
    const double HALFPId    = HALFPI;
    const float  HALFPIf    = static_cast<float>(HALFPI);

#ifndef TWOPI
    const double TWOPI      = 2.0 * PI;
#endif
    const double TWOPId     = TWOPI;
    const float  TWOPIf     = static_cast<float>(TWOPI);

#ifndef INVPI
    const double INVPI      = 1.0 / PI;
#endif
    const double INVPId     = INVPI;
    const float  INVPIf     = static_cast<float>(INVPI);

#ifndef INVTWOPI
    const double INVTWOPI   = 1.0 / TWOPI;
#endif
    const double INVTWOPId  = INVTWOPI;
    const float  INVTWOPIf  = static_cast<float>(INVTWOPI);

    const double SQRT2      = 1.41421356237309504880;
    const double SQRT2d     = SQRT2;
    const float  SQRT2f     = static_cast<float>(SQRT2);

    //degrees to radians
    const double DTOR       = 0.017453292519943295769;
    const double DTORd      = DTOR;
    const float  DTORf      = static_cast<float>(DTOR);

    //radians to degrees
    const double RTOD       = 57.29577951308232087679;
    const double RTODd      = RTOD;
    const float  RTODf      = static_cast<float>(RTOD);

    template <class Real>
    Vector<3, Real> x_axis()
    {
        return Vector<3, Real>(1, 0, 0);
    }

    template <class Real>
    Vector<3, Real> y_axis()
    {
        return Vector<3, Real>(0, 1, 0);
    }

    template <class Real>
    Vector<3, Real> z_axis()
    {
        return Vector<3, Real>(0, 0, 1);
    }
}

#endif