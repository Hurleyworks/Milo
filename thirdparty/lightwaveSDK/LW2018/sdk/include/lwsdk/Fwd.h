// Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
// 
// This file contains confidential and proprietary information of NewTek, Inc.,
// and is subject to the terms of the LightWave End User License Agreement (EULA).

#ifndef LWCPP_FWD_H
#define LWCPP_FWD_H

namespace lwsdk
{

    template <int N, class Real>
    class Vector;

    /// @short 2-component mathematical vector
    typedef Vector<2, int> Vector2i;

    /// @short 3-component mathematical vector
    typedef Vector<3, int> Vector3i;

    /// @short 4-component mathematical vector
    typedef Vector<4, int> Vector4i;

    /// @short 2-component mathematical vector, single precision
    typedef Vector<2, float> Vector2f;

    /// @short 2-component mathematical vector, double precision
    typedef Vector<2, double> Vector2d;

    /// @short 3-component mathematical vector, single precision
    typedef Vector<3, float> Vector3f;

    /// @short 3-component mathematical vector, double precision
    typedef Vector<3, double> Vector3d;

    /// @short 4-component mathematical vector, single precision
    typedef Vector<4, float> Vector4f;

    /// @short 4-component mathematical vector, double precision
    typedef Vector<4, double> Vector4d;

    typedef Vector<3, unsigned char> Vector3ub;
    typedef Vector<4, unsigned char> Vector4ub;

    template <class T>
    class Rectangle;

    typedef Rectangle<double> RectangleD;
    typedef Rectangle<float> RectangleF;
    typedef Rectangle<int> RectangleI;
}

#endif // LWCPP_FWD_H