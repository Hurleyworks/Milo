#ifndef LWSDK_VECTOR_OPERATIONS_INCLUDED
#define LWSDK_VECTOR_OPERATIONS_INCLUDED

/**
 *  @file VectorOperations.h
 *  @brief Functions operating on Vectors.
 *
 *  © Copyright 2009 Newtek, Inc. and its licensors. All rights reserved.
 *  This file contains confidential and proprietary information of Newtek,
 *  Inc., and is subject to the terms of the CORE end user license agreement
 *  (or EULA).
 */

#ifndef NOMINMAX
#define NOMINMAX // omit min/max macros in windef.h
#endif

#include <cmath>
#include <algorithm>
#include <limits>
#include <lwsdk/Vector.h>
#include <lwsdk/Constants.h>
#include <lwsdk/Rsqrt.h>

namespace lwsdk
{
    //*****************************************************************************
    //--------------------------- comparison operators ----------------------------

    /// true if all components are equal to a scalar
    template <int N, class Real>
    bool operator==(const Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
        {
            if (lvec[j] != scalar)
                return false; // found a component that's not equal
        }
        return true; // all components equal
    }

    /// true if any components are not equal to a scalar
    template <int N, class Real>
    bool operator!=(const Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
        {
            if (lvec[j] != scalar)
                return true; // found a component that's not equal
        }
        return false; // all components equal
    }

    /// true if any components are less than a scalar
    template <int N, class Real>
    bool operator< (const Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
        {
            if (lvec[j] < scalar)
                return true;
        }
        return false; // all components greater than or equal to scalar
    }

    /// True if all components are less than or equal to a scalar
    template <int N, class Real>
    bool operator<=(const Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
        {
            if (lvec[j] > scalar)
                return false;
        }
        return true; // all components less than or equal to scalar
    }

    /// true if any components are greater than a scalar
    template <int N, class Real>
    bool operator>(const Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
        {
            if (lvec[j] > scalar)
                return true;
        }
        return false; // all components less than scalar
    }

    /// True if all components are greater than or equal to a scalar
    template <int N, class Real>
    bool operator>=(const Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
        {
            if (lvec[j] < scalar)
                return false;
        }
        return true;
    }

    //*****************************************************************************
    //--------------------------- arithmetic operators ----------------------------

    template <int N, class Real>
    Vector<N,Real> operator+ (const Vector<N, Real>& lvec, const Vector<N, Real>& rvec)
    {
        Vector<N,Real> kSum;
        for (int i = 0; i < N; ++i)
            kSum[i] = lvec[i] + rvec[i];
        return kSum;
    }

    template <int N, class Real>
    Vector<N,Real> operator+ (const Vector<N, Real>& lvec, Real scalar)
    {
        Vector<N,Real> kSum;
        for (int i = 0; i < N; ++i)
            kSum[i] = lvec[i] + scalar;
        return kSum;
    }

    template <int N, class Real>
    Vector<N,Real> operator- (const Vector<N, Real>& lvec, const Vector<N, Real>& rvec)
    {
        Vector<N,Real> kDiff;
        for (int i = 0; i < N; ++i)
            kDiff[i] = lvec[i] - rvec[i];
        return kDiff;
    }

    template <int N, class Real>
    Vector<N,Real> operator- (const Vector<N, Real>& lvec, Real scalar)
    {
        Vector<N,Real> kDiff;
        for (int i = 0; i < N; ++i)
            kDiff[i] = lvec[i] - scalar;
        return kDiff;
    }

    template <int N, class Real>
    Vector<N,Real> operator* (const Vector<N, Real>& lvec, Real scalar)
    {
        Vector<N,Real> kprod;
        for (int i = 0; i < N; ++i)
            kprod[i] = scalar * lvec[i];
        return kprod;
    }

    template <int N, class Real>
    Vector<N,Real> operator/ (const Vector<N, Real>& lvec, Real scalar)
    {
        Vector<N,Real> quot;
        if ( scalar != static_cast<Real>(0.0) )
        {
            for (int i = 0; i < N; ++i)
                quot[i] = lvec[i] / scalar;
        }
        else
        {
            for (int i = 0; i < N; ++i)
                quot[i] = std::numeric_limits<Real>::max();
        }
        return quot;
    }

    template <int N> static
    Vector<N,float> operator/ (const Vector<N, float>& lvec, float scalar)
    {
        Vector<N,float> quot;
        if ( scalar != static_cast<float>(0.0) )
        {
            float finvscalar = (static_cast<float>(1.0))/scalar;
            for (int i = 0; i < N; ++i)
                quot[i] = finvscalar * lvec[i];
        }
        else
        {
            for (int i = 0; i < N; ++i)
                quot[i] = std::numeric_limits<float>::max();
        }
        return quot;
    }

    template <int N> static
    Vector<N,double> operator/ (const Vector<N, double>& lvec, double scalar)
    {
        Vector<N,double> quot;
        if ( scalar != static_cast<double>(0.0) )
        {
            double finvscalar = (static_cast<double>(1.0))/scalar;
            for (int i = 0; i < N; ++i)
                quot[i] = finvscalar * lvec[i];
        }
        else
        {
            for (int i = 0; i < N; ++i)
                quot[i] = std::numeric_limits<double>::max();
        }
        return quot;
    }

    template <int N, class Real>
    Vector<N,Real> operator/ (const Vector<N, Real>& lvec, const Vector<N,Real>& rvec)
    {
        Vector<N,Real> quot;
        int i;

        for (i = 0; i < N; ++i)
        {
            if( rvec[i] != static_cast<Real>(0.0) )
                quot[i] = lvec[i] / rvec[i];
            else if(lvec[i] > static_cast<Real>(0.0))
                quot[i] = std::numeric_limits<Real>::max();
            else if(lvec[i] < static_cast<Real>(0.0))
                quot[i] = -std::numeric_limits<Real>::max();
            else
                quot[i] = static_cast<Real>(0.0);
        }

        return quot;
    }

    template <int N, class Real>
    Vector<N,Real> operator- (const Vector<N, Real>& lvec)
    {
        Vector<N,Real> neg;
        for (int i = 0; i < N; ++i)
            neg[i] = -lvec[i];
        return neg;
    }

    template <int N, class Real>
    Vector<N,Real> operator* (Real scalar, const Vector<N,Real>& rvec)
    {
        Vector<N,Real> prod;
        for (int i = 0; i < N; ++i)
            prod[i] = scalar * rvec[i];
        return prod;
    }

    template <int N, class Real>
    Vector<N,Real> operator* (const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        Vector<N,Real> prod;
        for (int i = 0; i < N; ++i)
            prod[i] = lvec[i] * rvec[i];
        return prod;
    }

    template <int N, class Real>
    Vector<N, Real>& operator+= (Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        for (int j=0; j < N; ++j)
            lvec[j] += rvec[j];
        return lvec;
    }

    template <int N, class Real>
    Vector<N, Real>& operator+= (Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
            lvec[j] += scalar;
        return lvec;
    }

    template <int N, class Real>
    Vector<N, Real>& operator+= (Vector<N,Real>& lvec, const Real* rvec)
    {
        for (int j=0; j < N; ++j)
            lvec[j] += rvec[j];
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator-= (Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        for (int j=0; j < N; ++j)
            lvec[j] -= rvec[j];
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator-= (Vector<N,Real>& lvec, const Real* rvec)
    {
        for (int j=0; j < N; ++j)
            lvec[j] -= rvec[j];
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator-= (Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
            lvec[j] -= scalar;
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator*= (Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        for (int j=0; j < N; ++j)
            lvec[j] *= rvec[j];
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator*= (Vector<N,Real>& lvec, const Real* rvec)
    {
        for (int j=0; j < N; ++j)
            lvec[j] *= rvec[j];
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator*= (Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
            lvec[j] *= scalar;
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator/= (Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        for (int j=0; j < N; ++j)
            lvec[j] /= rvec[j];
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator/= (Vector<N,Real>& lvec, const Real* rvec)
    {
        for (int j=0; j < N; ++j)
            lvec[j] /= rvec[j];
        return lvec;
    }

    template <int N, class Real>
    Vector<N,Real>& operator/= (Vector<N,Real>& lvec, Real scalar)
    {
        for (int j=0; j < N; ++j)
            lvec[j] /= scalar;
        return lvec;
    }

    //*****************************************************************************
    //---------------------------- vector operations ------------------------------

    template <int N, class T>
    Vector<N + 1, T> expand(const Vector<N, T>& lhs, const T& val)
    {
        Vector<N + 1, T> vec(lhs);
        vec[N] = val;
        return vec;
    }

    template <int N, class Real>
    Real dot(const Vector<N, Real>& lvec, const Vector<N, Real>& rvec)
    {
        Real dot = static_cast<Real>(0.0);
        for (int i = 0; i < N; ++i)
            dot += lvec[i] * rvec[i];
        return dot;
    }

    template <class Real>
    Real vdot3(const Vector<4, Real>& lvec, const Vector<4, Real>& rvec)
    {
        Real dot = static_cast<Real>(0.0);
        for (int i = 0; i < 3; ++i)
            dot += lvec[i] * rvec[i];
        return dot;
    }

    template <int N, class Real>
    Real vector_distance_squared(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        Real sqrlen = static_cast<Real>(0);
        for (int i = 0; i < N; ++i)
        {
            Real diff = lvec[i] - rvec[i];
            sqrlen += diff * diff;
        }
        return sqrlen;
    }

    template <int N, class Real>
    Real vector_distance(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        return static_cast<Real>(std::sqrt(static_cast<double>(vector_distance_squared(lvec, rvec))));
    }

    template <int N, class Real>
    void to_array(Real* out, const Vector<N, Real>& vec)
    {
        for (int i = 0; i < N; ++i)
            out[i] = vec[i];
    }

    template <int N, class Real>
    void to_float_array(float* out, const Vector<N, Real>& vec)
    {
        for (int i = 0; i < N; ++i)
            out[i] = static_cast<float>(vec[i]);
    }

    template <int N, class Real>
    void to_double_array(double* out, const Vector<N, Real>& vec)
    {
        for (int i = 0; i < N; ++i)
            out[i] = static_cast<double>(vec[i]);
    }

    template <int N, class Real>
    Real angle(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        Real l_dot_r = dot(lvec, rvec);
        Real l = length(lvec) * length(rvec);
        if(l <= std::numeric_limits<Real>::epsilon())
            return 0;

        Real cos_angle = l_dot_r / l;
        cos_angle = std::max( std::min(cos_angle, (Real)1), (Real)-1);

        return std::acos(cos_angle);
    }

    /// @short 2-component vector cross product algorithm.
    /// @return a scalar.
    template <class Real>
    Real cross(const Vector<2, Real>& lvec, const Vector<2, Real>& rvec)
    {
        return lvec[0] * rvec[1] - rvec[0] * lvec[1];
    }

    /// @short 3-component vector cross product algorithm
    template <class Real>
    Vector<3, Real> cross(const Vector<3, Real>& lvec, const Vector<3, Real>& rvec)
    {
        return Vector<3, Real>(
            lvec[1] * rvec[2] - lvec[2] * rvec[1],
            lvec[2] * rvec[0] - lvec[0] * rvec[2],
            lvec[0] * rvec[1] - lvec[1] * rvec[0]
        );
    }

    /// @short 2-component vector cross product algorithm.
    /// @return a scalar.
    template <int N, class Real>
    Real vcross2(const Vector<N, Real>& lvec, const Vector<N, Real>& rvec)
    {
        return lvec[0] * rvec[1] - rvec[0] * lvec[1];
    }

    /// @short 3-component vector cross product algorithm
    template <int N, class Real>
    Vector<N, Real> vcross3(const Vector<N, Real>& lvec, const Vector<N, Real>& rvec)
    {
        return Vector<N, Real>(
            lvec[1] * rvec[2] - lvec[2] * rvec[1],
            lvec[2] * rvec[0] - lvec[0] * rvec[2],
            lvec[0] * rvec[1] - lvec[1] * rvec[0]
        );
    }

    template <int N, class Real>
    void clamp(Vector<N, Real>& v, Real low, Real high)
    {
        for (int i = 0; i < N; ++i)
        {
            v[i] = std::min(v[i], static_cast<Real>(high) );
            v[i] = std::max(v[i], static_cast<Real>(low) );
        }
    }

    template <int N, class Real>
    void clamp(Vector<N, Real>& v, const Vector<N, Real>& low, const Vector<N, Real>& high)
    {
        for(int i = 0; i < N; ++i)
        {
            v[i] = std::min(v[i], static_cast<Real>(high[i]));
            v[i] = std::max(v[i], static_cast<Real>(low[i]));
        }
    }

    template <int N, class Real>
    Vector<N, Real> clamped(Vector<N, Real> v, Real low, Real high)
    {
        clamp(v, low, high);
        return v;
    }

    template <int N, class Real>
    Vector<N, Real> clamped(Vector<N, Real> v, const Vector<N, Real>& low, const Vector<N, Real>& high)
    {
        clamp(v, low, high);
        return v;
    }

    template <int N, class Real>
    void floor(Vector<N, Real>& v)
    {
        for(int i = 0; i < N; ++i)
        {
            v[i] = std::floor(v[i]);
            v[i] = std::floor(v[i]);
        }
    }

    template <int N, class Real>
    Vector<N, Real> floored(Vector<N, Real> v)
    {
        floor(v);
        return v;
    }

    template <int N, class Real>
    void round(Vector<N, Real>& v)
    {
        for(int i = 0; i < N; ++i)
        {
            v[i] = std::round(v[i]);
            v[i] = std::round(v[i]);
        }
    }

    template <int N, class Real>
    Vector<N, Real> rounded(Vector<N, Real> v)
    {
        round(v);
        return v;
    }

    template <int N, class Real>
    void invert(Vector<N,Real>& v)
    {
        for (int i = 0; i < N; ++i)
        {
            // check for division by zero
            if (v[i] == 0.0)
                v[i] = std::numeric_limits<Real>::max();
            else
                v[i] = static_cast<Real>(1) / v[i];
        }
    }

    /// @ short Create a cartesian vector from homogeneous vector
    template <int N, class Real>
    Vector<N-1,Real> cart (const Vector<N,Real>& v)
    {
            Vector<N-1,Real> cartv;
            const Real w = v[N-1];
            if(w != 0)
            {
                const Real inv_w = static_cast<Real>(1) / w;
                for(int i = 0; i < N-1; ++i)
                    cartv[i] = v[i] * inv_w;
            }
            return cartv;
        }

    /// @ short Create a homogeneous vector from cartesion vector
    template <int N, class Real>
    Vector<N+1,Real> hom (const Vector<N,Real>& v)
    {
        Vector<N+1,Real> homv;
        for(int i = 0; i < N; ++i)
            homv[i] = v[i];
        homv[N] = static_cast<Real>(1);
        return homv;
    }

    /**
     *  @short Normalizes the given vector to length 1.0
     *
     *  @return the length of the vector before normalization.
     */
    template <int N, class Real>
    Real normalize (Vector<N,Real>& v)
    {
        Real len = squared_length(v);
        int i;

        if ( len > 0 )
        {
            len = std::sqrt(len);
            Real inv_len = (static_cast<Real>(1.0)) / len;
            for (i = 0; i < N; ++i)
                v[i] *= inv_len;
        }
        else
        {
            for (i = 0; i < N; ++i)
                v[i] = static_cast<Real>(0.0);
        }
        return len;
    }

    /**
     *  @short Normalizes the given vector to length 1.0
     *
     *  @return the length of the vector before normalization.
     */
    template <int N, class Real>
    Real normalize3 (Vector<N,Real>& v)
    {
        Real len = squared_length3(v);
        int i;

        if ( len > 0 )
        {
            len = std::sqrt(len);
            Real inv_len = (static_cast<Real>(1.0)) / len;
            for (i = 0; i < 3; ++i)
                v[i] *= inv_len;
        }
        else
        {
            for (i = 0; i < 3; ++i)
                v[i] = static_cast<Real>(0.0);
        }
        return len;
    }

    /// @return the perpendicular vector to the one specified.
    template <class Real>
    Vector<3, Real> perpendicular(const Vector<3, Real>& v)
    {
        return Vector<3, Real>(v[1] - v[2], v[2] - v[0], v[0] - v[1]);
    }

    template <int N>
    void fast_normalize(Vector<N, float>& v)
    {
        float sqrlen = v[0] * v[0];
        for (int i=1; i < N; ++i)
            sqrlen += v[i] * v[i];

        const float rlen = fast_rsqrt(sqrlen);
        for (int i=0; i < N; ++i)
	        v[i] *= rlen;
    }

    template <> inline
    void fast_normalize<3>(Vector<3, float>& v)
    {
        const float rlen = fast_rsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] *= rlen;
        v[1] *= rlen;
        v[2] *= rlen;
    }

    template <class Vec>
    void fast_normalize3(Vec& v)
    {
        const float rlen = fast_rsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] *= rlen;
        v[1] *= rlen;
        v[2] *= rlen;
    }

    /// @return a normalized version of the unit vector.
    template <int N, class Real>
    Vector<N, Real> unit_vector(const Vector<N, Real>& vec)
    {
        Vector<N, Real> normalized_vec = vec;
        normalize(normalized_vec);
        return normalized_vec;
    }

    template <int N, class Real>
    Real length (const Vector<N,Real>& v)
    {
        Real sqrlen = static_cast<Real>(0.0);
        for (int i = 0; i < N; ++i)
            sqrlen += v[i]*v[i];
        return static_cast<Real>(std::sqrt(sqrlen));
    }

    template <int N, class Real>
    Real length3 (const Vector<N,Real>& v)
    {
        Real sqrlen = static_cast<Real>(0.0);
        for (int i = 0; i < 3; ++i)
            sqrlen += v[i]*v[i];
        return static_cast<Real>(sqrt(sqrlen));
    }

    template <int N, class Real>
    Real squared_length (const Vector<N,Real>& v)
    {
        Real sqrlen = static_cast<Real>(0.0);
        for (int i = 0; i < N; ++i)
            sqrlen += v[i]*v[i];
        return static_cast<Real>(sqrlen);
    }

    template <int N, class Real>
    Real squared_length3(const Vector<N, Real>& v)
    {
        Real sqrlen = static_cast<Real>(0.0);
        for(int i = 0; i < 3; ++i)
            sqrlen += v[i] * v[i];
        return static_cast<Real>(sqrlen);
    }

    template <int N, class T>
    Vector<N, T> vmin(const Vector<N, T>& lhs, const Vector<N, T>& rhs)
    {
        Vector<N, T> min_vec(lhs);
        for (int j=0; j < 3; ++j)
        {
            if (rhs[j] < min_vec[j])
                min_vec[j] = rhs[j];
        }
        return min_vec;
    }

    template <int N, class T>
    Vector<N, T> vmax(const Vector<N, T>& lhs, const Vector<N, T>& rhs)
    {
        Vector<N, T> max_vec(lhs);
        for (int j=0; j < 3; ++j)
        {
            if (rhs[j] > max_vec[j])
                max_vec[j] = rhs[j];
        }
        return max_vec;
    }

    template <int N, class T>
    Vector<N, T> vmin(T lhs, const Vector<N, T>& rhs)
    {
        Vector<N, T> min_vec(lhs);
        for (int j = 0; j < 3; ++j)
        {
            if (rhs[j] < min_vec[j])
                min_vec[j] = rhs[j];
        }
        return min_vec;
    }

    template <int N, class T>
    Vector<N, T> vmax(T lhs, const Vector<N, T>& rhs)
    {
        Vector<N, T> max_vec(lhs);
        for (int j = 0; j < 3; ++j)
        {
            if (rhs[j] > max_vec[j])
                max_vec[j] = rhs[j];
        }
        return max_vec;
    }

    template <int N, class T>
    Vector<N, T> exp(const Vector<N, T> &v)
    {
        Vector<N, T> r;
        for (int i = 0; i < N; ++i)
            r[i] = std::exp(v[i]);
        return r;
    }

    template <class Real>
    Real hpb_to_axis_angle(const Vector<3, Real>& hpb, Vector<3, Real>& axis)
    {
        Real ch = cos(hpb[0] / 2);
        Real sh = sin(hpb[0] / 2);
        Real cp = cos(hpb[1] / 2);
        Real sp = sin(hpb[1] / 2);
        Real cb = cos(hpb[2] / 2);
        Real sb = sin(hpb[2] / 2);

        axis[0] = ch*sp*cb + sh*cp*sb;
        axis[1] = sh*cp*cb - ch*sp*sb;
        axis[2] = ch*cp*sb - sh*sp*cb;
        Real l = normalize(axis);
        if(l < std::numeric_limits<float>::epsilon())
        {
            return 0;
        }
        else
        {
            Real angle_a = ch*cp*cb + sh*sp*sb;
            if(angle_a < -1)
                angle_a = -1;
            else if(angle_a > 1)
                angle_a = 1;
            return 2 * acos(angle_a);
        }
    }

    template <class Real>
    Vector<3, Real> axis_angle_to_hpb(const Vector<3, Real>& axis, Real angle)
    {
        Vector<3, Real> hpb;

        Real s = sin(angle);
        Real c = cos(angle);
        Real t = 1 - c;

        hpb[0] = atan2(axis[0] * axis[2] * t + axis[1] * s, 1 - (axis[0] * axis[0] + axis[1] * axis[1]) * t);

        Real sp = axis[0] * s - axis[1] * axis[2] * t;
        if(sp < -1) sp = -1;
        else if(sp > 1) sp = 1;
        hpb[1] = asin(sp);

        hpb[2] = atan2(axis[0] * axis[1] * t + axis[2] * s, 1 - (axis[0] * axis[0] + axis[2] * axis[2]) * t);

        return hpb;
    }

    template <class Real>
    Vector<2, Real> vector_to_hp(const Vector<3, Real>& v)
    {
        Vector<2, Real> ret;
        if(v[0] == 0.0 && v[2] == 0.0)
        {
            ret[0] = 0.0;
            if(v[1] != 0.0)
                ret[1] = (v[1] < 0.0) ? HALFPI : -HALFPI;
            else
                ret[1] = 0.0;
        } else {
            if(v[2] == 0.0)
                ret[0] = (v[0] < 0.0) ? -HALFPI : HALFPI;
            else
                ret[0] = atan2(v[0], v[2]);
            Real hyp = std::sqrt(v[0] * v[0] + v[2] * v[2]);
            if(hyp == 0.0)
                ret[1] = (v[1] < 0.0f) ? HALFPI : -HALFPI;
            else
                ret[1] = -atan2(v[1], hyp);
        }
        return ret;
    }

    /// @return true if the vectors are close enough to being equal
    /// to each other.
    template <int N, class Real>
    bool close_enough(const Vector<N, Real>& v1, const Vector<N, Real>& v2, Real epsilon = static_cast<Real>(0.000001f))
    {
        return vector_distance(v1, v2) <= epsilon;
    }

    /// @return the normal given three vectors to define a plane.
    template <int N, class T>
    Vector<N, T> normal(const Vector<N, T>& v1, const Vector<N, T>& v2, const Vector<N, T>& v3)
    {
        Vector<N, T> diff1 = v2 - v1;
        Vector<N, T> diff2 = v3 - v1;
        Vector<N, T> norm = vcross3(diff1, diff2);
        normalize(norm);
        return norm;
    }

        /// @return a linearly interpolated vector.
    template <int N, class T>
    Vector<N, T> lerp(T x, const Vector<N, T>& v1, const Vector<N, T>& v2)
    {
        Vector<N, T> result;
        T ix = static_cast<T>(1) - x;
        for(auto i = 0; i < N; ++i)
            result[i] = ix * v1[i] + x * v2[i];
        return result;
    }

    template<typename T>
    inline T CosTheta(const lwsdk::Vector<3, T> &w) { return w[2]; }

    template<typename T>
    inline T Cos2Theta(const lwsdk::Vector<3, T> &w) { return w[2] * w[2]; }

    template<typename T>
    inline T Sin2Theta(const lwsdk::Vector<3, T> &w) { return std::max(static_cast<T>(0), static_cast<T>(1) - Cos2Theta(w)); }

    template<typename T>
    inline T SinTheta(const lwsdk::Vector<3, T> &w) { return sqrt(Sin2Theta(w)); }

    template<typename T>
    inline T CosPhi(const lwsdk::Vector<3, T> &w)
    {
        T sinTheta = SinTheta(w);
        return (sinTheta == static_cast<T>(0)) ? static_cast<T>(1) : std::min(std::max(w[0] / sinTheta, static_cast<T>(-1)), static_cast<T>(1));
    }

    template<typename T>
    inline T SinPhi(const lwsdk::Vector<3, T> &w)
    {
        T sinTheta = SinTheta(w);
        return (sinTheta == static_cast<T>(0)) ? static_cast<T>(0) : std::min(std::max(w[1] / sinTheta, static_cast<T>(-1)), static_cast<T>(1));
    }

    template<typename T>
    inline void CosSinPhi(const lwsdk::Vector<3, T> &w, T o[2])
    {
        T sinTheta = SinTheta(w);
        o[0] = (sinTheta == static_cast<T>(0)) ? static_cast<T>(0) : std::min(std::max(w[0] / sinTheta, static_cast<T>(-1)), static_cast<T>(1));
        o[1] = (sinTheta == static_cast<T>(0)) ? static_cast<T>(0) : std::min(std::max(w[1] / sinTheta, static_cast<T>(-1)), static_cast<T>(1));
    }

    template<typename T>
    inline T TanTheta(const lwsdk::Vector<3, T> &w) { return SinTheta(w) / CosTheta(w); }

    template<typename T>
    inline T Cos2Phi(const lwsdk::Vector<3, T> &w) { return CosPhi(w) * CosPhi(w); }

    template<typename T>
    inline T Sin2Phi(const lwsdk::Vector<3, T> &w) { return SinPhi(w) * SinPhi(w); }

    template<typename T>
    inline T Tan2Theta(const lwsdk::Vector<3, T> &w) { return Sin2Theta(w) / Cos2Theta(w); }

    template<typename T>
    inline T AbsCosTheta(const lwsdk::Vector<3, T> &w) { return fabs(w[2]); }

    template<typename T>
    inline T AbsDot(const lwsdk::Vector<3, T> &v1, const lwsdk::Vector<3, T> &v2) { return fabs(dot(v1, v2)); }

    //*****************************************************************************
}

#endif