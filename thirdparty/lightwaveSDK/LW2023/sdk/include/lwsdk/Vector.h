#ifndef LWSDK_VECTOR_INCLUDED
#define LWSDK_VECTOR_INCLUDED

/**
 *  @file Vector.h
 *  @brief Vector class definition.
 *
 *  © Copyright 2009 Newtek, Inc. and its licensors. All rights reserved.
 *  This file contains confidential and proprietary information of Newtek,
 *  Inc., and is subject to the terms of the CORE end user license agreement
 *  (or EULA).
 */

#include <cassert>
#include <string.h>
#include <algorithm>
#include <lwsdk/Fwd.h>

namespace lwsdk
{
    /**
     *  @short A mathematical vector of size N
     *
     *  The common mathematical operators "+", "-", "*", "/", "=", "==" and "!=" are available.
     *
     *  Use similar to the single component operations:
     *
     *  Vector3f vec1(9.3, 3.4, 1.1);
     *  Vector3f vec2(4.3, 5.4, 5.5);
     *
     *  Vector3f vec3 = vec1 + vec2;
     *  vec1 = vec3 * vec1;
     *  if (vec3 == vec2)
     *      vec3 = vec1;
     */
    template <int N, class Real>
    class Vector
    {
    public:
        typedef Real value_type;
        typedef Real* iterator;
        typedef const Real* const_iterator;

        static const Vector zero;

        /// @short Constructors
        /// @{
        Vector();
        Vector(const Vector& rvec);
        explicit Vector(const Real* rarray);

        template <class OtherReal>
        explicit Vector(const OtherReal* rarray);

        template<int rN, class OtherReal>
        /*implicit*/ Vector(const Vector<rN, OtherReal>& rvec);

        explicit Vector(Real scalar);
        Vector(Real x, Real y);
        Vector(Real x, Real y, Real z);
        Vector(Real x, Real y, Real z, Real w);
        /// @}

        /// @short Access to vector components
        /// @{
        const Real& operator[](int i) const;
        Real& operator[](int i);
        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;
        /// @}

        Real* data();
        const Real* data() const;

        /// @short Assigns the values of a vector to another.
        Vector& operator=(const Vector& rkV);

        Vector& operator=(Real scalar);

        template <class OtherReal>
        Vector<N, Real>& operator=(const OtherReal* rarray );

        /// @short Copy vectors of different sizes
        template <int rN, class OtherReal>
        Vector<N,Real>& operator=(const Vector<rN,OtherReal>& rvec);

        Real maxComponent() const;

        /// @short Zeroes the vector components
        void set_zero();

        enum {num_elements = N};

    private:
        /// @short Storage for vector data
        Real m_tuple[N];
    };


    //*****************************************************************************
    //------------------------------ Constructors ---------------------------------
    template <int N, class Real>
    Vector<N,Real>::Vector()
    {
        for (int i = 0; i < N; ++i)
            m_tuple[i] = 0;
    }

    template <int N, class Real>
    Vector<N,Real>::Vector(const Real* rarray)
    {
        /// @todo check if the compiler unrolls this loop in the release compile,
        ///       it should when it is set to optimize for speed, as it is constant.
        for (int i = 0; i < N; ++i)
            m_tuple[i] = rarray[i];
    }

    template <int N, class Real>
    Vector<N,Real>::Vector(const Vector& rvec)
    {
        for (int i = 0; i < N; ++i)
        {
            m_tuple[i] = rvec[i];
        }
    }

    template <int N, class Real>
    Vector<N,Real>::Vector(Real scalar)
    {
        for (int i = 0; i < N; ++i)
        {
            m_tuple[i] = scalar;
        }
    }

    template <int N, class Real>
    Vector<N,Real>::Vector(Real x, Real y)
    {
#ifdef VECTOR_RANGE_DEBUG
        assert(N >= 2);
#endif
        m_tuple[0] = x;
        m_tuple[1] = y;
        for (unsigned int j=2; j < N; ++j)
            m_tuple[j] = 0;
    }

    template <int N, class Real>
    Vector<N,Real>::Vector(Real x, Real y, Real z)
    {
#ifdef VECTOR_RANGE_DEBUG
        assert(N >= 3);
#endif
        m_tuple[0] = x;
        m_tuple[1] = y;
        m_tuple[2] = z;
        for (unsigned int j=3; j < N; ++j)
            m_tuple[j] = 0;
    }

    template <int N, class Real>
    Vector<N,Real>::Vector(Real x, Real y, Real z, Real w)
    {
#ifdef VECTOR_RANGE_DEBUG
        assert(N >= 4);
#endif
        m_tuple[0] = x;
        m_tuple[1] = y;
        m_tuple[2] = z;
        m_tuple[3] = w;
        for (unsigned int j=4; j < N; ++j)
            m_tuple[j] = 0;
    }

    template <int N, class Real>
    template <int OtherN, class OtherReal>
    Vector<N,Real>::Vector(const Vector<OtherN, OtherReal>& rvec)
    {
        enum {SmallN = OtherN >= N ? N: OtherN};
        for (int i = 0; i < SmallN; ++i)
            m_tuple[i] = static_cast<Real>(rvec[i]);
        for (int i = SmallN; i < N; ++i)
            m_tuple[i] = static_cast<Real>(0);
    }

    template <int N, class Real>
    template <class OtherReal>
    Vector<N,Real>::Vector(const OtherReal* rarray)
    {
        for (int i = 0; i < N; ++i)
            m_tuple[i] = static_cast<Real>(rarray[i]);
    }

    //*****************************************************************************
    //------------------------------- constants -----------------------------------

    namespace detail
    {
        template <int N, class Real>
        Vector<N, Real> compute_zero_vector(void)
        {
            Vector<N, Real> v;
            memset(&v[0], 0, N*sizeof(Real));
            return v;
        }
    }

    template <int N, class Real>
    const Vector<N, Real> Vector<N, Real>::zero = detail::compute_zero_vector<N, Real>();

    //*****************************************************************************
    //---------------------------- access functions -------------------------------
    template <int N, class Real>
    const Real& Vector<N,Real>::operator[](int i) const
    {
#ifdef VECTOR_RANGE_DEBUG
        assert( 0 <= i && i < N && "Out of bounds!");
#endif
        return m_tuple[i];
    }

    template <int N, class Real>
    Real& Vector<N,Real>::operator[](int i)
    {
#ifdef VECTOR_RANGE_DEBUG
        assert( 0 <= i && i < N && "Out of bounds!");
#endif
        return m_tuple[i];
    }

    template <int N, class Real>
    typename Vector<N, Real>::iterator Vector<N, Real>::begin()
    {
        return m_tuple;
    }

    template <int N, class Real>
    typename Vector<N, Real>::iterator Vector<N, Real>::end()
    {
        return m_tuple + N;
    }

    template <int N, class Real>
    typename Vector<N, Real>::const_iterator Vector<N, Real>::begin() const
    {
        return m_tuple;
    }

    template <int N, class Real>
    typename Vector<N, Real>::const_iterator Vector<N, Real>::end() const
    {
        return m_tuple + N;
    }

    template <int N, class Real>
    Real* Vector<N, Real>::data()
    {
        return m_tuple;
    }

    template <int N, class Real>
    const Real* Vector<N, Real>::data() const
    {
        return m_tuple;
    }

    //*****************************************************************************
    //-------------------------- assignment operator ------------------------------

    template <int N, class Real>
    Vector<N,Real>& Vector<N,Real>::operator=(const Vector& rvec)
    {
        for (int i = 0; i < N; ++i)
        {
            m_tuple[i] = rvec[i];
        }
        return *this;
    }

    template <int N, class Real>
    template <class OtherReal>
    Vector<N,Real>& Vector<N,Real>::operator=(const OtherReal* rarray)
    {
        for (int i = 0; i < N; ++i)
            m_tuple[i] = static_cast<Real>(rarray[i]);
        return *this;
    }

    template <int N, class Real>
    template <int rN, class OtherReal>
    Vector<N,Real>& Vector<N,Real>::operator=(const Vector<rN,OtherReal>& rvec)
    {
        int n = (rN >= N) ? N : rN;
        for (int i = 0; i < n; ++i)
            m_tuple[i] = static_cast<Real>(rvec[i]);
        for (int i = n; i < N; ++i)
            m_tuple[i] = static_cast<Real>(0);
        return *this;
    }

    template <int N, class Real>
    Vector<N,Real>& Vector<N,Real>::operator=(Real scalar)
    {
        for (int i = 0; i < N; ++i)
        {
            m_tuple[i] = scalar;
        }
        return *this;
    }

    //*****************************************************************************
    //---------------------------- vector operations ------------------------------

    template <int N, class Real>
    void Vector<N, Real>::set_zero()
    {
        for (int i = 0; i < N; ++i)
            m_tuple[i] = 0;
    }

    template <class OtherVectorType, int N, class Real>
    OtherVectorType vector_cast(const Vector<N, Real>& vec)
    {
        enum {SmallN = OtherVectorType::num_elements >= N ? N: OtherVectorType::num_elements};
        OtherVectorType other;
        for (int i = 0; i < SmallN; ++i)
            other[i] = static_cast<typename OtherVectorType::value_type>(vec[i]);
        for (int i = SmallN; i < OtherVectorType::num_elements; ++i)
            other[i] = static_cast<typename OtherVectorType::value_type>(0);
        return other;
    }

    //*****************************************************************************
    //--------------------------- comparison operators ----------------------------

    /// @return True if all components are equal.
    template <int N, class Real>
    bool operator==(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        for (int j=0; j < N; ++j)
        {
            if (lvec[j] != rvec[j])
                return false; // found a component that's not equal
        }
        return true; // all components equal
    }

    /// @return True if any component is not equal.
    template <int N, class Real>
    bool operator!=(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        return !(lvec == rvec);
    }

    /// we compare vectors like we compare strings
    template <int N, class Real>
    bool operator<(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        for (int j=0; j < N; ++j)
        {
            if (lvec[j] < rvec[j])
                return true;
            else if (lvec[j] > rvec[j])
                return false;
        }
        return false; // all components equal
    }

    template <int N, class Real>
    bool operator<=(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        return !(rvec > lvec);
    }

    template <int N, class Real>
    bool operator>(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        return rvec < lvec;
    }

    template <int N, class Real>
    bool operator>=(const Vector<N,Real>& lvec, const Vector<N,Real>& rvec)
    {
        return rvec <= lvec;
    }

    template <int N, class Real>
    Real Vector<N, Real>::maxComponent() const
    {
        Real r = m_tuple[0];
        for (int i = 1; i < N; ++i)
            r = std::max(r, m_tuple[i]);
        return r;
    }

    }

#endif