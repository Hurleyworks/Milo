
// modified from
// Geometric Tools, LLC
// Copyright (c) 1998-2010
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt

#pragma once

template <typename Real>
class Plane
{

 public:
    // The plane is represented as Dot(N,X) = c where N is a unit-length
    // normal vector, c is the plane constant, and X is any point on the
    // plane.  The user must ensure that the normal vector is unit length.

    // Construction and destruction.
    Plane ();  // uninitialized
    ~Plane ();

    // Specify N and c directly.
    Plane (const Eigen::Matrix<Real,3,1>& normal, Real constant);

    // N is specified, c = Dot(N,P) where P is a point on the plane.
    Plane (const Eigen::Matrix<Real,3,1>& normal, const Eigen::Matrix<Real,3,1>& p);

    // N = Cross(P1-P0,P2-P0)/Length(Cross(P1-P0,P2-P0)), c = Dot(N,P0) where
    // P0, P1, P2 are points on the plane.
    Plane (const Eigen::Matrix<Real,3,1>& p0, const Eigen::Matrix<Real,3,1>& p1,
        const Eigen::Matrix<Real,3,1>& p2);

    // Compute d = Dot(N,P)-c where N is the plane normal and c is the plane
    // constant.  This is a signed distance.  The sign of the return value is
    // positive if the point is on the positive side of the plane, negative if
    // the point is on the negative side, and zero if the point is on the
    // plane.
    Real distanceTo (const Eigen::Matrix<Real,3,1>& p) const;

    // The "positive side" of the plane is the half space to which the plane
    // normal points.  The "negative side" is the other half space.  The
    // function returns +1 when P is on the positive side, -1 when P is on the
    // the negative side, or 0 when P is on the plane.
    int whichSide (const Eigen::Matrix<Real,3,1>& p) const;

    Eigen::Matrix<Real,3,1> normal;
    Real constant;
};


template <typename Real>
Plane<Real>::Plane ()
{
}

template <typename Real>
Plane<Real>::~Plane ()
{
}

template <typename Real>
Plane<Real>::Plane (const Eigen::Matrix<Real,3,1>& normal, Real constant)
    :
    normal(normal),
    constant(constant)
{
}

template <typename Real>
Plane<Real>::Plane (const Eigen::Matrix<Real,3,1>& normal, const Eigen::Matrix<Real,3,1>& p)
    :
    normal(normal)
{
    constant = normal.dot(p);
}

template <typename Real>
Plane<Real>::Plane (const Eigen::Matrix<Real,3,1>& p0, const Eigen::Matrix<Real,3,1>& p1,
    const Eigen::Matrix<Real,3,1>& p2)
{
    Eigen::Matrix<Real,3,1> edge1 = p1 - p0;
    Eigen::Matrix<Real,3,1> edge2 = p2 - p0;
    normal = edge1.cross(edge2);
	normal.normalize();
    constant = normal.dot(p0);
}

template <typename Real>
Real Plane<Real>::distanceTo (const Eigen::Matrix<Real,3,1>& p) const
{
    return normal.dot(p) - constant;
}

template <typename Real>
int Plane<Real>::whichSide (const Eigen::Matrix<Real,3,1>& p) const
{
    Real distance = distanceTo(p);

    if (distance < (Real)0)
    {
        return -1;
    }
    else if (distance > (Real)0)
    {
        return +1;
    }
    else
    {
        return 0;
    }
}

typedef Plane<float> Planef;
typedef Plane<double> Planed;