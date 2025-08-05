#pragma once

template <class T>
inline bool isOrthogonal (const Eigen::Matrix<T, 3, 3>& m)
{
    // to test whether a matrix is orthogonal, mutliply the matrix by
    // it's transform and compare to identity matrix
    return (m * m.transpose()).isIdentity();
}

inline bool reOrthogonalize (Eigen::Matrix3f& m)
{
    // http://stackoverflow.com/questions/23080791/eigen-re-orthogonalization-of-rotation-matrix

    Eigen::Matrix3f mo = m;

    Eigen::Vector3f x = mo.row (0);
    Eigen::Vector3f y = mo.row (1);
    Eigen::Vector3f z = mo.row (2);

    float error = x.dot (y);

    Eigen::Vector3f x_ort = x - (error / 2) * y;
    Eigen::Vector3f y_ort = y - (error / 2) * x;
    Eigen::Vector3f z_ort = x_ort.cross (y_ort);

    mo.row (0) = x_ort.normalized();
    mo.row (1) = y_ort.normalized();
    mo.row (2) = z_ort.normalized();

    if (isOrthogonal (mo))
    {
        m = mo;
        return true;
    }
    else
    {
        return false;
    }
}

// from Instant Meshes
inline float fast_acos (float x)
{
    float negate = float (x < 0.0f);
    x = std::abs (x);
    float ret = -0.0187293f;
    ret *= x;
    ret = ret + 0.0742610f;
    ret *= x;
    ret = ret - 0.2121144f;
    ret *= x;
    ret = ret + 1.5707288f;
    ret = ret * std::sqrt (1.0f - x);
    ret = ret - 2.0f * negate * ret;
    return negate * (float)M_PI + ret;
}

// https://liuzhiguang.wordpress.com/2017/06/12/find-the-angle-between-two-vectors/
inline float angleBetweenVectors (Eigen::Vector3f a, Eigen::Vector3f b)
{
    float angle = 0.0f;

    angle = std::atan2 (a.cross (b).norm(), a.dot (b));

    return angle;
}

// rad_to_deg
template <typename T>
inline T rad_to_deg (const T rad)
{
    return rad * 180 / Math<T>::PI;
}

// deg_to_rad
template <typename T>
inline T deg_to_rad (const T deg)
{
    return deg * Math<T>::PI / 180;
}

template <typename T>
inline T dotPerp (const Eigen::Matrix<T, 2, 1>& vec1, const Eigen::Matrix<T, 2, 1>& vec2)
{
    return vec1[0] * vec2[1] - vec1[1] * vec2[0];
}

template <class T>
inline T clamp (T a, T b, T c)
{
    return (a < b) ? b : ((a > c) ? c : a);
}

// Input W must be a unit-length vector.  The output vectors {U,V} are
// unit length and mutually perpendicular, and {U,V,W} is an orthonormal
// basis.
template <typename T>
inline void generateComplementBasis (Eigen::Matrix<T, 3, 1>& u,
                                     Eigen::Matrix<T, 3, 1>& v,
                                     const Eigen::Matrix<T, 3, 1>& w)
{
    T invLength;

    if (Math<T>::FAbs (w[0]) >= Math<T>::FAbs (w[1]))
    {
        // W.x or W.z is the largest magnitude component, swap them
        invLength = Math<T>::InvSqrt (w[0] * w[0] + w[2] * w[2]);
        u[0] = -w[2] * invLength;
        u[1] = (T)0;
        u[2] = +w[0] * invLength;
        v[0] = w[1] * u[2];
        v[1] = w[2] * u[0] - w[0] * u[2];
        v[2] = -w[1] * u[0];
    }
    else
    {
        // W.y or W.z is the largest magnitude component, swap them
        invLength = Math<T>::InvSqrt (w[1] * w[1] + w[2] * w[2]);
        u[0] = (T)0;
        u[1] = +w[2] * invLength;
        u[2] = -w[1] * invLength;
        v[0] = w[1] * u[2] - w[2] * u[1];
        v[1] = -w[0] * u[2];
        v[2] = w[0] * u[1];
    }
}

template <typename T>
inline T normalizeVec2 (Eigen::Matrix<T, 2, 1>& vec, const T epsilon = Math<T>::ZERO_TOLERANCE)
{
    T length = vec.norm();

    if (length > epsilon)
    {
        T invLength = ((T)1) / length;
        vec[0] *= invLength;
        vec[1] *= invLength;
    }
    else
    {
        length = (T)0;
        vec[0] = (T)0;
        vec[1] = (T)0;
    }

    return length;
}
