#pragma once

#include "../mace_core/mace_core.h"

using mace::ToString;

using Eigen::Matrix;
using Pose = Eigen::Affine3f;
using Scale = Eigen::Vector3f;

using Float = float;
using MatrixXc = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXf = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXu = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>;
using Vector3u = Eigen::Matrix<uint32_t, 3, 1>;
using Vector4u = Eigen::Matrix<uint32_t, 4, 1>;
using Vector2u = Eigen::Matrix<uint32_t, 2, 1>;
// using Matrix3f = Eigen::Matrix<Float, 3, 3>;
using Matrix43f = Eigen::Matrix<Float, 4, 3>;
using MatrixRowMajor34f = Eigen::Matrix<Float, 3, 4, Eigen::RowMajor>;
using Matrix4f = Eigen::Matrix<Float, 4, 4>;
using MatrixXu16 = Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXu = Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>;
using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
using MatrixXi = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>;


//using Eigen::Vector3f;
using Eigen::Vector2f;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::Matrix;

using Vertices3d = Eigen::MatrixXd;
using Normals3d = Eigen::MatrixXd;
using Vertices3f = Eigen::MatrixXf;
using UV2f = Eigen::MatrixXf;
using Normals3f = Eigen::MatrixXf;
using Indices3i = Eigen::MatrixXi;


const Eigen::Vector3f MAX_DIST_POINT_F = Eigen::Vector3f::Constant (std::numeric_limits<float>::max());
const Eigen::Vector3d MAX_DIST_POINT_D = Eigen::Vector3d::Constant (std::numeric_limits<double>::max());
const Eigen::Vector3f INVALID_NORMAL_F = Eigen::Vector3f::Constant (0.0);
const Eigen::Vector3d INVALID_NORMAL_D = Eigen::Vector3d::Constant (0.0);
const Eigen::Vector3i INVALID_TRI = Eigen::Vector3i::Constant (INVALID_INDEX);



// Memory Allocation Functions
#if defined(_WIN32)
#include <malloc.h>
#define memalign(a, b) _aligned_malloc (b, a)
#define freeAligned(ptr) _aligned_free (ptr)
#else
#include <stdlib.h>
static void* memalign (size_t alignment, size_t size)
{
    void* ptr = nullptr;
    if (posix_memalign (&ptr, alignment, size) != 0)
        ptr = nullptr;
    return ptr;
}
#define freeAligned free
#endif

#ifndef L1_CACHE_LINE_SIZE
#define L1_CACHE_LINE_SIZE 64
#endif

inline void* AllocAligned (size_t size)
{
    return memalign (L1_CACHE_LINE_SIZE, size);
}

inline void FreeAligned (void* ptr)
{
    freeAligned (ptr);
}

inline void fillWithMatrixWithRandomVectors (MatrixXf& matrix, int rows, int cols)
{
    // Resize the matrix
    matrix.resize (rows, cols);

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen (rd());
    std::uniform_real_distribution<> dis (-1.0, 1.0); // Uniform distribution between -1.0 and 1.0

    // Fill the matrix
    for (int col = 0; col < matrix.cols(); ++col)
    {
        // Create a random Vector3f
        Eigen::Vector3f randomVector (dis (gen), dis (gen), dis (gen));

        // Assign it to the current column
        matrix.col (col) = randomVector;
    }
}

inline Eigen::Vector3f extractScaleFromAffine3f (const Eigen::Affine3f& t)
{
    // Extract the linear part of the transformation
    const Eigen::Matrix3f& linearPart = t.linear();

    // Compute the singular value decomposition
    Eigen::JacobiSVD<Eigen::Matrix3f> svd (linearPart, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // The singular values correspond to the scaling factors
    return svd.singularValues();
}

namespace wabi
{
    // math
	#include "excludeFromBuild/math/TypeLimits.h"
	#include "excludeFromBuild/math/Maths.h"
	#include "excludeFromBuild/math/EigenExtras.h"
	#include "excludeFromBuild/math/Intersector.h"
	#include "excludeFromBuild/math/Intersector1.h"
	#include "excludeFromBuild/math/Ray3.h"
	#include "excludeFromBuild/math/BoundingBox3.h"
	#include "excludeFromBuild/math/Box3.h"
	#include "excludeFromBuild/math/Triangle3.h"
	#include "excludeFromBuild/math/Triangle2.h"
	#include "excludeFromBuild/math/Box3Box3Intersect.h"
	#include "excludeFromBuild/math/Line2.h"
	#include "excludeFromBuild/math/Line2Tri2Intersect.h"
	#include "excludeFromBuild/math/Line3.h"
	#include "excludeFromBuild/math/Plane.h"
	#include "excludeFromBuild/math/Plane3.h"
	#include "excludeFromBuild/math/MathUtil.h"
	#include "excludeFromBuild/math/Quad3.h"
	#include "excludeFromBuild/math/Query.h"
	#include "excludeFromBuild/math/Query2.h"
	#include "excludeFromBuild/math/Segment2.h"
	#include "excludeFromBuild/math/Segment3.h"
	#include "excludeFromBuild/math/Tetrahedron3.h"
	#include "excludeFromBuild/math/Seg2Seg2Intersect.h"
	#include "excludeFromBuild/math/Seg2Tri2Intersect.h"
	#include "excludeFromBuild/math/Seg3Tri3Intersect.h"
	#include "excludeFromBuild/math/Tri2Tri2Intersect.h"
	#include "excludeFromBuild/math/Tri3Tri3Intersect.h"
	#include "excludeFromBuild/math/Distance.h"
	#include "excludeFromBuild/math/Point3Tri3Dist.h"
	#include "excludeFromBuild/math/Line3Seg3Dist.h"
	#include "excludeFromBuild/math/Line3Tri3Dist.h"
	#include "excludeFromBuild/math/Seg3Tri3Dist.h"
	#include "excludeFromBuild/math/Tri3Tri3Dist.h"

	// mesh
	#include "excludeFromBuild/mesh/Tri.h"
	#include "excludeFromBuild/mesh/Triangulator.h"
	#include "excludeFromBuild/mesh/ZenMesh.h"
	#include "excludeFromBuild/mesh/FeatureKey.h"
	#include "excludeFromBuild/mesh/EdgeKey.h"
	#include "excludeFromBuild/mesh/TriangleKey.h"
	#include "excludeFromBuild/mesh/TetrahedronKey.h"
	#include "excludeFromBuild/mesh/ManifoldMesh.h"

	// acceleration structures
	#include "excludeFromBuild/accel/SimonGrid.h"
	#include "excludeFromBuild/accel/Voxel.h"
	#include "excludeFromBuild/accel/GridAccel.h"
} // namespace wabi