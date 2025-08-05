#include "berserkpch.h"
#include "wabi_core.h"

namespace wabi
{
	// math
	#include "excludeFromBuild/math/Maths.cpp"
	#include "excludeFromBuild/math/MathUtil.cpp"
	#include "excludeFromBuild/math/Triangle3.cpp"
	#include "excludeFromBuild/math/Intersector.cpp"
	#include "excludeFromBuild/math/Intersector1.cpp"
	#include "excludeFromBuild/math/Box3Box3Intersect.cpp"
	#include "excludeFromBuild/math/Line2Tri2Intersect.cpp"
	#include "excludeFromBuild/math/Line3.cpp"
	#include "excludeFromBuild/math/Plane3.cpp"
	#include "excludeFromBuild/math/Query.cpp"
	#include "excludeFromBuild/math/Seg2Seg2Intersect.cpp"
	#include "excludeFromBuild/math/Seg2Tri2Intersect.cpp"
	#include "excludeFromBuild/math/Seg3Tri3Intersect.cpp"
	#include "excludeFromBuild/math/Tetrahedron3.cpp"
	#include "excludeFromBuild/math/Tri2Tri2Intersect.cpp"
	#include "excludeFromBuild/math/Tri3Tri3Intersect.cpp"
	#include "excludeFromBuild/math/Distance.cpp"
	#include "excludeFromBuild/math/Point3Tri3Dist.cpp"
	#include "excludeFromBuild/math/Line3Seg3Dist.cpp"
	#include "excludeFromBuild/math/Line3Tri3Dist.cpp"
	#include "excludeFromBuild/math/Seg3Tri3Dist.cpp"
	#include "excludeFromBuild/math/Tri3Tri3Dist.cpp"

	// mesh
	#include "excludeFromBuild/mesh/Triangulator.cpp"
	#include "excludeFromBuild/mesh/EdgeKey.cpp"
	#include "excludeFromBuild/mesh/TriangleKey.cpp"
	#include "excludeFromBuild/mesh/TetrahedronKey.cpp"
	#include "excludeFromBuild/mesh/ManifoldMesh.cpp"

	// acceleration structures
	#include "excludeFromBuild/accel/GridAccel.cpp"

} // namespace wabi
