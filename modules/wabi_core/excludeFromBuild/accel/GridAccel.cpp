
// ctor
template <typename T>
GridAccel<T>::GridAccel ()
	: isBuilt_(false),
	  voxels_(0),
	  voxelCount_(0)
{	
	LOG(TESTING) << "Grid created";
}

// dtor
template <typename T>
GridAccel<T>::~GridAccel ()
{	
	cleanUp();
	LOG(TESTING) << "Grid destroyed";
}

// cleanUp
template <typename T>
void GridAccel<T>::cleanUp()
{
	if (!voxelCount_) return;

	// cleanup voxels
	for( int i = 0; i < voxelCount_; i++ )
	{
		Voxel<T> *voxel = voxels_[i];
		if( voxel ) 
		{
			delete voxel;
			voxel = 0;
		}
	}
	FreeAligned(voxels_);
}

// construct
template <typename T>
void GridAccel<T>::construct (const BoundingBox3<T> meshBBox)
{	
	ScopedStopWatch sw(__FUNCTION_NAME__);

	isBuilt_ = false;

	int triCount = (int)triangles.size();

	// can't populate an acceleration structure without triangles!
	if( !triCount )
		return;

	// bounding box is in world space
	bbox_ = meshBBox;
	bbox_.enlargeByEpsilon();

	// compute a magic number for determining the voxel granularity
	// it's the cube root of the average volume of a single polygon
	T volume = bbox_.getVolume();
	T magic = Math<T>::Pow( volume/triCount, (T)1/(T)3.0 );
	Matrix<T,3,1> extents = bbox_.getExtents();

	// some initializations
	for( int axis = 0; axis < 3; axis++ )
	{
		// compute number of voxels per axis, clamping 
		// the minimum to 1 and maximum to VOXEL_MAX
		voxPerAxis_[axis] = clamp<int>((int)((extents[axis]/magic) + 0.5),1,VOXEL_MAX);

		// no sense having more voxels on an axis than polys... or is there?
		// this solves the problem of having a gazillion voxels for just a
		// single flat triangle
		if( voxPerAxis_[axis] > triCount ) 
			voxPerAxis_[axis] = triCount;

		// compute the dimensions of a single voxel
		voxDim_[axis] = extents[axis]/voxPerAxis_[axis];

		// and store it's inverse for later use
		invVoxDim_[axis] = 1/voxDim_[axis];

		// voxel extent will be used to 
		// compute the voxel bounding boxes
		voxelExtent_[axis] = voxDim_[axis];
	}

	// allocate and initialze to 0 an array to hold voxel ptrs
	voxelCount_ = voxPerAxis_[0] * voxPerAxis_[1] * voxPerAxis_[2];
	voxels_ = (Voxel<T> **)AllocAligned(voxelCount_ * sizeof(Voxel<T> *));
	memset(voxels_, 0, voxelCount_ * sizeof(Voxel<T> *));
}

// populate
template <typename T>
void GridAccel<T>::populate ()
{	
	ScopedStopWatch sw(__FUNCTION_NAME__);

	// fill the voxels with triangle hits
	// and compute the bounding box of only 
	// the voxels that actually contain geometry
	int min[3];
	int max[3];
	int triIndex = 0;

	for ( auto & tri : triangles)
	{
		getVoxelIndex( tri.getBound(0),
			           tri.getBound(2),
					   tri.getBound(4),
					   min );

		getVoxelIndex( tri.getBound(1),
			           tri.getBound(3),
					   tri.getBound(5),
					   max );

		for( int x = min[0]; x <= max[0]; x++ )
		{
			for( int y = min[1]; y <= max[1]; y++)
			{
				for( int z = min[2]; z <= max[2]; z++ )
				{
					// compute the index of this voxel
					int voxelIndex = z * voxPerAxis_[0] * voxPerAxis_[1] + y * voxPerAxis_[0] + x;

					// if this voxel hasn't been hit yet then make a
					// new voxel and compute it's bounding box
					if ( voxels_[voxelIndex] == 0 )
					{
						voxels_[voxelIndex] = new Voxel<T>;
						voxels_[voxelIndex]->id = voxelIndex;
						
						// compute the min/max corner points
						// of this voxel's bounding box
						voxels_[voxelIndex]->bbox.min()[0] = bbox_.min()[0] + x * voxDim_[0];
						voxels_[voxelIndex]->bbox.min()[1] = bbox_.min()[1] + y * voxDim_[1];
						voxels_[voxelIndex]->bbox.min()[2] = bbox_.min()[2] + z * voxDim_[2];
						voxels_[voxelIndex]->bbox.max() = voxels_[voxelIndex]->bbox.min() + voxelExtent_;
					}

					// add this triangle index to the list of
					// primitives indices that fall within this voxel
					voxels_[voxelIndex]->hits.push_back(triIndex);
				}
			}
		}
		++triIndex;
	}

	isBuilt_ = true;
}

// getVoxelIndex
template<typename T>
void GridAccel<T>::getVoxelIndex (T x, T y, T z, int i[])
{	
	i[0] = clamp<int>( (int)((x - bbox_.min()[0]) * invVoxDim_[0]), 0, voxPerAxis_[0]-1 );
	i[1] = clamp<int>( (int)((y - bbox_.min()[1]) * invVoxDim_[1]), 0, voxPerAxis_[1]-1 );
	i[2] = clamp<int>( (int)((z - bbox_.min()[2]) * invVoxDim_[2]), 0, voxPerAxis_[2]-1 );
}

// offset
template<typename T>
int GridAccel<T>::offset (int x, int y, int z) const
{	
	return z * voxPerAxis_[0] * voxPerAxis_[1] + y * voxPerAxis_[0] + x;
}

// posToVoxel
template<typename T>
int GridAccel<T>::posToVoxel (const Matrix<T,3,1> & pos, int axis) const
{	
	int vp = (int)((pos[axis] - bbox_.min()[axis]) * invVoxDim_[axis]);
	return clamp<int>(vp, 0, voxPerAxis_[axis]-1);
}

// voxelToPos
template<typename T>
T GridAccel<T>::voxelToPos (int p, int axis) const
{	
	return bbox_.min()[axis] + p * voxDim_[axis];
}

// voxelToPos
template<typename T>
Matrix<T,3,1> GridAccel<T>::voxelToPos (int x, int y, int z) const
{	
	return bbox_.min() +  Matrix<T,3,1>(x * voxDim_[0], y * voxDim_[1], z * voxDim_[2]);
}

// getVoxelCount
template<typename T>
long GridAccel<T>::getVoxelCount () const
{	
	return voxelCount_;
}

// getVoxels
template<typename T>
Voxel<T>** GridAccel<T>::getVoxels () const
{	
	return voxels_;
}

// testForIntersection
template<typename T>
bool GridAccel<T>::testForIntersection (Ray3<T> & ray) 
{	
	if( !isBuilt_ ) 
		return false;

	T rayT;
	if (bbox_.contains(ray(ray.tMin)))
		rayT = ray.tMin;
	else if (!bbox_.intersect(ray, &rayT))
		return false;

	Matrix<T,3,1> entryPoint = ray(rayT);

	// compute the inverse ray direction so 
	// we can multiply instead of divide
	T invRayDir[3];
	for( int i = 0; i < 3; i++ )
		invRayDir[i] = 1/ray.dir[i];

	T nextT[3], deltaT[3];
	int step[3], stop[3], pos[3];

	// traversal initialization 
	for( int i = 0; i < 3; ++i)
	{
		pos[i] = posToVoxel( entryPoint, i );
		if (ray.dir[i] >= 0)
		{
			nextT[i] = rayT +  ( voxelToPos(pos[i]+1, i) - entryPoint[i] ) * invRayDir[i];
			deltaT[i] = voxDim_[i] * invRayDir[i]; 
			step[i] = 1;
			stop[i] = voxPerAxis_[i];
		}
		else 
		{
			nextT[i] = rayT + ( voxelToPos(pos[i], i) - entryPoint[i] ) * invRayDir[i]; 
			deltaT[i] = -voxDim_[i] * invRayDir[i]; 
			step[i] = -1;
			stop[i] = -1;
		}
	}

	// Walk the voxel grid walk
	T closestHit = Math<T>::MAX_REAL;
	bool polyWasHit = false;
	for (;;) 
	{
		Voxel<T> *voxel = voxels_[offset(pos[0],pos[1],pos[2])];
		if (voxel != 0)
		{
			// check if this ray intersects with any tris that
			// have been stored in this voxel
			for (auto & polyID : voxel->hits)
			{
				Triangle3<T> & tri = triangles[polyID];
				if( tri.findIntersect(ray) )
				{
					if( ray.distToHit < closestHit && polyID != ray.hitPolyID)
					{
						closestHit = ray.distToHit;
						ray.hitPolyID = polyID;
						ray.surfaceNormal = tri.getNormal();
					}
				}
			}
		}

		if( closestHit < Math<T>::MAX_REAL )  
		{
			ray.distToHit = closestHit;
			ray.hitPoint = ray(ray.distToHit);
			ray.surfaceNormal.normalize();
			ray.wasHit = true;
		}

		// advance to next voxel
		int bits = ((nextT[0] < nextT[1]) << 2) +
				   ((nextT[0] < nextT[2]) << 1) +
				   ((nextT[1] < nextT[2]));
		const int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
		int stepAxis = cmpToAxis[bits];
		if (ray.tMax < nextT[stepAxis])
			break;
		pos[stepAxis] += step[stepAxis];
		if (pos[stepAxis] == stop[stepAxis])
			break;
		nextT[stepAxis] += deltaT[stepAxis];
	}
	if( ray.wasHit )  
		return true;
	else
		return false;
}

// dumpVoxelInfo
template<typename T>
void GridAccel<T>::dumpVoxelInfo()
{	
	int usedVoxels = 0;
	int unusedVoxels = 0;
	int voxelHits = 0;
	int highCount = -1000000;
	int lowCount = 1000000;
	for( int i = 0; i < voxelCount_; i++ )
	{
		Voxel<T>* voxel = voxels_[i];
		if( voxel )
		{
			++usedVoxels;
			int hitCount = (int)voxel->hits.size();
			voxelHits += hitCount;
			if( hitCount > highCount )
				highCount = hitCount;
			if( hitCount < lowCount )
				lowCount = hitCount;
		}
		else
			++unusedVoxels;
	}

	std::ostringstream ostr;
	ostr << "\nVoxel count: " + ToString<long>(voxelCount_) << std::endl;
	ostr << "Used voxels: " + ToString<int>(usedVoxels) << std::endl;
	ostr << "Unused voxels: " + ToString<int>(unusedVoxels) << std::endl;
	ostr << "Average triCount in voxel: " + ToString<int>(voxelHits/usedVoxels) << std::endl;
	ostr << "Highest voxel triCount: " + ToString<int>(highCount) << std::endl;
	ostr << "Lowest voxel triCount: " + ToString<int>(lowCount) << std::endl;

	LOG(TESTING) << ostr.str();

	ostr.clear();

	for( int i = 0; i < voxelCount_; i++ )
	{
		Voxel<T>* voxel = voxels_[i];
		if( voxel )
		{
			ostr << "-------Voxel ID: " << ToString<size_t>(voxel->id) << std::endl;
			ostr << voxel->bbox.asString() << std::endl;

			typename Voxel<T>::VoxelHits::iterator it = voxel->hits.begin();
            typename Voxel<T>::VoxelHits::iterator end = voxel->hits.end();
			while( it != end )
			{
				ostr << ToString<size_t>(*it++) << std::endl;
			}
		}
	}
	LOG(TESTING) << ostr.str();
}

// dumpVoxelInfo
template<typename T>
void GridAccel<T>::dumpVoxelInfo(std::ostringstream& ostr)
{	
	int usedVoxels = 0;
	int unusedVoxels = 0;
	int voxelHits = 0;
	int highCount = -1000000;
	int lowCount = 1000000;
	for( int i = 0; i < voxelCount_; i++ )
	{
		Voxel<T>* voxel = voxels_[i];
		if( voxel )
		{
			++usedVoxels;
			int hitCount = (int)voxel->hits.size();
			voxelHits += hitCount;
			if( hitCount > highCount )
				highCount = hitCount;
			if( hitCount < lowCount )
				lowCount = hitCount;
		}
		else
			++unusedVoxels;
	}

	ostr << "\nVoxel count: " + ToString<long>(voxelCount_) << std::endl;
	ostr << "Used voxels: " + ToString<int>(usedVoxels) << std::endl;
	ostr << "Unused voxels: " + ToString<int>(unusedVoxels) << std::endl;
	ostr << "Average triCount in voxel: " + ToString<int>(voxelHits/usedVoxels) << std::endl;
	ostr << "Highest voxel triCount: " + ToString<int>(highCount) << std::endl;
	ostr << "Lowest voxel triCount: " + ToString<int>(lowCount) << std::endl;

	for( int i = 0; i < voxelCount_; i++ )
	{
		Voxel<T>* voxel = voxels_[i];
		if( voxel )
		{
			ostr << "-------Voxel ID: " << ToString<size_t>(voxel->id) << std::endl;
			ostr << voxel->bbox.asString() << std::endl;

			typename Voxel<T>::VoxelHits::iterator it = voxel->hits.begin();
            typename Voxel<T>::VoxelHits::iterator end = voxel->hits.end();
			while( it != end )
			{
				ostr << ToString<size_t>(*it++) << std::endl;
			}
		}
	}
}

template
class GridAccel<float>;

template
class GridAccel<double>;
