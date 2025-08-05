#pragma once

// Modified version of this

/*
Grid - Space Partitioning algorithms for Cinder

Copyright (c) 2016, Simon Geilfus, All rights reserved.
This code is intended for use with the Cinder C++ library: http://libcinder.org

Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

template<uint8_t DIM, class T, class DataT> struct GridTraits {};

//! Represents a SimonGrid / Bin-lattice space partitioning structure
template<uint8_t DIM, class T, class DataT>
class SimonGrid
{

 public:
	class Node;
	using vec_t = Eigen::Matrix<T, DIM, 1>;
	using ivec_t = Eigen::Matrix<int, DIM, 1>; 

	//! Constructs an unbounded SimonGrid, insertion will be much slower than the bounded version
	SimonGrid(uint32_t k = 3);
	SimonGrid(const vec_t &min, const vec_t &max, uint32_t k = 3);

	//! Inserts a new point in the SimonGrid with optional user data
	void insert(const vec_t & position, DataT data = DataT());

	//! Represents a single element of the SimonGrid
	class Node
	{
		public:
			//! Returns the position of the node
			vec_t getPosition() const { return mPosition; }

			//! Returns the user data
			DataT getData() const { return mData; }

			Node(const vec_t &position, DataT data, std::vector<Node*> *bin);
		protected:
			vec_t mPosition;
			DataT mData;
			std::vector<Node*>* mBinPtr;
			friend class SimonGrid;
	};

	using NodePair = std::pair<Node*, T>;
	using Vector = std::vector<std::vector<Node*>>;
	using bounds_t = typename GridTraits<DIM, T, DataT>::Bounds;

	//! Returns a pointer to the nearest Node with its square distance to the position
	Node* nearestNeighborSearch(const vec_t &position, T *distanceSq = nullptr) const;

	//! Returns a vector of Nodes within a radius along with their square distances to the position
	std::vector<NodePair> rangeSearch(const vec_t &position, T radius) const;

	//! Returns a vector of Nodes within a radius along with their square distances to the position
	void rangeSearch(const vec_t &position, T radius, const std::function<void(Node*, T)> &visitor) const;


	//! Returns the number of bins of the grid
	size_t getNumBins() const { return mBins.size(); }

	//! Returns the number of bins of the grid in each dimension
	vec_t getNumCells() const { return mNumCells.template cast<float>(); }

	//! Returns the ith bin as a std::vector of Node*
	const std::vector<Node*> getBin(size_t i) const { return mBins[i]; }

	//! Returns the bin at a position as a std::vector of Node*
	const std::vector<Node*> getBinAt(const vec_t &position) const;

	//! Returns the bin index at a position
	size_t getBinIndexAt(const vec_t &position) const;

	//! Returns the center of the ith bin
	vec_t getBinCenter(size_t i) const;

	//! Returns the bounds of the ith bin
	bounds_t getBinBounds(size_t i) const;

	//! Returns the size of a bin
	vec_t getBinsSize() const { return vec_t(mCellSize, mCellSize, mCellSize); }

	//! Returns the minimum of the SimonGrid
	vec_t getMin() const { return mMin; }

	//! Returns the maximum of the SimonGrid
	vec_t getMax() const { return mMax; }

	//! Returns the bounds of the SimonGrid
	bounds_t getBounds() const;

 protected:
	 void resize(uint32_t k);
	 void resize(const vec_t &min, const vec_t &max);
	 void resize(const vec_t &min, const vec_t &max, uint32_t k);
	 void insert(Node *node);

	Vector		mBins;
	ivec_t		mNumCells, mGridMin, mGridMax;
	vec_t		mMin, mMax, mOffset;
	uint32_t	mK, mCellSize;
};



template<class T, class DataT>
struct GridTraits<3, T, DataT>
{
	// toGridPosition
	static typename SimonGrid<3, T, DataT>::ivec_t toGridPosition(const typename SimonGrid<3, T, DataT>::vec_t &position, const typename SimonGrid<3, T, DataT>::vec_t &offset, uint32_t k)
	{
		return typename SimonGrid<3, T, DataT>::ivec_t( static_cast<uint32_t>(position.x() + offset.x()) >> k,
														static_cast<uint32_t>(position.y() + offset.y()) >> k,
														static_cast<uint32_t>(position.z() + offset.z()) >> k);
	}

	// toGridPosition
	static typename SimonGrid<3, T, DataT>::ivec_t toGridPosition(uint32_t index, const typename SimonGrid<3, T, DataT>::ivec_t &numCells)
	{
		return typename SimonGrid<3, T, DataT>::ivec_t(index % numCells.x(), (index / numCells.x()) % numCells.y(), (index / numCells.x()) / numCells.y());
	}

	// toPosition
	static typename SimonGrid<3, T, DataT>::vec_t toPosition(const typename SimonGrid<3, T, DataT>::ivec_t &gridPosition, const typename SimonGrid<3, T, DataT>::vec_t &offset, uint32_t k)
	{
		return typename SimonGrid<3, T, DataT>::vec_t(static_cast<T>(gridPosition.x() << k) - offset.x(),
													  static_cast<T>(gridPosition.y() << k) - offset.y(),
													  static_cast<T>(gridPosition.z() << k) - offset.z());
	}

	// toIndex
	static uint32_t toIndex(const typename SimonGrid<3, T, DataT>::ivec_t &gridPos, const typename SimonGrid<3, T, DataT>::ivec_t &numCells)
	{
		return gridPos.x() + numCells.x() * (gridPos.y() + numCells.y() * gridPos.z());
	}

	// toIndex
	static uint32_t toIndex(const typename SimonGrid<3, T, DataT>::vec_t &position, const typename SimonGrid<3, T, DataT>::vec_t &offset, const typename SimonGrid<3, T, DataT>::ivec_t &numCells, uint32_t k)
	{
		typename SimonGrid<3, T, DataT>::ivec_t gridPos = toGridPosition(position, offset, k);
		return gridPos.x() + numCells.x() * (gridPos.y() + numCells.y() * gridPos.z());
	}

	// gridSize
	static uint32_t gridSize(const typename SimonGrid<3, T, DataT>::vec_t &numCells)
	{
		return numCells.x() * numCells.y() * numCells.z();
	}

	// rangeSearch
	static void rangeSearch(std::vector<typename SimonGrid<3, T, DataT>::NodePair> *results, const typename SimonGrid<3, T, DataT>::Vector &bins, const typename SimonGrid<3, T, DataT>::vec_t &position, T radius, const typename SimonGrid<3, T, DataT>::ivec_t &minCell, const typename SimonGrid<3, T, DataT>::ivec_t &maxCell, const typename SimonGrid<3, T, DataT>::ivec_t &numCells)
	{
		T distSq;
		T radiusSq = radius * radius;
		typename SimonGrid<3, T, DataT>::ivec_t pos;
		for (pos.z() = minCell.z(); pos.z() < maxCell.z(); pos.z()++)
		{
			for (pos.y() = minCell.y(); pos.y() < maxCell.y(); pos.y()++)
			{
				for (pos.x() = minCell.x(); pos.x() < maxCell.x(); pos.x()++) 
				{
					uint32_t index = GridTraits<3, T, DataT>::toIndex(pos, numCells);
					const std::vector<typename SimonGrid<3, T, DataT>::Node*>& cell = bins[index];
					for (const auto& node : cell)
					{
						distSq = (position - node->getPosition()).squaredNorm();
						if (distSq < radiusSq) 
						{
							results->emplace_back(std::make_pair(node, distSq));
						}
					}
				}
			}
		}
	}

	// rangeSearch
	static void rangeSearch(const std::function<void(typename SimonGrid<3, T, DataT>::Node*, T)> &visitor, const typename SimonGrid<3, T, DataT>::Vector &bins, const typename SimonGrid<3, T, DataT>::vec_t &position, T radius, const typename SimonGrid<3, T, DataT>::ivec_t &minCell, const typename SimonGrid<3, T, DataT>::ivec_t &maxCell, const typename SimonGrid<3, T, DataT>::ivec_t &numCells)
	{
		T distSq;
		T radiusSq = radius * radius;
		typename SimonGrid<3, T, DataT>::ivec_t pos;
		for (pos.z() = minCell.z(); pos.z() < maxCell.z(); pos.z()++) 
		{
			for (pos.y() = minCell.y(); pos.y() < maxCell.y(); pos.y()++) 
			{
				for (pos.x() = minCell.x(); pos.x() < maxCell.x(); pos.x()++) 
				{
					const std::vector<typename SimonGrid<3, T, DataT>::Node*>& cell = bins[GridTraits<3, T, DataT>::toIndex(pos, numCells)];
					for (const auto& node : cell) 
					{
						distSq = (position - node->getPosition()).squaredNorm();
						if (distSq < radiusSq) 
						{
							visitor(node, distSq);
						}
					}
				}
			}
		}
	}

	typedef BoundingBox3<T> Bounds;
};

// ctor
template<uint8_t DIM, class T, class DataT>
SimonGrid<DIM, T, DataT>::Node::Node(const vec_t &position, DataT data, std::vector<Node*> *bin)
	: mPosition(position), mData(data), mBinPtr(bin)
{
}

// ctor
template<uint8_t DIM, class T, class DataT>
SimonGrid<DIM, T, DataT>::SimonGrid(uint32_t k)
{
	resize(vec_t(std::numeric_limits<T>::max()), vec_t(std::numeric_limits<T>::min()), k);
}

// ctor
template<uint8_t DIM, class T, class DataT>
SimonGrid<DIM, T, DataT>::SimonGrid(const vec_t &min, const vec_t &max, uint32_t k)
{
	resize(min, max, k);
}

// resize
template<uint8_t DIM, class T, class DataT>
void SimonGrid<DIM, T, DataT>::resize(const vec_t &min, const vec_t &max)
{
	mMin = min;
	mMax = max;
	resize(mK);
}

// resize
template<uint8_t DIM, class T, class DataT>
void SimonGrid<DIM, T, DataT>::resize(const vec_t &min, const vec_t &max, uint32_t k)
{
	mMin = min;
	mMax = max;
	resize(k);
}

// resize
template<uint8_t DIM, class T, class DataT>
void SimonGrid<DIM, T, DataT>::resize(uint32_t k)
{
	// If we have existing nodes we need to save them
	std::vector<Node*> nodes;
	for (auto& cell : mBins)
	{
		nodes.insert(nodes.end(), cell.begin(), cell.end());
		cell.clear();
	}

	// Update grid settings
	mK = k;
	mCellSize = 1 << k;
	mOffset = eigenEx::ceil<T>(-mMin);
	mGridMin = GridTraits<DIM, T, DataT>::toGridPosition(mMin, mOffset, mK);
	mGridMax = GridTraits<DIM, T, DataT>::toGridPosition(mMax, mOffset, mK);
	

	vec_t v = ((mMax - mMin) + vec_t(1,1,1)) / static_cast<T>(mCellSize);
	v = eigenEx::ceil<T>(v);
	mNumCells = v.template cast<int>();

	mBins.resize(GridTraits<DIM, T, DataT>::gridSize(mNumCells.template cast<float>()));

	// Re-insert old nodes
	for (const auto& node : nodes)
	{
		insert(node);
	}
}

// insert
template<uint8_t DIM, class T, class DataT>
void SimonGrid<DIM, T, DataT>::insert(const vec_t &position, DataT data)
{
	// Check if it fits the size of the grid's container
	if (eigenEx::any(eigenEx::greaterThan(position, mMax)) || eigenEx::any(eigenEx::lessThan(position, mMin)))
	{
		resize(eigenEx::min(position, mMin), eigenEx::max(position, mMax));
	}
		
	// Convert the position to 1D index
	uint32_t j = GridTraits<DIM, T, DataT>::toIndex(position, mOffset, mNumCells, mK);
	
	// And try to insert it in the grid
	if (j >= 0 && j < mBins.size()) {
		mBins[j].push_back(new Node(position, data, &mBins[j]));
	}
	else
		throw std::runtime_error("Position out of range!");
}

// insert
template<uint8_t DIM, class T, class DataT>
void SimonGrid<DIM, T, DataT>::insert(Node *node)
{
	// Check if it fits the size of the grid's container
	if (eigenEx::any(eigenEx::greaterThan(node->getPosition(), mMax)) || eigenEx::any(eigenEx::lessThan(node->getPosition(), mMin)))
	{
		resize(eigenEx::min(node->getPosition(), mMin), eigenEx::max(node->getPosition(), mMax));
	}

	// Convert the position to 1D index
	uint32_t j = GridTraits<DIM, T, DataT>::toIndex(node->getPosition(), mOffset, mNumCells, mK);

	// And try to insert it in the grid
	if (j >= 0 && j < mBins.size())
		mBins[j].push_back(node);
	else
		throw std::runtime_error("Position out of range!");
}

// TODO: Must be a better way to do this
// nearestNeighborSearch
template<uint8_t DIM, class T, class DataT>
typename SimonGrid<DIM, T, DataT>::Node* SimonGrid<DIM, T, DataT>::nearestNeighborSearch(const vec_t &position, T *distanceSq) const
{
	// Grow search radius until found something
	// TODO: !!! Might grow forever !!!
	std::vector<typename SimonGrid<DIM, T, DataT>::NodePair> results;
	T cellSize = static_cast<T>(mCellSize);
	while (!results.size()) 
	{
		results = rangeSearch(position, cellSize);
		cellSize *= 2;
	}

	// Once we have nodes to look at, iterate and find the closest one
	Node* nearestNode = nullptr;
	T minDist = std::numeric_limits<T>::max();
	for (const auto& node : results) 
	{
		if (node.second < minDist)
		{
			nearestNode = node.first;
			minDist = node.second;
		}
	}
	if (distanceSq != nullptr)
		*distanceSq = minDist;

	return nearestNode;
}

// rangeSearch
template<uint8_t DIM, class T, class DataT>
std::vector<typename SimonGrid<DIM, T, DataT>::NodePair> SimonGrid<DIM, T, DataT>::rangeSearch(const vec_t &position, T radius) const
{
	vec_t radiusVec = vec_t(radius, radius, radius);
	vec_t min = eigenEx::clamp<T>(position - radiusVec, mMin, mMax + vec_t(1,1,1));
	vec_t max = eigenEx::clamp<T>(position + radiusVec, mMin, mMax + vec_t(1,1,1));
	
	ivec_t minCell = eigenEx::max(GridTraits<DIM, T, DataT>::toGridPosition(min, mOffset, mK), ivec_t(0,0,0));
	ivec_t maxCell = eigenEx::min<int>(ivec_t(1,1,1) + GridTraits<DIM, T, DataT>::toGridPosition(max, mOffset, mK), mNumCells);

	std::vector<typename SimonGrid<DIM, T, DataT>::NodePair> results;
	GridTraits<DIM, T, DataT>::rangeSearch(&results, mBins, position, radius, minCell, maxCell, mNumCells);

	return results;
}

// rangeSearch
template<uint8_t DIM, class T, class DataT>
void SimonGrid<DIM, T, DataT>::rangeSearch(const vec_t &position, T radius, const std::function<void(Node*, T)> &visitor) const
{
	vec_t radiusVec = vec_t(radius, radius, radius);
	vec_t min = eigenEx::clamp<T>(position - radiusVec, mMin, mMax + vec_t(1,1,1));
	vec_t max = eigenEx::clamp<T>(position + radiusVec, mMin, mMax + vec_t(1,1,1));

	ivec_t minCell = eigenEx::max(GridTraits<DIM, T, DataT>::toGridPosition(min, mOffset, mK), ivec_t(0, 0, 0));
	ivec_t maxCell = eigenEx::min<int>(ivec_t(1,1,1) + GridTraits<DIM, T, DataT>::toGridPosition(max, mOffset, mK), mNumCells);

	GridTraits<DIM, T, DataT>::rangeSearch(visitor, mBins, position, radius, minCell, maxCell, mNumCells);
}

// getBinAt
template<uint8_t DIM, class T, class DataT>
const std::vector<typename SimonGrid<DIM, T, DataT>::Node*> SimonGrid<DIM, T, DataT>::getBinAt(const vec_t &position) const
{
	// throw an exception if we're not in the grid bounds
	auto gridPos = GridTraits<DIM, T, DataT>::toGridPosition(position, mOffset, mK);
	if (eigenEx::any(eigenEx::greaterThan(gridPos, mGridMax)) || eigenEx::any(eigenEx::lessThan(gridPos, mGridMin)))
		throw std::runtime_error("Position out of range!");

	// get the converted position as a 1D index and return the corresponding bin
	auto i = GridTraits<DIM, T, DataT>::toIndex(gridPos, mNumCells);
	return mBins[i];
}

// getBinIndexAt
template<uint8_t DIM, class T, class DataT>
size_t SimonGrid<DIM, T, DataT>::getBinIndexAt(const vec_t &position) const
{
	// throw an exception if we're not in the grid bounds
	auto gridPos = GridTraits<DIM, T, DataT>::toGridPosition(position, mOffset, mK);
	if (eigenEx::any(eigenEx::greaterThan(gridPos, mGridMax)) || eigenEx::any(eigenEx::lessThan(gridPos, mGridMin)))
		throw std::runtime_error("Position out of range!");

	return GridTraits<DIM, T, DataT>::toIndex(gridPos, mNumCells);
}

// getBinCenter
template<uint8_t DIM, class T, class DataT>
typename SimonGrid<DIM, T, DataT>::vec_t SimonGrid<DIM, T, DataT>::getBinCenter(size_t i) const
{
	ivec_t gridPosition = GridTraits<DIM, T, DataT>::toGridPosition(i, mNumCells);
	vec_t position = GridTraits<DIM, T, DataT>::toPosition(gridPosition, mOffset, mK);

	return position + vec_t(static_cast<T>(mCellSize), static_cast<T>(mCellSize), static_cast<T>(mCellSize)) / static_cast<T>(2);
}

// getBinBounds
template<uint8_t DIM, class T, class DataT>
typename SimonGrid<DIM, T, DataT>::bounds_t SimonGrid<DIM, T, DataT>::getBinBounds(size_t i) const
{
	ivec_t gridPosition = GridTraits<DIM, T, DataT>::toGridPosition(i, mNumCells);
	vec_t position = GridTraits<DIM, T, DataT>::toPosition(gridPosition, mOffset, mK);

	return bounds_t(position, position + vec_t(static_cast<T>(mCellSize)));
}

// getBounds
template<uint8_t DIM, class T, class DataT>
typename SimonGrid<DIM, T, DataT>::bounds_t SimonGrid<DIM, T, DataT>::getBounds() const
{
	vec_t min = GridTraits<DIM, T, DataT>::toPosition(mGridMin, mOffset, mK);
	vec_t max = GridTraits<DIM, T, DataT>::toPosition(mGridMax + ivec_t(1), mOffset, mK);

	return bounds_t(min, max);
}

template<class DataT = uint32_t> using SimonGrid3f = SimonGrid<3, float, DataT>;
template<class DataT = uint32_t> using SimonGrid3d = SimonGrid<3, double, DataT>;