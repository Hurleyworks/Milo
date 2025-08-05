#pragma once

namespace eigenEx
{

	template<typename T>
	inline Eigen::Matrix<T, 3, 1> ceil(const Eigen::Matrix<T, 3, 1> & v1)
	{
		Eigen::Matrix<T, 3, 1> result;
		result.x() = std::ceil(v1.x());
		result.y() = std::ceil(v1.y());
		result.z() = std::ceil(v1.z());

		return result;
	}

	template<typename T>
	inline Eigen::Matrix<T, 3, 1> floor(const Eigen::Matrix<T, 3, 1> & v1)
	{
		Eigen::Matrix<T, 3, 1> result;
		result.x() = std::floor(v1.x());
		result.y() = std::floor(v1.y());
		result.z() = std::floor(v1.z());

		return result;
	}
	

	template<typename T>
	inline Eigen::Matrix<bool, 3, 1> greaterThan(const Eigen::Matrix<T, 3, 1> & v1, const Eigen::Matrix<T, 3, 1> & v2)
	{
		Eigen::Matrix<bool, 3, 1> result;

		result.x() = v1.x() > v2.x();
		result.y() = v1.y() > v2.y();
		result.z() = v1.z() > v2.z();

		return result;
	}

	template<typename T>
	inline Eigen::Matrix<bool, 3, 1> lessThan(const Eigen::Matrix<T, 3, 1> & v1, const Eigen::Matrix<T, 3, 1> & v2)
	{
		Eigen::Matrix<bool, 3, 1> result;

		result.x() = v1.x() < v2.x();
		result.y() = v1.y() < v2.y();
		result.z() = v1.z() < v2.z();

		return result;
	}

	inline bool any(const Eigen::Matrix<bool, 3, 1> & v)
	{
		bool result = false;
		for (int i = 0; i < 3; i++)
		{
			result = result || v[i];
		}

		return result;
	}

	inline bool all(const Eigen::Matrix<bool, 3, 1> & v)
	{
		bool result = true;
		for (int i = 0; i < 3; i++)
		{
			result = result && v[i];
		}

		return result;
	}

	template<typename T>
	inline Eigen::Matrix<T, 3, 1> min(const Eigen::Matrix<T, 3, 1> & v1, const Eigen::Matrix<T, 3, 1> & v2)
	{
		Eigen::Matrix<T, 3, 1> result;

		result.x() = std::min(v1.x() , v2.x());
		result.y() = std::min(v1.y(), v2.y());
		result.z() = std::min(v1.z(), v2.z());

		return result;
	}

	template<typename T>
	inline Eigen::Matrix<T, 3, 1> max(const Eigen::Matrix<T, 3, 1> & v1, const Eigen::Matrix<T, 3, 1> & v2)
	{
		Eigen::Matrix<T, 3, 1> result;

		result.x() = std::max(v1.x(), v2.x());
		result.y() = std::max(v1.y(), v2.y());
		result.z() = std::max(v1.z(), v2.z());

		return result;
	}

	template<typename T>
	inline Eigen::Matrix<T, 3, 1> clamp(const Eigen::Matrix<T, 3, 1> & v, const Eigen::Matrix<T, 3, 1> & minVal, const Eigen::Matrix<T, 3, 1> & maxVal)
	{
		return eigenEx::min(eigenEx::max(v, minVal), maxVal);
	}

	// from WildMagic
	template<typename T>
	inline void generateComplementBasis(Eigen::Matrix<T, 3, 1> & vec0, Eigen::Matrix<T, 3, 1> & vec1, const Eigen::Matrix<T, 3, 1> & vec2)
	{
		T invLength;

		if (Math<T>::FAbs(vec2[0]) >= Math<T>::FAbs(vec2[1]))
		{
			// vec2.x or vec2.z is the largest magnitude component, swap them
			invLength = (T)1.0 / Math<T>::Sqrt(vec2[0] * vec2[0] + vec2[2] * vec2[2]);
			vec0[0] = -vec2[2] * invLength;
			vec0[1] = 0.0f;
			vec0[2] = +vec2[0] * invLength;
			vec1[0] = vec2[1] * vec0[2];
			vec1[1] = vec2[2] * vec0[0] - vec2[0] * vec0[2];
			vec1[2] = -vec2[1] * vec0[0];
		}
		else
		{
			// vec2.y or vec2.z is the largest magnitude component, swap them
			invLength = (T)1.0 / Math<T>::Sqrt(vec2[1] * vec2[1] + vec2[2] * vec2[2]);
			vec0[0] = 0.0f;
			vec0[1] = +vec2[2] * invLength;
			vec0[2] = -vec2[1] * invLength;
			vec1[0] = vec2[1] * vec0[2] - vec2[2] * vec0[1];
			vec1[1] = -vec2[0] * vec0[2];
			vec1[2] = vec2[0] * vec0[1];
		}
	}

	// from WildMagic
	template<typename T>
	inline T normalize(Eigen::Matrix<T, 3, 1> & vec, const T epsilon = (T)0.0)
	{
		T length = vec.norm();

		if (length > epsilon)
		{
			float invLength = 1.0f / length;
			vec[0] *= invLength;
			vec[1] *= invLength;
			vec[2] *= invLength;
		}
		else
		{
			length = 0.0f;
			vec[0] = 0.0f;
			vec[1] = 0.0f;
			vec[2] = 0.0f;
		}

		return length;
	}
}