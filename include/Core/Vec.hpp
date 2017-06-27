/* ************************************************************************************************************* */
/*                                                                                                               */
/*     Kartet                                                                                                    */
/*     A Simple C++ Array Library for CUDA                                                                       */
/*                                                                                                               */
/*     LICENSE : The MIT License                                                                                 */
/*     Copyright (c) 2015 Ronan Kerviche                                                                         */
/*                                                                                                               */
/*     Permission is hereby granted, free of charge, to any person obtaining a copy                              */
/*     of this software and associated documentation files (the "Software"), to deal                             */
/*     in the Software without restriction, including without limitation the rights                              */
/*     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell                                 */
/*     copies of the Software, and to permit persons to whom the Software is                                     */
/*     furnished to do so, subject to the following conditions:                                                  */
/*                                                                                                               */
/*     The above copyright notice and this permission notice shall be included in                                */
/*     all copies or substantial portions of the Software.                                                       */
/*                                                                                                               */
/*     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR                                */
/*     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,                                  */
/*     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE                               */
/*     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                                    */
/*     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,                             */
/*     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN                                 */
/*     THE SOFTWARE.                                                                                             */
/*                                                                                                               */
/* ************************************************************************************************************* */

/**
	\file    Vec.hpp
	\brief   Vector maths tools.
	\author  R. Kerviche
	\date    May 13th 2017
**/

#ifndef __KARTET_VECTOR_MATHS_TOOLS__
#define __KARTET_VECTOR_MATHS_TOOLS__

	#include "Core/LibTools.hpp"
	#include "Core/Meta.hpp"
	#include "Core/Mat.hpp"

namespace Kartet
{
	template<int d, typename T>
	struct Vec : public Mat<d,1,T>
	{
		__host__ __device__ inline const T& operator()(const int& i) const;
		__host__ __device__ inline T& operator()(const int& i);
		__host__ __device__ inline Vec(void);
		//__host__ __device__ inline Vec(const T& val); // Ambiguous
		__host__ __device__ inline Vec(const Mat<d,1,T>& o);
		template<typename U>
		__host__ __device__ inline Vec(const Mat<d,1,U>& o);
		template<typename U>
		__host__ __device__ inline Vec(const U* ptr);
		using Mat<d,1,T>::operator ();
		using Mat<d,1,T>::operator=;
		using Mat<d,1,T>::clear;

		__host__ __device__ inline const T& x(void) const;
		__host__ __device__ inline T& x(void);
		__host__ __device__ inline const T& y(void) const;
		__host__ __device__ inline T& y(void);
		__host__ __device__ inline const T& z(void) const;
		__host__ __device__ inline T& z(void);
		__host__ __device__ inline const T& w(void) const;
		__host__ __device__ inline T& w(void);

	};

	// Functions :
	template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(void)
	{ }

	/*template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(const T& val)
	 : Mat<d,1,T>(val)
	{ }*/

	template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(const Mat<d,1,T>& o)
	 : Mat<d,1,T>(o)
	{ }

	template<int d, typename T>
	template<typename U>
	__host__ __device__ inline Vec<d,T>::Vec(const Mat<d,1,U>& o)
	 : Mat<d,1,T>(o)
	{ }

	template<int d, typename T>
	template<typename U>
	__host__ __device__ inline Vec<d,T>::Vec(const U* ptr)
	 : Mat<d,1,T>(ptr)
	{ }

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::operator()(const int& i) const
	{
		return Mat<d,1,T>::m[i];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::operator()(const int& i)
	{
		return Mat<d,1,T>::m[i];
	}

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::x(void) const
	{
		return Mat<d,1,T>::m[0];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::x(void)
	{
		return Mat<d,1,T>::m[0];
	}

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::y(void) const
	{
		return Mat<d,1,T>::m[1];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::y(void)
	{
		return Mat<d,1,T>::m[1];
	}

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::z(void) const
	{
		STATIC_ASSERT_VERBOSE(d>2, INVALID_DIMENSION)
		return Mat<d,1,T>::m[2];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::z(void)
	{
		STATIC_ASSERT_VERBOSE(d>2, INVALID_DIMENSION)
		return Mat<d,1,T>::m[2];
	}

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::w(void) const
	{
		STATIC_ASSERT_VERBOSE(d>3, INVALID_DIMENSION)
		return Mat<d,1,T>::m[3];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::w(void)
	{
		STATIC_ASSERT_VERBOSE(d>3, INVALID_DIMENSION)
		return Mat<d,1,T>::m[3];
	}

	// Non-members :
	template<int ra, int ca, typename Ta, int rb, typename Tb>
	__host__ __device__ inline Vec<ra,typename ResultingType<Ta,Tb>::Type> operator*(const Mat<ra,ca,Ta>& a, const Vec<rb,Tb>& b)
	{
		STATIC_ASSERT_VERBOSE(ca==rb, INVALID_DIMENSION)
		Vec<ra,typename ResultingType<Ta,Tb>::Type> res;
		for(int i=0; i<ra; i++)
			res(i) = metaBinaryProductSum<ca, typename ResultingType<Ta,Tb>::Type>(a.m+i, b.m, ra, 1);
		return res;
	}

	template<int d, typename T>
	__host__ std::ostream& operator<<(std::ostream& os, const Vec<d,T>& v)
	{
		const T zero = static_cast<T>(0),
			minusZero = -static_cast<T>(0);
		const bool flag = !(os.flags() & std::ios_base::showpos);
		os << '(';
		for(int k=0; k<(d-1); k++)
		{
			const bool f = (v.m[k]>=zero) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(v.m[k], minusZero));
			if(flag && f)
				os << ' ';
			os << v.m[k] << ", ";
		}
		const bool f = (v.m[d-1]>=zero) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(v.m[d-1], minusZero));
		if(flag && f)
			os << ' ';
		os << v.m[d-1] << ')';
		return os;
	}
}

#endif

