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
	\file    Mat.hpp
	\brief   Matrix maths tools.
	\author  R. Kerviche
	\date    June 26th 2017
**/

#ifndef __KARTET_MATRIX_MATHS_TOOLS__
#define __KARTET_MATRIX_MATHS_TOOLS__

	#include "Core/LibTools.hpp"
	#include "Core/Meta.hpp"
	#include "Core/Complex.hpp"

namespace Kartet
{
	template<int r, int c, typename T>
	struct Mat
	{
		STATIC_ASSERT_VERBOSE((r*c)>1, INVALID_DIMENSION)
		typedef T BaseType;
		static const int	rows = r,
					cols = c,
					dim = r*c;
		T m[r*c];

		__host__ __device__ inline Mat(void);
		__host__ __device__ inline Mat(const T& val);
		__host__ __device__ inline Mat(const Mat<r, c, T>& o);
		template<typename U>
		__host__ __device__ inline Mat(const Mat<r, c, U>& o);
		template<typename U>
		__host__ __device__ inline Mat(const U* ptr);

		__host__ __device__ inline const T& operator()(const int& i, const int& j) const;
		__host__ __device__ inline T& operator()(const int& i, const int& j);
		__host__ __device__ inline const Mat<r,c,T>& operator=(const Mat<r, c, T>& o);
		template<typename U>
		__host__ __device__ inline const Mat<r,c,T>& operator=(const Mat<r, c, U>& o);
		__host__ __device__ inline const Mat<r,c,T>& clear(const T& val);

		// Tools :
		__host__ __device__ inline static Mat<r,c,T> identity(void);
	};

	// Functions :
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(void)
	{ } // Leave unitialized.

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(const T& val)
	{
		clear(val);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(const Mat<r,c,T>& o)
	{
		metaUnaryEqual<dim>(m, o.m);
	}

	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>::Mat(const Mat<r,c,U>& o)
	{
		metaUnaryEqual<dim>(m, o.m);
	}

	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>::Mat(const U* ptr)
	{
		metaUnaryEqual<dim>(m, ptr);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline const T& Mat<r,c,T>::operator()(const int& i, const int& j) const
	{
		return m[j*r+i];
	}

	template<int r, int c, typename T>
	__host__ __device__ inline T& Mat<r,c,T>::operator()(const int& i, const int& j)
	{
		return m[j*r+i];
	}

	template<int r, int c, typename T>
	__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::operator=(const Mat<r,c,T>& o)
	{
		metaUnaryEqual<dim>(this->m, o.m);
		return (*this);
	}

	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::operator=(const Mat<r,c,U>& o)
	{
		metaUnaryEqual<Mat<r,c,T>::dim>(this->m, o.m);
		return (*this);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::clear(const T& val)
	{
		metaUnaryEqual<Mat<r,c,T>::dim>(this->m, val);
		return (*this);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T> Mat<r,c,T>::identity(void)
	{
		Mat<r,c,T> res;
		for(int k=0; k<Mat<r,c,T>::dim; k++)
			res.m[k] = (k%(r+1)==0) ? static_cast<T>(1) : static_cast<T>(0);
		return res;
	}

	// Non-members :
	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator+(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryPlus<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator-(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryMinus<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator*(const Mat<r,c,T>& a, const U& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a.m, b);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator*(const T& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a, b.m);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> modulate(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	template<int ra, int ca, typename Ta, int rb, int cb, typename Tb>
	__host__ __device__ inline Mat<ra,cb,typename ResultingType<Ta,Tb>::Type> operator*(const Mat<ra,ca,Ta>& a, const Mat<rb,cb,Tb>& b)
	{
		STATIC_ASSERT_VERBOSE(ca==rb, INVALID_DIMENSION)
		Mat<ra,cb,typename ResultingType<Ta,Tb>::Type> res;
		for(int j=0; j<cb; j++)
			for(int i=0; i<ra; i++)
				res(i,j) = metaBinaryProductSum<ca, typename ResultingType<Ta,Tb>::Type>(a.m+i, b.m+j*rb, ra, 1);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator/(const Mat<r,c,T>& a, const U& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryQuotient<Mat<r,c,T>::dim>(res.m, a.m, b);
		return res;
	}

	template<int r, typename T>
	__host__ __device__ inline T trace(const Mat<r,r,T>& a)
	{
		return metaUnaryPlusSum<r>(a.m, r+1);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<c,r,T> transpose(const Mat<r,c,T>& a)
	{	
		Mat<c,r,T> res;
		for(int j=0; j<r; j++)
			for(int i=0; i<c; i++)
				res(i,j) = a(j,i);
		return res;
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<c,r,T> hermTranspose(const Mat<r,c,T>& a)
	{	
		Mat<c,r,T> res;
		for(int j=0; j<r; j++)
			for(int i=0; i<c; i++)
				res(i,j) = conj(a(j,i));
		return res;
	}

	template<int r, typename T, typename U>
	__host__ __device__ inline typename ResultingType<T,U>::Type dot(const Mat<r,1,T>& a, const Mat<r,1,U>& b)
	{
		return metaBinaryProductSum<r, typename ResultingType<T,U>::Type>(a.m, b.m);
	}

	template<int r, typename T, typename U>
	__host__ __device__ inline typename ResultingType<T,U>::Type hermDot(const Mat<r,1,T>& a, const Mat<r,1,U>& b)
	{
		return metaBinaryHermProductSum<r, typename ResultingType<T,U>::Type>(a.m, b.m);
	}

	template<int r, typename T>
	__host__ __device__ inline T frobenius(const Mat<r,r,T>& a)
	{
		return 0;
	}

	template<int r, typename T>
	__host__ __device__ inline T normSquared(const Mat<r,1,T>& v)
	{
		return metaUnaryAbsSquareSum<r>(v.m);
	}

	template<int r, typename T>
	__host__ __device__ inline T lengthSquared(const Mat<r,1,T>& v) // alias normSquared
	{
		return metaUnaryAbsSquareSum<r>(v.m);
	}

	template<int r, typename T>
	__host__ __device__ inline T norm(const Mat<r,1,T>& v)
	{
		return ::sqrt(metaUnaryAbsSquareSum<r>(v.m));
	}

	template<int r, typename T>
	__host__ __device__ inline T length(const Mat<r,1,T>& v) // alias norm
	{
		return ::sqrt(metaUnaryAbsSquareSum<r>(v.m));
	}

	template<int r, typename T>
	__host__ __device__ inline Mat<r,1,T> normalize(const Mat<r,1,T>& v)
	{
		return v/norm(v);
	}

	template<int r, typename T, typename U>
	__host__ __device__ inline Mat<r,1,typename ResultingType<T,U>::Type> reflect(const Mat<r,1,T>& dir, const Mat<r,1,U>& normal)
	{
		return dir - (static_cast<typename ResultingType<T,U>::Type>(2)*dot(dir, normal))*normal;
	}

	template<typename T, typename U>
	__host__ __device__ inline Mat<3,1,typename ResultingType<T,U>::Type> cross(const Mat<3,1,T>& a, const Mat<3,1,U>& b)
	{
		Mat<3,1,typename ResultingType<T,U>::Type> c;
		c.m[0] = a.m[1]*b.m[2]-a.m[2]*b.m[1];
		c.m[1] = a.m[2]*b.m[0]-a.m[0]*b.m[2];
		c.m[2] = a.m[0]*b.m[1]-a.m[1]*b.m[0];
		return c;
	}

	template<int r, int c, typename T>
	__host__ std::ostream& operator<<(std::ostream& os, const Mat<r,c,T>& v)
	{
		const T zero = static_cast<T>(0),
			minusZero = -static_cast<T>(0);
		const bool flag = !(os.flags() & std::ios_base::showpos);
		for(int i=0; i<r; i++)
		{
			os << " |";
			for(int j=0; j<(c-1); j++)
			{
				const T val = v(i,j);
				const bool f = (val>=zero) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(val, minusZero));
				if(flag && f)
					os << ' ';
				os << val << ", ";
			}
			const T val = v(i, c-1);
			const bool f = (val>=zero) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(val, minusZero));
			if(flag && f)
				os << ' ';
			os << val << " | " << std::endl;
		}
		return os;
	}
}

#endif

