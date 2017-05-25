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

namespace Kartet
{
	template<int d, typename T>
	struct Vec
	{
		STATIC_ASSERT_VERBOSE(d>1, INVALID_DIMENSION)
		typedef T BaseType;
		static const int dim = d;
		T m[d];

		__host__ __device__ inline Vec(void);
		__host__ __device__ inline Vec(const T& val);
		__host__ __device__ inline Vec(const Vec<d, T>& c);
		template<typename U>
		__host__ __device__ inline Vec(const Vec<d, U>& c);
		template<typename U>
		__host__ __device__ inline Vec(const U* ptr);

		__host__ __device__ inline const T& x(void) const;
		__host__ __device__ inline T& x(void);
		__host__ __device__ inline const T& y(void) const;
		__host__ __device__ inline T& y(void);
		__host__ __device__ inline const T& z(void) const;
		__host__ __device__ inline T& z(void);
		__host__ __device__ inline const T& w(void) const;
		__host__ __device__ inline T& w(void);

		__host__ __device__ inline const Vec<d,T>& operator=(const Vec<d, T>& c);
		template<typename U>
		__host__ __device__ inline const Vec<d,T>& operator=(const Vec<d, U>& c);
		__host__ __device__ inline const Vec<d,T>& clear(const T& val);
	};

	// Functions :
	template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(void)
	{ } // Leave unitialized.

	template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(const T& val)
	{
		clear(val);
	}

	template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(const Vec<d,T>& c)
	{
		metaUnaryEqual<d>(m, c.m);
	}

	template<int d, typename T>
	template<typename U>
	__host__ __device__ inline Vec<d,T>::Vec(const Vec<d,U>& c)
	{
		metaUnaryEqual<d>(m, c.m);
	}

	template<int d, typename T>
	template<typename U>
	__host__ __device__ inline Vec<d,T>::Vec(const U* ptr)
	{
		metaUnaryEqual<d>(m, ptr);
	}

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::x(void) const
	{
		return m[0];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::x(void)
	{
		return m[0];
	}

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::y(void) const
	{
		return m[1];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::y(void)
	{
		return m[1];
	}

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::z(void) const
	{
		STATIC_ASSERT_VERBOSE(d>2, INVALID_DIMENSION)
		return m[2];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::z(void)
	{
		STATIC_ASSERT_VERBOSE(d>2, INVALID_DIMENSION)
		return m[2];
	}

	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::w(void) const
	{
		STATIC_ASSERT_VERBOSE(d>3, INVALID_DIMENSION)
		return m[3];
	}

	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::w(void)
	{
		STATIC_ASSERT_VERBOSE(d>3, INVALID_DIMENSION)
		return m[3];
	}

	template<int d, typename T>
	__host__ __device__ inline const Vec<d,T>& Vec<d,T>::operator=(const Vec<d,T>& c)
	{
		metaUnaryEqual<d>(this->m, c.m);
		return (*this);
	}

	template<int d, typename T>
	template<typename U>
	__host__ __device__ inline const Vec<d,T>& Vec<d,T>::operator=(const Vec<d,U>& c)
	{
		metaUnaryEqual<d>(this->m, c.m);
		return (*this);
	}

	template<int d, typename T>
	__host__ __device__ inline const Vec<d,T>& Vec<d,T>::clear(const T& val)
	{
		metaUnaryEqual<d>(this->m, val);
		return (*this);
	}

	// Non-members :
	template<int d, typename T, typename U>
	__host__ __device__ inline Vec<d,typename ResultingType<T,U>::Type> operator+(const Vec<d,T>& a, const Vec<d,U>& b)
	{
		Vec<d,typename ResultingType<T,U>::Type> r;
		metaBinaryPlus<d>(r.m, a.m, b.m);
		return r;
	}

	template<int d, typename T, typename U>
	__host__ __device__ inline Vec<d,typename ResultingType<T,U>::Type> operator-(const Vec<d,T>& a, const Vec<d,U>& b)
	{
		Vec<d,typename ResultingType<T,U>::Type> r;
		metaBinaryMinus<d>(r.m, a.m, b.m);
		return r;
	}

	template<int d, typename T, typename U>
	__host__ __device__ inline Vec<d,typename ResultingType<T,U>::Type> operator*(const Vec<d,T>& a, const U& b)
	{
		Vec<d,typename ResultingType<T,U>::Type> r;
		metaBinaryProduct<d>(r.m, a.m, b.m);
		return r;
	}

	template<int d, typename T, typename U>
	__host__ __device__ inline Vec<d,typename ResultingType<T,U>::Type> operator*(const T& a, const Vec<d,U>& b)
	{
		Vec<d,typename ResultingType<T,U>::Type> r;
		metaBinaryProduct<d>(r.m, a, b.m);
		return r;
	}

	template<int d, typename T, typename U>
	__host__ __device__ inline Vec<d,typename ResultingType<T,U>::Type> operator/(const Vec<d,T>& a, const U& b)
	{
		Vec<d,typename ResultingType<T,U>::Type> r;
		metaBinaryQuotient<d>(r.m, a.m, b);
		return r;
	}

	template<int d, typename T, typename U>
	__host__ __device__ inline typename ResultingType<T,U>::Type dot(const Vec<d,T>& a, const Vec<d,U>& b)
	{
		return metaBinaryProductSum<d, typename ResultingType<T,U>::Type>(a.m, b.m);
	}

	template<int d, typename T>
	__host__ __device__ inline T normSquared(const Vec<d,T>& v)
	{
		return metaUnarySquareSum<d>(v.m);
	}

	template<int d, typename T>
	__host__ __device__ inline T lengthSquared(const Vec<d,T>& v) // alias normSquared
	{
		return metaUnarySquareSum<d>(v.m);
	}

	template<int d, typename T>
	__host__ __device__ inline T norm(const Vec<d,T>& v)
	{
		return ::sqrt(metaUnarySquareSum<d>(v.m));
	}

	template<int d, typename T>
	__host__ __device__ inline T length(const Vec<d,T>& v) // alias norm
	{
		return ::sqrt(metaUnarySquareSum<d>(v.m));
	}

	template<int d, typename T>
	__host__ __device__ inline Vec<d,T> normalize(const Vec<d,T>& v)
	{
		return v/norm(v);
	}

	template<int d, typename T, typename U>
	__host__ __device__ inline Vec<d,typename ResultingType<T,U>::Type> reflect(const Vec<d,T>& dir, const Vec<d,U>& normal)
	{
		return dir - (static_cast<typename ResultingType<T,U>::Type>(2)*dot(dir, normal))*normal;
	}

	template<typename T, typename U>
	__host__ __device__ inline Vec<3,typename ResultingType<T,U>::Type> cross(const Vec<3,T>& a, const Vec<3,U>& b)
	{
		Vec<3,typename ResultingType<T,U>::Type> c;
		c.m[0] = a.m[1]*b.m[2]-a.m[2]*b.m[1];
		c.m[1] = a.m[2]*b.m[0]-a.m[0]*b.m[2];
		c.m[2] = a.m[0]*b.m[1]-a.m[1]*b.m[0];
		return c;
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

// For builtin types :
	// Tools :
	#ifdef __CUDACC__
		#define NORMALIZE(a) (a*::rsqrtf(normSquared(a)))
	#else
		#define NORMALIZE(a) (a/norm(a))
	#endif
	// Maths :
	#define VEC2_TOOLS(NAME, T, TBase, SQRTFUN) \
		__host__ __device__ inline T make##NAME##2(const TBase& x, const TBase& y) \
		{ \
			T c; \
			c.x = x; \
			c.y = y; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator-(const T& a) \
		{ \
			T c; \
			c.x = -a.x; \
			c.y = -a.y; \
			return c; \
		} \
		__host__ __device__ inline T operator+(const T& a, const T& b) \
		{ \
			T c; \
			c.x = a.x+b.x; \
			c.y = a.y+b.y; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator-(const T& a, const T& b) \
		{ \
			T c; \
			c.x = a.x-b.x; \
			c.y = a.y-b.y; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator*(const T& a, const TBase& s) \
		{ \
			T c; \
			c.x = a.x*s; \
			c.y = a.y*s; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator*(const TBase& s, const T& b) \
		{ \
			T c; \
			c.x = s*b.x; \
			c.y = s*b.y; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator/(const T& a, const TBase& s) \
		{ \
			T c; \
			c.x = a.x/s; \
			c.y = a.y/s; \
			return c; \
		} \
		 \
		__host__ __device__ inline T& operator+=(T& a, const T& b) \
		{ \
			a.x+=b.x; \
			a.y+=b.y; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator-=(T& a, const T& b) \
		{ \
			a.x-=b.x; \
			a.y-=b.y; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator*=(T& a, const TBase& s) \
		{ \
			a.x*=s; \
			a.y*=s; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator/=(T& a, const TBase& s) \
		{ \
			a.x/=s; \
			a.y/=s; \
			return a; \
		} \
		 \
		__host__ __device__ inline TBase dot(const T& a, const T& b) \
		{ \
			return a.x*b.x + a.y*b.y; \
		} \
		 \
		__host__ __device__ inline TBase normSquared(const T& a) \
		{ \
			return a.x*a.x + a.y*a.y; \
		} \
		 \
		__host__ __device__ inline TBase norm(const T& a) \
		{ \
			return SQRTFUN(normSquared(a)); \
		} \
		 \
		__host__ __device__ inline T normalize(const T& a) \
		{ \
			return NORMALIZE(a); \
		} \
		 \
		__host__ __device__ inline T reflect(const T& dir, const T& normal) \
		{ \
			return dir - (2.0f*dot(dir, normal))*normal; \
		} \

	#define VEC2_TOOLS_SIGNED(NAME, T, TBase, SQRTFUN)	VEC2_TOOLS(NAME, T, TBase, SQRTFUN) \
		__host__ inline std::ostream& operator<<(std::ostream& os, const T& z) \
		{ \
			const TBase 	zero = static_cast<TBase>(0), \
					minusZero = -static_cast<TBase>(0); \
			const bool 	typeTest = Kartet::IsSame<TBase, float>::value || Kartet::IsSame<TBase, double>::value, \
					px = (z.x>=zero) && !(typeTest && compareBits(z.x, minusZero)), \
					py = (z.y>=zero) && !(typeTest && compareBits(z.y, minusZero)), \
					flag = !(os.flags() & std::ios_base::showpos); \
			os << '('; \
			if(flag && px) \
				os << ' '; \
			os << z.x << ", "; \
			if(flag && py) \
				os << ' '; \
			os << z.y << ')'; \
			return os; \
		}

	#define VEC2_TOOLS_UNSIGNED(NAME, T, TBase, SQRTFUN)	VEC2_TOOLS(NAME, T, TBase, SQRTFUN) \
		__host__ inline std::ostream& operator<<(std::ostream& os, const T& z) \
		{ \
			const bool flag = !(os.flags() & std::ios_base::showpos); \
			os << '('; \
			if(flag) \
				os << ' '; \
			os << z.x << ", "; \
			if(flag) \
				os << ' '; \
			os << z.y << ')'; \
			return os; \
		}

		VEC2_TOOLS_SIGNED(	Char,		char2, 		char,		::sqrtf)
		VEC2_TOOLS_UNSIGNED(	UChar,		uchar2,		unsigned char,	::sqrtf)
		VEC2_TOOLS_SIGNED(	Short,		short2, 	short,		::sqrtf)
		VEC2_TOOLS_UNSIGNED(	UShort,		ushort2,	unsigned short,	::sqrtf)
		VEC2_TOOLS_SIGNED(	Int,		int2,		int,		::sqrtf)
		VEC2_TOOLS_UNSIGNED(	UInt,		uint2,		unsigned int,	::sqrtf)
		VEC2_TOOLS_SIGNED(	Float,		float2, 	float,		::sqrtf)
		VEC2_TOOLS_SIGNED(	Double,		double2,	double,		::sqrt)

	#undef VEC2_TOOLS_UNSIGNED 
	#undef VEC2_TOOLS_SIGNED
	#undef VEC2_TOOLS

	#define VEC3_TOOLS(NAME, T, TBase, SQRTFUN) \
		__host__ __device__ inline T make##NAME##3(const TBase& x, const TBase& y, const TBase& z) \
		{ \
			T c; \
			c.x = x; \
			c.y = y; \
			c.z = z; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator-(const T& a) \
		{ \
			T c; \
			c.x = -a.x; \
			c.y = -a.y; \
			c.z = -a.z; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator+(const T& a, const T& b) \
		{ \
			T c; \
			c.x = a.x+b.x; \
			c.y = a.y+b.y; \
			c.z = a.z+b.z; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator-(const T& a, const T& b) \
		{ \
			T c; \
			c.x = a.x-b.x; \
			c.y = a.y-b.y; \
			c.z = a.z-b.z; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator*(const T& a, const TBase& s) \
		{ \
			T c; \
			c.x = a.x*s; \
			c.y = a.y*s; \
			c.z = a.z*s; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator*(const TBase& s, const T& b) \
		{ \
			T c; \
			c.x = s*b.x; \
			c.y = s*b.y; \
			c.z = s*b.z; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator/(const T& a, const TBase& s) \
		{ \
			T c; \
			c.x = a.x/s; \
			c.y = a.y/s; \
			c.z = a.z/s; \
			return c; \
		} \
		 \
		__host__ __device__ inline T& operator+=(T& a, const T& b) \
		{ \
			a.x+=b.x; \
			a.y+=b.y; \
			a.z+=b.z; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator-=(T& a, const T& b) \
		{ \
			a.x-=b.x; \
			a.y-=b.y; \
			a.z-=b.z; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator*=(T& a, const TBase& s) \
		{ \
			a.x*=s; \
			a.y*=s; \
			a.z*=s; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator/=(T& a, const TBase& s) \
		{ \
			a.x/=s; \
			a.y/=s; \
			a.z/=s; \
			return a; \
		} \
		 \
		__host__ __device__ inline TBase dot(const T& a, const T& b) \
		{ \
			return a.x*b.x + a.y*b.y + a.z*b.z; \
		} \
		 \
		__host__ __device__ inline TBase normSquared(const T& a) \
		{ \
			return a.x*a.x + a.y*a.y + a.z*a.z; \
		} \
		 \
		__host__ __device__ inline TBase norm(const T& a) \
		{ \
			return ::sqrtf(normSquared(a)); \
		} \
		 \
		__host__ __device__ inline T normalize(const T& a) \
		{ \
			return NORMALIZE(a); \
		} \
		 \
		__host__ __device__ inline T cross(const T& a, const T& b) \
		{ \
			T c; \
			c.x = a.y*b.z-a.z*b.y; \
			c.y = a.z*b.x-a.x*b.z; \
			c.z = a.x*b.y-a.y*b.x; \
			return c; \
		} \
		 \
		__host__ __device__ inline T reflect(const T& dir, const T& normal) \
		{ \
			return dir - (2.0f*dot(dir, normal))*normal; \
		} \
	
	#define VEC3_TOOLS_SIGNED(NAME, T, TBase, SQRTFUN)	VEC3_TOOLS(NAME, T, TBase, SQRTFun) \
		__host__ inline std::ostream& operator<<(std::ostream& os, const T& z) \
		{ \
			const TBase 	zero = static_cast<TBase>(0), \
					minusZero = -static_cast<TBase>(0); \
			const bool 	typeTest = Kartet::IsSame<TBase, float>::value || Kartet::IsSame<TBase, double>::value, \
					px = (z.x>=zero) && !(typeTest && compareBits(z.x, minusZero)), \
					py = (z.y>=zero) && !(typeTest && compareBits(z.y, minusZero)), \
					pz = (z.z>=zero) && !(typeTest && compareBits(z.z, minusZero)), \
					flag = !(os.flags() & std::ios_base::showpos); \
			os << '('; \
			if(flag && px) \
				os << ' '; \
			os << z.x << ", "; \
			if(flag && py) \
				os << ' '; \
			os << z.y << ", "; \
			if(flag && pz) \
				os << ' '; \
			os << z.z << ')'; \
			return os; \
		}

	#define VEC3_TOOLS_UNSIGNED(NAME, T, TBase, SQRTFUN)	VEC3_TOOLS(NAME, T, TBase, SQRTFun) \
		__host__ inline std::ostream& operator<<(std::ostream& os, const T& z) \
		{ \
			const bool flag = !(os.flags() & std::ios_base::showpos); \
			os << '('; \
			if(flag) \
				os << ' '; \
			os << z.x << ", "; \
			if(flag) \
				os << ' '; \
			os << z.y << ", "; \
			if(flag) \
				os << ' '; \
			os << z.z << ')'; \
			return os; \
		}

		VEC3_TOOLS_SIGNED(	Char,		char3, 		char,		::sqrtf)
		VEC3_TOOLS_UNSIGNED(	UChar,		uchar3,		unsigned char,	::sqrtf)
		VEC3_TOOLS_SIGNED(	Short,		short3, 	short,		::sqrtf)
		VEC3_TOOLS_UNSIGNED(	UShort,		ushort3,	unsigned short,	::sqrtf)
		VEC3_TOOLS_SIGNED(	Int,		int3,		int,		::sqrt)
		VEC3_TOOLS_UNSIGNED(	UInt,		uint3,		unsigned int,	::sqrt)
		VEC3_TOOLS_SIGNED(	Float,		float3, 	float,		::sqrtf)
		VEC3_TOOLS_SIGNED(	Double,		double3,	double,		::sqrt)

	#undef VEC3_TOOLS_SIGNED
	#undef VEC3_TOOLS_UNSIGNED
	#undef VEC3_TOOLS

	#define VEC4_TOOLS(NAME, T, TBase, SQRTFUN) \
		__host__ __device__ inline T make##NAME##4(const TBase& x, const TBase& y, const TBase& z, const TBase& w) \
		{ \
			T c; \
			c.x = x; \
			c.y = y; \
			c.z = z; \
			c.w = w; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator-(const T& a) \
		{ \
			T c; \
			c.x = -a.x; \
			c.y = -a.y; \
			c.z = -a.z; \
			c.w = -a.w; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator+(const T& a, const T& b) \
		{ \
			T c; \
			c.x = a.x+b.x; \
			c.y = a.y+b.y; \
			c.z = a.z+b.z; \
			c.w = a.w+b.w; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator-(const T& a, const T& b) \
		{ \
			T c; \
			c.x = a.x-b.x; \
			c.y = a.y-b.y; \
			c.z = a.z-b.z; \
			c.w = a.w-b.w; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator*(const T& a, const TBase& s) \
		{ \
			T c; \
			c.x = a.x*s; \
			c.y = a.y*s; \
			c.z = a.z*s; \
			c.w = a.w*s; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator*(const TBase& s, const T& b) \
		{ \
			T c; \
			c.x = s*b.x; \
			c.y = s*b.y; \
			c.z = s*b.z; \
			c.w = s*b.w; \
			return c; \
		} \
		 \
		__host__ __device__ inline T operator/(const T& a, const TBase& s) \
		{ \
			T c; \
			c.x = a.x/s; \
			c.y = a.y/s; \
			c.z = a.z/s; \
			c.w = a.w/s; \
			return c; \
		} \
		 \
		__host__ __device__ inline T& operator+=(T& a, const T& b) \
		{ \
			a.x+=b.x; \
			a.y+=b.y; \
			a.z+=b.z; \
			a.w+=b.w; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator-=(T& a, const T& b) \
		{ \
			a.x-=b.x; \
			a.y-=b.y; \
			a.z-=b.z; \
			a.w-=b.w; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator*=(T& a, const TBase& s) \
		{ \
			a.x*=s; \
			a.y*=s; \
			a.z*=s; \
			a.w*=s; \
			return a; \
		} \
		 \
		__host__ __device__ inline T& operator/=(T& a, const TBase& s) \
		{ \
			a.x/=s; \
			a.y/=s; \
			a.z/=s; \
			a.w/=s; \
			return a; \
		} \
		 \
		__host__ __device__ inline TBase dot(const T& a, const T& b) \
		{ \
			return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; \
		} \
		 \
		__host__ __device__ inline TBase normSquared(const T& a) \
		{ \
			return a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w; \
		} \
		 \
		__host__ __device__ inline TBase norm(const T& a) \
		{ \
			return SQRTFUN(normSquared(a)); \
		} \
		 \
		__host__ __device__ inline T normalize(const T& a) \
		{ \
			return NORMALIZE(a); \
		} \
		 \
		__host__ __device__ inline T reflect(const T& dir, const T& normal) \
		{ \
			return dir - (2.0f*dot(dir, normal))*normal; \
		} \

	#define VEC4_TOOLS_SIGNED(NAME, T, TBase, SQRTFUN)	VEC4_TOOLS(NAME, T, TBase, SQRTFUN) \
		__host__ inline std::ostream& operator<<(std::ostream& os, const T& z) \
		{ \
			const TBase 	zero = static_cast<TBase>(0), \
					minusZero = -static_cast<TBase>(0); \
			const bool 	typeTest = Kartet::IsSame<TBase, float>::value || Kartet::IsSame<TBase, double>::value, \
					px = (z.x>=zero) && !(typeTest && compareBits(z.x, minusZero)), \
					py = (z.y>=zero) && !(typeTest && compareBits(z.y, minusZero)), \
					pz = (z.z>=zero) && !(typeTest && compareBits(z.z, minusZero)), \
					pw = (z.w>=zero) && !(typeTest && compareBits(z.w, minusZero)), \
					flag = !(os.flags() & std::ios_base::showpos); \
			os << '('; \
			if(flag && px) \
				os << ' '; \
			os << z.x << ", "; \
			if(flag && py) \
				os << ' '; \
			os << z.y << ", "; \
			if(flag && pz) \
				os << ' '; \
			os << z.z << ", "; \
			if(flag && pw) \
				os << ' '; \
			os << z.w << ')'; \
			return os; \
		}

	#define VEC4_TOOLS_UNSIGNED(NAME, T, TBase, SQRTFUN)	VEC4_TOOLS(NAME, T, TBase, SQRTFUN) \
		__host__ inline std::ostream& operator<<(std::ostream& os, const T& z) \
		{ \
			const bool flag = !(os.flags() & std::ios_base::showpos); \
			os << '('; \
			if(flag) \
				os << ' '; \
			os << z.x << ", "; \
			if(flag) \
				os << ' '; \
			os << z.y << ", "; \
			if(flag) \
				os << ' '; \
			os << z.z << ", "; \
			if(flag) \
				os << ' '; \
			os << z.w << ')'; \
			return os; \
		}

		VEC4_TOOLS_SIGNED(	Char,		char4, 		char,		::sqrtf)
		VEC4_TOOLS_UNSIGNED(	Uchar,		uchar4,		unsigned char,	::sqrtf)
		VEC4_TOOLS_SIGNED(	Short,		short4, 	short,		::sqrtf)
		VEC4_TOOLS_UNSIGNED(	UShort,		ushort4,	unsigned short,	::sqrtf)
		VEC4_TOOLS_SIGNED(	Int,		int4,		int,		::sqrtf)
		VEC4_TOOLS_UNSIGNED(	UInt,		uint4,		unsigned int,	::sqrtf)
		VEC4_TOOLS_SIGNED(	Float,		float4, 	float,		::sqrtf)
		VEC4_TOOLS_SIGNED(	Double,		double4,	double,		::sqrt)
	#undef VEC4_TOOLS_UNSIGNED
	#undef VEC4_TOOLS_SIGNED
	#undef VEC4_TOOLS

#endif

