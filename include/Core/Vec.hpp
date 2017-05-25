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

