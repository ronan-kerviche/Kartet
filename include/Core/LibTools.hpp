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
	\file    LibTools.hpp
	\brief   Library tools.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_LIBRARY_TOOLS__
#define __KARTET_LIBRARY_TOOLS__

	#define UNUSED_PARAMETER(x) (void)(x);
	#define K_PI  (3.14159265358979323846)
	#define K_2PI (6.28318530717958647692)
	#define K_L10 (2.30258509299404568401)
	#define K_L2  (0.69314718055994530941)

	// Alignment :
	#if defined(__GNUC__)
		#define KARTET_ALIGN(b) __attribute__ ((aligned(b)))
	#elif defined(_MSC_VER)
		#define KARTET_ALIGN(b) __declspec(align(b))
	#else
		#warning "Kartet structures alignment not enforced."
		#define KARTET_ALIGN(b)
	#endif

	#ifdef __CUDACC__
		#define __cuda_typename typename
	#else
		#define __cuda_typename 
		#define __host__ 
		#define __device__ 
		#define __global__ 
		#define __shared__ 
		#define cudaStream_t void*

	// Common missing structures :
		struct char2
		{
			char x, y;
		} KARTET_ALIGN(2);

		struct char3
		{
			char x, y, z;
		} KARTET_ALIGN(4);

		struct char4
		{
			char x, y, z, w;
		} KARTET_ALIGN(4);

		struct uchar2
		{
			unsigned char x, y;
		} KARTET_ALIGN(2);

		struct uchar3
		{
			unsigned char x, y, z;
		} KARTET_ALIGN(4);

		struct uchar4
		{
			unsigned char x, y, z, w;
		} KARTET_ALIGN(4);

		struct short2
		{
			short x, y;
		} KARTET_ALIGN(4);

		struct short3
		{
			short x, y, z;
		} KARTET_ALIGN(8);

		struct short4
		{
			short x, y, z, w;
		} KARTET_ALIGN(8);

		struct ushort2
		{
			unsigned short x, y;
		} KARTET_ALIGN(4);

		struct ushort3
		{
			unsigned short x, y, z;
		} KARTET_ALIGN(8);

		struct ushort4
		{
			unsigned short x, y, z, w;
		} KARTET_ALIGN(8);

		struct int2
		{
			int x, y;
		} KARTET_ALIGN(8);

		struct int3
		{
			int x, y, z;
		} KARTET_ALIGN(16);

		struct int4
		{
			int x, y, z, w;
		} KARTET_ALIGN(16);

		struct uint2
		{
			unsigned int x, y;
		} KARTET_ALIGN(8);

		struct uint3
		{
			unsigned int x, y, z;
		} KARTET_ALIGN(16);

		struct uint4
		{
			unsigned int x, y, z, w;
		} KARTET_ALIGN(16);

		struct float2
		{
			float x, y;
		} KARTET_ALIGN(8);

		struct float3
		{
			float x, y, z;
		} KARTET_ALIGN(16);

		struct float4
		{
			float x, y, z, w;
		} KARTET_ALIGN(16);

		struct double2
		{
			double x, y;
		} KARTET_ALIGN(16);

		struct double3
		{
			double x, y, z;
		} KARTET_ALIGN(32);

		struct double4
		{
			double x, y, z, w;
		} KARTET_ALIGN(32);

		struct dim3
		{
			unsigned int x, y, z;
			inline dim3(const unsigned int& _x = 1, const unsigned int& _y = 1, const unsigned int& _z = 1)
			 : x(_x), y(_y), z(_z)
			{ }
		};

	// Missing functions :
		template<typename T>
		T min(const T& a, const T& b)
		{
			return (a<=b) ? a : b;
		}

		template<typename T>
		T max(const T& a, const T& b)
		{
			return (a>=b) ? a : b;
		}
	#endif

	// Other tools :
	template<typename T1, typename T2>
	bool compareBits(const T1& x1, const T2& x2)
	{
		bool t = true;
		const size_t m = min(sizeof(T1), sizeof(T2));
		for(size_t k=0; k<m; k++)
		{
			const char *a = reinterpret_cast<const char*>(&x1)+k,
				   *b = reinterpret_cast<const char*>(&x2)+k;
			t = t && (*a==*b);
		}
		return (sizeof(T1)==sizeof(T2)) && t;
	}

#endif

