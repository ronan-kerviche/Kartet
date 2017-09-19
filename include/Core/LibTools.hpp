/* ************************************************************************************************************* */
/*                                                                                                               */
/*     Kartet                                                                                                    */
/*     A Simple C++ Array Library for CUDA                                                                       */
/*                                                                                                               */
/*     LICENSE : The MIT License                                                                                 */
/*     Copyright (c) 2015-2017 Ronan Kerviche                                                                    */
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
	#define K_PI	(3.14159265358979323846)
	#define K_PIf	(3.14159265358979323846f)
	#define K_2PI	(6.28318530717958647692)
	#define K_2PIf	(6.28318530717958647692f)
	#define K_L10	(2.30258509299404568401)
	#define K_L10f	(2.30258509299404568401f)
	#define K_L2	(0.69314718055994530941)
	#define K_L2f	(0.69314718055994530941f)

	// Alignment :
	#if defined(__GNUC__)
		#define KARTET_ALIGN(b) __attribute__ ((aligned(b)))
	#elif defined(_MSC_VER)
		#define KARTET_ALIGN(b) __declspec(align(b))
	#else
		#warning "Kartet structures alignment not enforced."
		#define KARTET_ALIGN(b)
	#endif

	// Anonymous structures support :
	#if defined(__GNUC__) && !defined( __STRICT_ANSI__ )
		#define KARTET_SUPPORTS_ANON_STRUCT 1
		#define KARTET_ANON_STRUCT __extension__
	#elif defined(_WIN32) && (_MSC_VER>=1500)
		// Microsoft Developer Studio 2008 supports anonymous structs, but complains by default.
		#define KARTET_SUPPORTS_ANON_STRUCT 1
		#define KARTET_ANON_STRUCT
		#pragma warning( push )
		#pragma warning( disable : 4201 )
	#else
		#define KARTET_SUPPORTS_ANON_STRUCT 0
		#define KARTET_ANON_STRUCT
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

	#ifdef KARTET_USE_OMP
		#define OMP_PARALLEL_STATIC _Pragma("omp parallel for schedule(static)")
	#else
		#define OMP_PARALLEL_STATIC
	#endif

	// Other tools :
namespace Kartet
{
	template<typename T>
	void swap(T& a, T& b)
	{
		const T c = a;
		a = b;
		b = c;
	}

	template<typename T>
	bool softLargerThan(const T& a, const T& b)
	{
		return (a>b);
	}

	template<typename T>
	bool softLargerEqual(const T& a, const T& b)
	{
		return (a>=b);
	}

	template<typename T>
	bool softSmallerThan(const T& a, const T& b)
	{
		return (a<b);
	}

	template<typename T>
	bool softSmallerEqual(const T& a, const T& b)
	{
		return (a<=b);
	}

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
}

#endif

