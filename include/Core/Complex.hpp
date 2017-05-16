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
	\file    Complex.hpp
	\brief   Complex implementation.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_COMPLEX__
#define __KARTET_COMPLEX__

	#include <cmath>
	#include <iostream>
	#include "Core/LibTools.hpp"
	#include "Core/Meta.hpp"

#ifdef __CUDACC__
	#include <cufft.h>
#else
	#ifdef __cplusplus
	extern "C"
	{
	#endif
		typedef struct cuFloatComplex
		{
			float x, y;
		} cuFloatComplex;

		typedef struct cuDoubleComplex
		{
			double x, y;
		} cuDoubleComplex;
	#ifdef __cplusplus
	}
	#endif
#endif

namespace Kartet
{
	// Required prototype :
	template<typename T1, typename T2>
	struct ResultingType;

	// Complex : 
	/**
	\brief Complex class.
	
	Usual operators are also defined and implemented. Kartet::Complex<float> and Kartet::Complex<double> are binary compatible respectively with cuFloatComplex and cuDoubleComplex. Note that std::complex might not work with Cuda.

	Example : 
	\code
	Kartet::Complex<float>	z1(-3.0f, 2),
				z2(4, 1.0);
	std::cout << (z1*2+3*z2)*z2 << std::endl;
	std::cout << abs(z1) << std::endl;
	std::cout << conj(z1)/z2 << std::endl;
	\endcode
	**/
	template<typename T>
	struct Complex
	{
		typedef T BaseType;
		
			/// Real part.
		T 	x, 
			/// Imaginary part.
			y; 

		// Constructors : 
		/**
		\brief Default constructor to zero.
		**/
		__host__ __device__ Complex(void)
		 : x(0), y(0)
		{ }

		/**
		\brief Copy constructor.
		\param z Original complex.
		**/
		__host__ __device__ Complex(const Complex<T>& z)
		 : x(z.x), y(z.y)
		{ }

		/**
		\brief Copy constructor.
		\param z Original complex.
		**/
		template<typename T2>
		__host__ __device__ Complex(const Complex<T2>& z)
		 : x(z.x), y(z.y)
		{ }

		/**
		\brief Constructor.
		\param _x Real part.
		\param _y Imaginary part.
		**/
		template<typename T2>
		__host__ __device__ Complex(const T2& _x, const T2& _y=0)
		 : x(_x), y(_y)
		{ }

		/**
		\brief Constructor.
		\param _x Real part.
		\param _y Imaginary part.
		**/
		template<typename T2, typename T3>
		__host__ __device__ Complex(const T2& _x, const T3& _y=0)
		 : x(_x), y(_y)
		{ }

		/**
		\brief Copy constructor.
		\param z Original complex.
		**/
		__host__ __device__ Complex(const cuFloatComplex& z)
		 : x(z.x), y(z.y)
		{ }

		/**
		\brief Copy constructor.
		\param z Original complex.
		**/
		__host__ __device__ Complex(const cuDoubleComplex& z)
		 : x(z.x), y(z.y)
		{ }

		// Member functions : 
		/**
		\return Constant reference to the real part.
		**/
		__host__ __device__ const T& real(void) const
		{
			return x;
		}

		/**
		\return Reference to the real part.
		**/
		__host__ __device__ T& real(void)
		{
			return x;
		}

		/**
		\return Constant reference to the imaginary part.
		**/
		__host__ __device__ const T& imag(void) const
		{
			return y;
		}

		/**
		\return Reference to the imaginary part.
		**/
		__host__ __device__ T& imag(void)
		{
			return y;
		}

		/// \cond FALSE
		#define COUMPOUND_ASSIGNMENT( operator, CxOperation, ReOperation ) \
			__host__ __device__ Complex<T>& operator (const Complex<T>& z) \
			{ \
				CxOperation \
				return (*this); \
			} \
			 \
			template<typename T2> \
			__host__ __device__ Complex<T>& operator (const Complex<T2>& z) \
			{ \
				CxOperation \
				return (*this); \
			} \
			 \
			__host__ __device__ Complex<T>& operator (const cuFloatComplex& z) \
			{ \
				CxOperation \
				return (*this); \
			} \
			 \
			__host__ __device__ Complex<T>& operator (const cuDoubleComplex& z) \
			{ \
				CxOperation \
				return (*this); \
			} \
			 \
			template<typename T2> \
			__host__ __device__ Complex<T>& operator (const T2& r) \
			{ \
				ReOperation \
				return (*this); \
			}

			COUMPOUND_ASSIGNMENT( operator=,	x=z.x; y=z.y;,											x=r; y=0; )
			COUMPOUND_ASSIGNMENT( operator+=,	x+=z.x; y+=z.y;,										x+=r; )
			COUMPOUND_ASSIGNMENT( operator-=,	x-=z.x; y-=z.y;,										x-=r; )
			COUMPOUND_ASSIGNMENT( operator*=,	const T _x = x; x=_x*z.x-y*z.y; y=_x*z.y+y*z.x;,						x*=r; y*=r;)
			COUMPOUND_ASSIGNMENT( operator/=,	const T _x = x; x=(_x*z.x+y*z.y)/(z.x*z.x + z.y*z.y); y=(y*z.x-_x*z.y)/(z.x*z.x + z.y*z.y);,	x/=r; y/=r;)
		#undef COUMPOUND_ASSIGNMENT
		/// \endcond

		/**
		\return The inverse of the complex.
		**/
		Complex<T> operator-(void) const
		{
			return Complex<T>(-x, -y);
		}
	};

	/**
	\tparam T Base type.
	\return The imaginary unit.
	\related Kartet::Complex
	**/
	template<typename T>
	__host__ __device__ Complex<T> I(void)
	{
		return Complex<T>(0, 1);
	}

	/**
	\return The imaginary unit.
	\related Kartet::Complex
	**/
	__host__ __device__ inline Complex<double> I(void)
	{
		return Complex<double>(0, 1);
	}

	/**
	\return The imaginary unit.
	\related Kartet::Complex
	**/
	__host__ __device__ inline Complex<float> If(void)
	{
		return Complex<float>(0, 1);
	}

	/**
	\tparam T Base type.
	\return The imaginary unit.
	\related Kartet::Complex
	**/
	template<typename T>
	__host__ __device__ Complex<T> J(void)
	{
		return Complex<T>(0, 1);
	}

	/**
	\return The imaginary unit.
	\related Kartet::Complex
	**/
	__host__ __device__ inline Complex<double> J(void)
	{
		return Complex<double>(0, 1);
	}

	/**
	\return The imaginary unit.
	\related Kartet::Complex
	**/
	__host__ __device__ inline Complex<float> Jf(void)
	{
		return Complex<float>(0, 1);
	}

	// Operators :
		/**
		\fn Complex<T> operator+(const T1& a, const T2& b)
		\related Kartet::Complex
		\param a Left side.
		\param b Right side.
		\return The sum of left and right side.

		\fn Complex<T> operator-(const T1& a, const T2& b)
		\related Kartet::Complex
		\param a Left side.
		\param b Right side.
		\return The difference of left and right side.

		\fn Complex<T> operator*(const T1& a, const T2& b)
		\related Kartet::Complex
		\param a Left side.
		\param b Right side.
		\return The product of left and right side.

		\fn Complex<T> operator/(const T1& a, const T2& b)
		\related Kartet::Complex
		\param a Left side.
		\param b Right side.
		\return The quotient of left by right side.

		\fn bool operator==(const T1& a, const T2& b)
		\related Kartet::Complex
		\param a Left side.
		\param b Right side.
		\return True if the two sides are equal.

		\fn bool operator!=(const T1& a, const T2& b)
		\related Kartet::Complex
		\param a Left side.
		\param b Right side.
		\return True if the two sides are different.

		\fn template<typename T> T real(const Complex<T>& z)
		\related Kartet::Complex
		\param z Complex.
		\return The real part of the complex.
		
		\fn template<typename T> T imag(const Complex<T>& z)
		\related Kartet::Complex
		\param z Complex.
		\return The imaginary part of the complex.

		\fn template<typename T> T abs(const Complex<T>& z)
		\related Kartet::Complex
		\param z Complex.
		\return The absolute value of the complex (L2 norm).
		
		\fn template<typename T> T absSq(const Complex<T>& z)
		\related Kartet::Complex
		\param z Complex.
		\return The absolute value squared of the complex (L2 norm squared).

		\fn template<typename T> T arg(const Complex<T>& z)
		\related Kartet::Complex
		\param z Complex.
		\return The argument (angle) of the complex, within \f$ [-\pi; +\pi] \f$.

		\fn template<typename T> Complex<T> conj(const Complex<T>& z)
		\related Kartet::Complex
		\param z Complex.
		\return The conjugate of the complex.
		**/

	/// \cond FALSE
	#define COMPLEX_OPERATOR( operator, CxCxOperation, CxReOperation, ReCxOperation ) \
		template<typename T> \
		__host__ __device__ Complex<T> operator (const Complex<T>& z1, const Complex<T>& z2) \
		{ \
			return Complex<T> CxCxOperation ;\
		} \
		 \
		template<typename T1, typename T2> \
		__host__ __device__ Complex<typename ResultingType<T1,T2>::Type> operator (const Complex<T1>& z1, const Complex<T2>& z2) \
		{ \
			return Complex<typename ResultingType<T1,T2>::Type> CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<T> operator (const Complex<T>& z1, const cuFloatComplex& z2) \
		{ \
			return Complex<T> CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<T> operator (const Complex<T>& z1, const cuDoubleComplex& z2) \
		{ \
			return Complex<T> CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<T> operator (const cuFloatComplex& z1, const Complex<T>& z2) \
		{ \
			return Complex<T> CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<T> operator (const cuDoubleComplex& z1, const Complex<T>& z2) \
		{ \
			return Complex<T> CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,int>::Type> operator (const Complex<T>& z, const int& r) \
		{ \
			return Complex<typename ResultingType<T,int>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,unsigned int>::Type> operator (const Complex<T>& z, const unsigned int& r) \
		{ \
			return Complex<typename ResultingType<T,unsigned int>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,long long>::Type> operator (const Complex<T>& z, const signed long long& r) \
		{ \
			return Complex<typename ResultingType<T,long long>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,unsigned long long>::Type> operator (const Complex<T>& z, const unsigned long long& r) \
		{ \
			return Complex<typename ResultingType<T,unsigned long long>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,float>::Type> operator (const Complex<T>& z, const float& r) \
		{ \
			return Complex<typename ResultingType<T,float>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,double>::Type> operator (const Complex<T>& z, const double& r) \
		{ \
			return Complex<typename ResultingType<T,double>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ Complex<typename ResultingType<T,double>::Type> operator (const Complex<T>& z, const long double& r) \
		{ \
			return Complex<typename ResultingType<T,long double>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,int>::Type> operator (const int& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,int>::Type> ReCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,unsigned int>::Type> operator (const unsigned int& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,unsigned int>::Type> ReCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,long long>::Type> operator (const signed long long& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,long long>::Type> ReCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,unsigned long long>::Type> operator (const unsigned long long& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,unsigned long long>::Type> ReCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,float>::Type> operator (const float& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,float>::Type> ReCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,double>::Type> operator (const double& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,double>::Type> ReCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ Complex<typename ResultingType<T,double>::Type> operator (const long double& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,long double>::Type> ReCxOperation ;\
		}

		COMPLEX_OPERATOR(operator+, (z1.x+z2.x, z1.y+z2.y), 										(z.x + r, z.y),	(z.x + r, z.y)  )
		COMPLEX_OPERATOR(operator-, (z1.x-z2.x, z1.y-z2.y), 										(z.x - r, z.y),	(r - z.x, -z.y) )
		COMPLEX_OPERATOR(operator*, (z1.x*z2.x - z1.y*z2.y, z1.x*z2.y + z1.y*z2.x), 							(z.x*r, z.y*r), (z.x*r, z.y*r)  )
		COMPLEX_OPERATOR(operator/, ((z1.x*z2.x+z1.y*z2.y)/(z2.x*z2.x + z2.y*z2.y), (z1.y*z2.x-z1.x*z2.y)/(z2.x*z2.x + z2.y*z2.y)), 	(z.x/r, z.y/r), (r*z.x/(z.x*z.x + z.y*z.y), -r*z.y/(z.x*z.x + z.y*z.y)))
	#undef 	COMPLEX_OPERATOR

	#define COMPARISON_OPERATOR( operator, CxCxOperation, CxReOperation ) \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z1, const Complex<T>& z2) \
		{ \
			return CxCxOperation ;\
		} \
		 \
		template<typename T1, typename T2> \
		__host__ __device__ bool operator (const Complex<T1>& z1, const Complex<T2>& z2) \
		{ \
			return CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z1, const cuFloatComplex& z2) \
		{ \
			return CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z1, const cuDoubleComplex& z2) \
		{ \
			return CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const cuFloatComplex& z1, const Complex<T>& z2) \
		{ \
			return CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const cuDoubleComplex& z1, const Complex<T>& z2) \
		{ \
			return CxCxOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z, const int& r) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z, const unsigned int& r) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z, const signed long long& r) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z, const unsigned long long& r) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z, const float& r) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const Complex<T>& z, const double& r) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ bool operator (const Complex<T>& z, const long double& r) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const int& r, const Complex<T>& z) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const unsigned int& r, const Complex<T>& z) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const signed long long& r, const Complex<T>& z) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const unsigned long long& r, const Complex<T>& z) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const float& r, const Complex<T>& z) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ bool operator (const double& r, const Complex<T>& z) \
		{ \
			return CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ bool operator (const long double& r, const Complex<T>& z) \
		{ \
			return CxReOperation ;\
		}

		COMPARISON_OPERATOR( operator==, (z1.x==z2.x && z1.y==z2.y), (z.x==r && z.y==0) )
		COMPARISON_OPERATOR( operator!=, (z1.x!=z2.x || z1.y!=z2.y), (z.x!=r || z.y!=0) )
	#undef COMPARISON_OPERATOR

	#define SPECIAL_ReFUNCTION( function, CxOperation, ReOperationSigned, ReOperationUnsigned ) \
		__host__ __device__ inline int function (const int& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperationSigned ; \
		} \
		 \
		__host__ __device__ inline unsigned int function (const unsigned int& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperationUnsigned ; \
		} \
		 \
		__host__ __device__ inline signed long long function (const signed long long& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperationSigned ; \
		} \
		 \
		__host__ __device__ inline unsigned long long function (const unsigned long long& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperationUnsigned ; \
		} \
		 \
		__host__ __device__ inline float function (const float& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperationSigned ; \
		} \
		 \
		__host__ __device__ inline double function (const double& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperationSigned ; \
		} \
		 \
		__host__ inline long double function (const long double& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperationSigned ; \
		} \
		 \
		template<typename T> \
		__host__ __device__ T function (const Complex<T>& z) \
		{ \
			UNUSED_PARAMETER(z) \
			return CxOperation ; \
		} \
		 \
		__host__ __device__ inline float function (const cuFloatComplex& z) \
		{ \
			UNUSED_PARAMETER(z) \
			return CxOperation ; \
		} \
		 \
		__host__ __device__ inline double function (const cuDoubleComplex& z) \
		{ \
			UNUSED_PARAMETER(z) \
			return CxOperation ; \
		}
		
		
		SPECIAL_ReFUNCTION( real, 	z.x, 				r,			r)
		SPECIAL_ReFUNCTION( imag, 	z.y, 				0,			0)
		SPECIAL_ReFUNCTION( abs, 	::sqrt(z.x*z.x+z.y*z.y), 	r<0 ? -r : r,		r)
		SPECIAL_ReFUNCTION( absSq, 	z.x*z.x+z.y*z.y, 		r*r,			r*r)
		SPECIAL_ReFUNCTION( arg, 	::atan2(z.y, z.x), 		r<0 ? K_PI : 0,		0)
	#undef SPECIAL_ReFUNCTION

		// More specials :
		template<typename T>
		__host__ __device__ T fabs(const Complex<T>& z)
		{
			return ::sqrt(z.x*z.x+z.y*z.y);
		}

		__host__ __device__ inline float fabs(const cuFloatComplex& z)
		{
			return ::sqrt(z.x*z.x+z.y*z.y);
		}

		__host__ __device__ inline float fabsf(const cuFloatComplex& z)
		{
			return ::sqrtf(z.x*z.x+z.y*z.y);
		}

		__host__ __device__ inline double fabs(const cuDoubleComplex& z)
		{
			return ::sqrt(z.x*z.x+z.y*z.y);
		}

	#define SPECIAL_CxFUNCTION( function, CxOperation, ReOperation ) \
		__host__ __device__ inline int function (const int& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperation ; \
		} \
		 \
		__host__ __device__ inline unsigned int function (const unsigned int& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperation ; \
		} \
		 \
		__host__ __device__ inline signed long long function (const signed long long& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperation ; \
		} \
		 \
		__host__ __device__ inline unsigned long long function (const unsigned long long& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperation ; \
		} \
		 \
		__host__ __device__ inline float function (const float& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperation ; \
		} \
		 \
		__host__ __device__ inline double function (const double& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperation ; \
		} \
		 \
		__host__ inline long double function (const long double& r) \
		{ \
			UNUSED_PARAMETER(r) \
			return ReOperation ; \
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<T> function (const Complex<T>& z) \
		{ \
			UNUSED_PARAMETER(z) \
			return Complex<T> CxOperation ; \
		} \
		 \
		__host__ __device__ inline Complex<float> function (const cuFloatComplex& z) \
		{ \
			UNUSED_PARAMETER(z) \
			return Complex<float> CxOperation ; \
		} \
		 \
		__host__ __device__ inline Complex<double> function (const cuDoubleComplex& z) \
		{ \
			UNUSED_PARAMETER(z) \
			return Complex<double> CxOperation ; \
		}
		
		SPECIAL_CxFUNCTION( conj, (z.x, -z.y), r)
	#undef SPECIAL_CxFUNCTION
	/// \endcond

	/**
	\return Complex number from polar representation.
	\related Kartet::Complex
	\param r Radius.
	\param theta Angle.
	**/
	template<typename T>
	__host__ __device__ Complex<T> polar(const T& r, const T& theta)
	{
		const bool t =	IsSame<T,bool>::value ||
				IsSame<T,char>::value || IsSame<T,unsigned char>::value ||
				IsSame<T,short>::value || IsSame<T,unsigned short>::value ||
				IsSame<T,int>::value || IsSame<T,unsigned int>::value ||
				IsSame<T,long long>::value || IsSame<T,unsigned long long>::value ||
				IsSame<T,float>::value || IsSame<T,double>::value;
		STATIC_ASSERT_VERBOSE(t, TYPE_MUST_BE_REAL)
		return Complex<T>(r*::cos(theta), r*::sin(theta));
	}

	/**
	\return Complex number from polar representation.
	\related Kartet::Complex
	\param theta Angle.
	**/
	template<typename T>
	__host__ __device__ Complex<T> polar(const T& theta)
	{
		const bool t =	IsSame<T,bool>::value ||
				IsSame<T,char>::value || IsSame<T,unsigned char>::value ||
				IsSame<T,short>::value || IsSame<T,unsigned short>::value ||
				IsSame<T,int>::value || IsSame<T,unsigned int>::value ||
				IsSame<T,long long>::value || IsSame<T,unsigned long long>::value ||
				IsSame<T,float>::value || IsSame<T,double>::value;
		STATIC_ASSERT_VERBOSE(t, TYPE_MUST_BE_REAL)
		return Complex<T>(::cos(theta), ::sin(theta));
	}

	/**
	\brief Print complex number in a stream.
	\related Kartet::Complex
	\param os Stream.
	\param z Complex.
	\return Reference to modified stream.
	**/
	template<typename T>
	__host__ std::ostream& operator<<(std::ostream& os, const Complex<T>& z)
	{
		const T zero = static_cast<T>(0),
			minusZero = -static_cast<T>(0);
		const bool px = (z.x>=zero) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(z.x, minusZero));
		const bool flag = !(os.flags() & std::ios_base::showpos);
		// Vector notation :
		#ifdef KARTET_VECTOR_COMPLEX_NOTATION
			const bool py = (z.y>=zero) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(z.y, minusZero));
			os << '(';
			if(flag && px)
				os << ' ';
			os << z.x << ", ";
			if(flag && py)
				os << ' ';
			os << z.y << ')';
		#else //#ifdef KARTET_LITERAL_COMPLEX_NOTATION
			if(flag && px)
				os << ' ';
			os << z.x << std::showpos << z.y << 'i';
			if(flag)
				os << std::noshowpos;
		#endif
		return os;
	}

} // namespace Kartet

// The following functions must be kept in the global namespace (::) :
		/**
		\fn Complex<TFloat> cos(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The cosine of the argument.

		\fn Complex<TFloat> cosh(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The hyperbolic cosine of the argument.

		\fn Complex<TFloat> exp(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The exponential of the argument.

		\fn Complex<TFloat> log(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The logarithm of the argument.
		
		\fn Complex<TFloat> log10(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The logarithm in base 10 of the argument.

		\fn Complex<TFloat> log2(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The logarithm in base 2 of the argument.

		\fn Complex<TFloat> sin(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The sine of the argument.

		\fn Complex<TFloat> sinh(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The hyperbolic sine of the argument.

		\fn Complex<TFloat> sqrt(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The square root of the argument (on the positive branch).

		\fn Complex<TFloat> tan(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The tangent of the argument.

		\fn Complex<TFloat> tanh(const T& x)
		\related Kartet::Complex
		\param x Argument.
		\return The hyperbolic tangent of the argument.
		**/
	/// \cond FALSE 
	#define TRANSCENDENTAL_CxFUNCTION(function, realPart, imagPart, ...) \
		template<typename T> \
		__host__ __device__ Kartet::Complex<double> function (const Kartet::Complex<T>& x) \
		{ \
			typedef double WorkType; \
			const double __VA_ARGS__; \
			return Kartet::Complex<double>(realPart, imagPart); \
		} \
		 \
		__host__ __device__ inline Kartet::Complex<float> function (const Kartet::Complex<float>& x) \
		{ \
			typedef float WorkType; \
			const float __VA_ARGS__; \
			return Kartet::Complex<float>(realPart, imagPart); \
		} \
		 \
		/*__host__ inline Kartet::Complex<long double> function (const Kartet::Complex<long double>& x) \
		{ \
			typedef long double WorkType; \
			const long double __VA_ARGS__; \
			return Kartet::Complex<long double>(realPart, imagPart); \
		}*/
		// NVCC will generate warnings on this last part.

		TRANSCENDENTAL_CxFUNCTION( cos, 
						ca/static_cast<WorkType>(2)*(static_cast<WorkType>(1)/e + e), 
						sa/static_cast<WorkType>(2)*(static_cast<WorkType>(1)/e - e), 
							ca = ::cos(x.real()), 
							sa = ::sin(x.real()),
							e = ::exp(x.imag())
					)

		TRANSCENDENTAL_CxFUNCTION( cosh,
						ach*bc,
						ash*bs,
							ach = ::cosh(x.real()),
							ash = ::sinh(x.real()),
							bc = ::cos(x.imag()),
							bs = ::sin(x.imag())
					)

		TRANSCENDENTAL_CxFUNCTION( exp,
						ea*cb,
						ea*sb,
							ea = ::exp(x.real()),
							cb = ::cos(x.imag()),
							sb = ::sin(x.imag())
					)

		TRANSCENDENTAL_CxFUNCTION( log,
						::log(a*a+b*b)/static_cast<WorkType>(2),
						::atan2(b, a),
							a = x.real(),
							b = x.imag()
					)

		TRANSCENDENTAL_CxFUNCTION( log10,
						::log10(a*a+b*b)/static_cast<WorkType>(2),
						::atan2(b, a)/static_cast<WorkType>(K_L10),
							a = x.real(),
							b = x.imag()
					)

		TRANSCENDENTAL_CxFUNCTION( log2,
						::log2(a*a+b*b)/static_cast<WorkType>(2),
						::atan2(b, a)/static_cast<WorkType>(K_L2),
							a = x.real(),
							b = x.imag()
					)

		TRANSCENDENTAL_CxFUNCTION( sin,
						sa/static_cast<WorkType>(2)*(static_cast<WorkType>(1)/e + e),
						ca/static_cast<WorkType>(2)*(static_cast<WorkType>(1)/e - e),
							ca = ::cos(x.real()),
							sa = ::sin(x.real()),
							e = ::exp(x.imag())
					)

		TRANSCENDENTAL_CxFUNCTION( sinh,
						ash*bc,
						ach*bs,
							ach = ::cosh(x.real()),
							ash = ::sinh(x.real()),
							bc = ::cos(x.imag()),
							bs = ::sin(x.imag())
					)

		TRANSCENDENTAL_CxFUNCTION( sqrt,
						n*::cos(p),
						n*::sin(p),
							a = x.real(),
							b = x.imag(),
							n = ::pow(a*a+b*b, static_cast<WorkType>(0.25)),
							p = ::atan2(b, a)/static_cast<WorkType>(2)
					)

		TRANSCENDENTAL_CxFUNCTION( tan,
						s2a/(c2a + (e2b+static_cast<WorkType>(1)/e2b)/static_cast<WorkType>(2)),
						(e2b-static_cast<WorkType>(1)/e2b)/(static_cast<WorkType>(2)*c2a + e2b+static_cast<WorkType>(1)/e2b),
							c2a = ::cos(x.real()*static_cast<WorkType>(2)),
							s2a = ::sin(x.real()*static_cast<WorkType>(2)),
							e2b = ::exp(x.imag()*static_cast<WorkType>(2))
					)

		TRANSCENDENTAL_CxFUNCTION( tanh,
						(bst+static_cast<WorkType>(1))*at/d,
						(static_cast<WorkType>(1)-ast)*bt/d,
							at = ::tanh(x.real()),
							bt = ::tan(x.imag()),
							ast = at*at,
							bst = bt*bt,
							d = (bst*ast + static_cast<WorkType>(1))
					)

		/* To be implemented :
			pow	Power of complex (binary).
			acos	Arc cosine of complex.
			acosh	Arc hyperbolic cosine of complex.
			asin	Arc sine of complex.
			asinh	Arc hyperbolic sine of complex.
			atan	Arc tangent of complex.
			atanh	Arc hyperbolic tangent of complex.
		*/
	#undef TRANSCENDENTAL_CxFUNCTION
	/// \endcond

// And they need to be imported to Kartet Namespace for some expressions to work :
namespace Kartet
{
	using ::cos;
	using ::cosh;
	using ::exp;
	using ::log;
	using ::log10;
	using ::log2;
	using ::sin;
	using ::sinh;
	using ::sqrt;
	using ::tan;
	using ::tanh;
}

#endif

