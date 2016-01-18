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
	
	Usual operators are also defined and implemented. Kartet::Complex<float> and Kartet::Complex<double> are binary compatible respectively with cuFloatComplex and cuDoubleComplex.

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

		/**
		\return The imaginary unit.
		**/
		static Complex<T> i(void)
		{
			return Complex<T>(0, 1);
		}

		/**
		\return The imaginary unit.
		**/
		static Complex<T> j(void)
		{
			return Complex<T>(0, 1);
		}
	};

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
	#define COMPLEX_OPERATOR( operator, CxCxOperation, CxReOperation ) \
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
			return Complex<typename ResultingType<T,int>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,unsigned int>::Type> operator (const unsigned int& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,unsigned int>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,long long>::Type> operator (const signed long long& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,long long>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,unsigned long long>::Type> operator (const unsigned long long& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,unsigned long long>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,float>::Type> operator (const float& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,float>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ __device__ Complex<typename ResultingType<T,double>::Type> operator (const double& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,double>::Type> CxReOperation ;\
		} \
		 \
		template<typename T> \
		__host__ Complex<typename ResultingType<T,double>::Type> operator (const long double& r, const Complex<T>& z) \
		{ \
			return Complex<typename ResultingType<T,long double>::Type> CxReOperation ;\
		}

		

		COMPLEX_OPERATOR(operator+, (z1.x+z2.x, z1.y+z2.y), 										(z.x + r, z.y) )
		COMPLEX_OPERATOR(operator-, (z1.x-z2.x, z1.y-z2.y), 										(z.x - r, z.y) )
		COMPLEX_OPERATOR(operator*, (z1.x*z2.x - z1.y*z2.y, z1.x*z2.y + z1.y*z2.x), 							(z.x*r, z.y*r) )
		COMPLEX_OPERATOR(operator/, ((z1.x*z2.x+z1.y*z2.y)/(z2.x*z2.x + z2.y*z2.y), (z1.y*z2.x-z1.x*z2.y)/(z2.x*z2.x + z2.y*z2.y)), 	(z.x/r, z.y/r) )
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
		SPECIAL_ReFUNCTION( arg, 	::atan2(z.y, z.y), 		r<0 ? -K_PI : K_PI,	K_PI)
	#undef SPECIAL_ReFUNCTION

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
		return Complex<T>(r*::cos(theta), r*::sin(theta));
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
		os << '(' << z.x << ", " << z.y << ')';
		return os;
	}

} // namespace Kartet

#endif

