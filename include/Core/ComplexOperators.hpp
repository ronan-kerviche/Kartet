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

#ifndef __KARTET_COMPLEX_OPERATORS__
#define __KARTET_COMPLEX_OPERATORS__

	#include <cmath>
	#include <iostream>
	#include <iomanip>
	#include "Core/LibTools.hpp"

// Missing functions :
#ifndef __CUDACC__
	inline void sincos(float a, float* s, float* c)
	{
		#ifdef _WIN32
			if(s!=NULL) (*s) = sinf(a);
			if(c!=NULL) (*c) = cosf(a);
		#else
			sincosf(a,s,c);
		#endif
	}

	#ifdef _WIN32
	inline void sincos(double a, double* s, double* c)
	{
			if(s!=NULL) (*s) = sin(a);
			if(c!=NULL) (*c) = cos(a);		
	}
	#endif

	inline void sincospi(float a, float* s, float* c)
	{
		sincos(a*K_PI,s,c);
	}

	inline void sincospi(double a, double* s, double* c)
	{
		sincos(a*K_PI,s,c);
	}
#endif

namespace Kartet
{
	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::BaseType real(const T& a);
	
	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::BaseType imag(const T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::BaseType& realRef(T& a);
	
	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::BaseType& imagRef(T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::ComplexType toComplex(const T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::ComplexType toComplex(const T& a, const T& b);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<float>::ComplexType toFloatComplex(const T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<float>::ComplexType toFloatComplex(const T& a, const T& b);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<double>::ComplexType toDoubleComplex(const T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<double>::ComplexType toDoubleComplex(const T& a, const T& b);
	
	template<typename TOut, typename TIn>
	__host__ __device__ inline void complexCopy(TOut& a, const TIn& b)
	{
		// If the input type is complex, the output type must be complex :
		StaticAssert<!(!TypeInfo<TOut>::isComplex && TypeInfo<TIn>::isComplex)>();

		realRef(a) = real(b);
		if(TypeInfo<TOut>::isComplex)
		{
			if(TypeInfo<TIn>::isComplex)
				imagRef(a) = imag(b);
			else
				imagRef(a) = static_cast<typename TypeInfo<TOut>::BaseType>(0);
		}
	}

	template<typename TOut, typename TIn>
	__host__ __device__ inline TOut complexCopy(const TIn& b)
	{
		// If the input type is complex, the output type must be complex :
		StaticAssert<!(!TypeInfo<TOut>::isComplex && TypeInfo<TIn>::isComplex)>();
		
		TOut res;
		realRef(res) = real(b);
		if(TypeInfo<TOut>::isComplex)
		{
			if(TypeInfo<TIn>::isComplex)
				imagRef(res) = imag(b);
			else
				imagRef(res) = static_cast<typename TypeInfo<TOut>::BaseType>(0);
		}
		return res;
	}

	template<typename TOut, typename TIn>
	__host__ __device__ inline void unprotectedComplexCopy(TOut& a, const TIn& b)
	{
		// Do not test the operation first.
		realRef(a) = real(b);
		if(TypeInfo<TOut>::isComplex)
		{
			if(TypeInfo<TIn>::isComplex)
				imagRef(a) = imag(b);
			else
				imagRef(a) = static_cast<typename TypeInfo<TOut>::BaseType>(0);
		}
	}

	template<typename TOut, typename TIn>
	__host__ __device__ inline TOut unprotectedComplexCopy(const TIn& b)
	{
		// Do not test the operation first.
		TOut res;
		realRef(res) = real(b);
		if(TypeInfo<TOut>::isComplex)
		{
			if(TypeInfo<TIn>::isComplex)
				imagRef(res) = imag(b);
			else
				imagRef(res) = static_cast<typename TypeInfo<TOut>::BaseType>(0);
		}
		return res;
	}

	#define BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE(TypeName) \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType real(const TypeName& a) \
		{ \
			return a; \
		} \
		 \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType imag(const TypeName& a) \
		{ \
			UNUSED_PARAMETER(a) \
			return static_cast<TypeName>(0); \
		} \
		 \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType& realRef(TypeName& a) \
		{ \
			return a; \
		} \
		 \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType& imagRef(TypeName& a) \
		{ \
			return a; \
		} \
		 \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::ComplexType toComplex(const TypeName& a) \
		{ \
			typedef TypeInfo<TypeInfo<TypeName>::ComplexType>::BaseType TCast; \
			TypeInfo<TypeName>::ComplexType tmp = {static_cast<TCast>(a), static_cast<TCast>(0)}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::ComplexType toComplex(const TypeName& a, const TypeName& b) \
		{ \
			typedef TypeInfo<TypeInfo<TypeName>::ComplexType>::BaseType TCast; \
			TypeInfo<TypeName>::ComplexType tmp = {static_cast<TCast>(a), static_cast<TCast>(b)}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<float>::ComplexType toFloatComplex(const TypeName& a) \
		{ \
			TypeInfo<float>::ComplexType tmp = {static_cast<float>(a), 0.0f}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<float>::ComplexType toFloatComplex(const TypeName& a, const TypeName& b) \
		{ \
			TypeInfo<float>::ComplexType tmp = {static_cast<float>(a), static_cast<float>(b)}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<double>::ComplexType toDoubleComplex(const TypeName& a) \
		{ \
			TypeInfo<double>::ComplexType tmp = {static_cast<double>(a), 0.0}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<double>::ComplexType toDoubleComplex(const TypeName& a, const TypeName& b) \
		{ \
			TypeInfo<double>::ComplexType tmp = {static_cast<double>(a), static_cast<double>(b)}; \
			return tmp; \
		}

	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( bool )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( char )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( signed char )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( unsigned char )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( short )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( unsigned short )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( int )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( unsigned int )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( long )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( unsigned long )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( long long )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( unsigned long long )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( float )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( double )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE( long double )

	#undef BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE

	// Special cases :
	__host__ __device__ inline __cuda_typename TypeInfo<double>::ComplexType toComplex(const float& a, const double& b)
	{
		TypeInfo<double>::ComplexType tmp = {static_cast<double>(a), b};
		return tmp;
	}

	__host__ __device__ inline __cuda_typename TypeInfo<double>::ComplexType toComplex(const double& a, const float& b)
	{
		TypeInfo<double>::ComplexType tmp = {a, static_cast<double>(b)};
		return tmp;
	}

	#define BUILD_SPECIALIZATION_FROM_COMPLEX_TYPE(TypeName) \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType real(const TypeName& a) \
		{ \
			return a.x; \
		} \
		 \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType imag(const TypeName& a) \
		{ \
			return a.y; \
		} \
		 \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType& realRef(TypeName& a) \
		{ \
			return a.x; \
		} \
		 \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType& imagRef(TypeName& a) \
		{ \
			return a.y; \
		} \
		 \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::ComplexType toComplex(const TypeName& a) \
		{ \
			return a; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<float>::ComplexType toFloatComplex(const TypeName& a) \
		{ \
			TypeInfo<float>::ComplexType tmp = {static_cast<float>(a.x), static_cast<float>(a.y)}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<double>::ComplexType toDoubleComplex(const TypeName& a) \
		{ \
			TypeInfo<double>::ComplexType tmp = {static_cast<double>(a.x), static_cast<double>(a.y)}; \
			return tmp; \
		}

		BUILD_SPECIALIZATION_FROM_COMPLEX_TYPE(cuFloatComplex)
		BUILD_SPECIALIZATION_FROM_COMPLEX_TYPE(cuDoubleComplex)

	#undef BUILD_SPECIALIZATION_FROM_COMPLEX_TYPE

// Other functions :
	template<typename T>
	__host__ __device__ inline T conj(const T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::BaseType absSq(const T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::BaseType abs(const T& a);
	
	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::BaseType angle(const T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::ComplexType angleToComplex(const T& a);

	template<typename T>
	__host__ __device__ inline typename TypeInfo<T>::ComplexType piAngleToComplex(const T& a);

	#define BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_COMMON(TypeName) \
		template<> \
		__host__ __device__ inline TypeName conj<TypeName>(const TypeName& a) \
		{ \
			return a; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType absSq<TypeName>(const TypeName& a) \
		{ \
			return a*a; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::ComplexType angleToComplex<TypeName>(const TypeName& a) \
		{ \
			TypeInfo<__cuda_typename TypeInfo<TypeName>::ComplexType>::BaseType c, s; \
			sincos(a, &s, &c); \
			TypeInfo<TypeName>::ComplexType tmp = {c, s}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::ComplexType piAngleToComplex<TypeName>(const TypeName& a) \
		{ \
			TypeInfo<__cuda_typename TypeInfo<TypeName>::ComplexType>::BaseType c, s; \
			sincospi(a, &s, &c); \
			TypeInfo<TypeName>::ComplexType tmp = {c, s}; \
			return tmp; \
		}

	#define BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_SIGNED(TypeName) \
		BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_COMMON(TypeName) \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType abs<TypeName>(const TypeName& a) \
		{ \
			return (a<static_cast<TypeName>(0)) ? (-a) : a; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType angle<TypeName>(const TypeName& a) \
		{ \
			UNUSED_PARAMETER(a) \
			return ((a>=0) ? static_cast<TypeInfo<TypeName>::BaseType>(0) : static_cast<__cuda_typename TypeInfo<TypeName>::BaseType>(K_PI)); \
		} \

	#define BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_UNSIGNED(TypeName) \
		BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_COMMON(TypeName) \
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType abs<TypeName>(const TypeName& a) \
		{ \
			return a; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType angle<TypeName>(const TypeName& a) \
		{ \
			UNUSED_PARAMETER(a) \
			return static_cast<TypeInfo<TypeName>::BaseType>(0); \
		} \

	#define BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_REAL(TypeName) \
		template<> \
		__host__ __device__ inline TypeName conj<TypeName>(const TypeName& a) \
		{ \
			return a; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType absSq<TypeName>(const TypeName& a) \
		{ \
			return a*a; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType abs<TypeName>(const TypeName& a) \
		{ \
			return (TypeInfo<TypeName>::isSigned && a<0) ? (-a) : a; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType angle<TypeName>(const TypeName& a) \
		{ \
			UNUSED_PARAMETER(a) \
			return ((a>=0) ? 0 : static_cast<__cuda_typename TypeInfo<TypeName>::BaseType>(K_PI)); \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::ComplexType angleToComplex<TypeName>(const TypeName& a) \
		{ \
			TypeInfo<__cuda_typename TypeInfo<TypeName>::ComplexType>::BaseType c, s; \
			sincos(a, &s, &c); \
			TypeInfo<TypeName>::ComplexType tmp = {c, s}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::ComplexType piAngleToComplex<TypeName>(const TypeName& a) \
		{ \
			TypeInfo<__cuda_typename TypeInfo<TypeName>::ComplexType>::BaseType c, s; \
			sincospi(a, &s, &c); \
			TypeInfo<TypeName>::ComplexType tmp = {c, s}; \
			return tmp; \
		}

	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_UNSIGNED( bool )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_SIGNED( char )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_SIGNED( signed char )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_UNSIGNED( unsigned char )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_SIGNED( short )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_UNSIGNED( unsigned short )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_SIGNED( int )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_UNSIGNED( unsigned int )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_SIGNED( long )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_UNSIGNED( unsigned long )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_SIGNED( long long )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_INTEGER_UNSIGNED( unsigned long long )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_REAL( float )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_REAL( double )
	BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE_REAL( long double )

	#undef BUILD_SPECIALIZATION_FROM_REAL_BASE_TYPE

	#define BUILD_SPECIALIZATION_FROM_COMPLEX_TYPE(TypeName) \
		template<> \
		__host__ __device__ inline TypeName conj<TypeName>(const TypeName& a) \
		{ \
			TypeName tmp = {a.x, -a.y}; \
			return tmp; \
		} \
		\
		template<> \
		__host__ __device__ inline __cuda_typename TypeInfo<TypeName>::BaseType absSq<TypeName>(const TypeName& a) \
		{ \
			return a.x*a.x + a.y*a.y; \
		} \
		\
		inline std::ostream& operator<<(std::ostream& os, const TypeName& a) \
		{ \
			const int width = os.width(); \
			const char fillCharacter = os.fill(); \
			os << std::right << std::setfill(fillCharacter) << std::setw(width) << a.x << " + " << std::right << std::setfill(fillCharacter) << std::setw(width) << a.y << " i"; \
			return os; \
		}

	BUILD_SPECIALIZATION_FROM_COMPLEX_TYPE(cuFloatComplex)
	BUILD_SPECIALIZATION_FROM_COMPLEX_TYPE(cuDoubleComplex)

	#undef BUILD_SPECIALIZATION_FROM_COMPLEX_TYPE

	template<>
	__host__ __device__ inline __cuda_typename TypeInfo<cuFloatComplex>::BaseType angle<cuFloatComplex>(const cuFloatComplex& a)
	{
		return atan2f(a.y, a.x);
	}

	template<>
	__host__ __device__ inline __cuda_typename TypeInfo<cuDoubleComplex>::BaseType angle<cuDoubleComplex>(const cuDoubleComplex& a)
	{
		return atan2(a.y, a.x);
	}

	template<>
	__host__ __device__ inline __cuda_typename TypeInfo<cuFloatComplex>::BaseType abs<cuFloatComplex>(const cuFloatComplex& a)
	{
		return sqrtf(a.x*a.x + a.y*a.y);
	}

	template<>
	__host__ __device__ inline __cuda_typename TypeInfo<cuDoubleComplex>::BaseType abs<cuDoubleComplex>(const cuDoubleComplex& a)
	{
		return sqrt(a.x*a.x + a.y*a.y);
	}

// Template for the operators : 
	#define MIXED_SCALAR_COMPLEX_OPERATOR( type , retCXFloat , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		__host__ __device__ inline retCXFloat operator symbol (const type & a, const cuFloatComplex& c) \
		{ \
			NLeft \
			return resScalLeft ; \
		} \
		 \
		__host__ __device__ inline cuDoubleComplex operator symbol (const type & a, const cuDoubleComplex& c) \
		{ \
			NLeft \
			return resScalLeft ; \
		} \
		 \
		__host__ __device__ inline retCXFloat operator symbol (const cuFloatComplex& c, const type & a) \
		{ \
			NRight \
			return resScalRight; \
		} \
		 \
		__host__ __device__ inline cuDoubleComplex operator symbol (const cuDoubleComplex& c, const type & a) \
		{ \
			NRight \
			return resScalRight; \
		} \

	#define MIXED_SCALAR_COMPLEX_OPERATOR_DEFINITION( symbol, resScalLeft, resScalRight, complexComplex, NLeft, NRight, NCC) \
		MIXED_SCALAR_COMPLEX_OPERATOR( char , 		cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( unsigned char , 	cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( short , 		cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( unsigned short , cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( int , 		cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( unsigned int , 	cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( long, 		cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( unsigned long , 	cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( long long , 	cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( float , 		cuFloatComplex  , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		MIXED_SCALAR_COMPLEX_OPERATOR( double , 	cuDoubleComplex , symbol, resScalLeft, resScalRight, NLeft, NRight) \
		 \
		__host__ __device__ inline cuDoubleComplex operator symbol (const cuDoubleComplex& c1, const cuFloatComplex& c2) \
		{ \
			NCC \
			return complexComplex; \
		} \
		 \
		__host__ __device__ inline cuDoubleComplex operator symbol (const cuFloatComplex& c1, const cuDoubleComplex& c2) \
		{ \
			NCC \
			return complexComplex; \
		} \
		 \
		__host__ __device__ inline cuFloatComplex operator symbol (const cuFloatComplex& c1, const cuFloatComplex& c2) \
		{ \
			NCC \
			return complexComplex; \
		} \
		 \
		__host__ __device__ inline cuDoubleComplex operator symbol (const cuDoubleComplex& c1, const cuDoubleComplex& c2)  \
		{ \
			NCC ; \
			return complexComplex; \
		}

	// Plus :
		MIXED_SCALAR_COMPLEX_OPERATOR_DEFINITION( +, toComplex(a + c.x, c.y), toComplex(c.x + a, c.y), toComplex(c1.x+c2.x, c1.y+c2.y), ; , ; , ;)
	// Minus :
		MIXED_SCALAR_COMPLEX_OPERATOR_DEFINITION( -, toComplex(a - c.x, c.y), toComplex(c.x - a, c.y), toComplex(c1.x-c2.x, c1.y-c2.y), ; , ; , ;)
	// Times :
		MIXED_SCALAR_COMPLEX_OPERATOR_DEFINITION( *, toComplex(a * c.x, a * c.y), toComplex(c.x * a, c.y * a), toComplex(c1.x*c2.x - c1.y*c2.y, c1.x*c2.y + c1.y*c2.x), ; , ; , ; )
	// Div :
		MIXED_SCALAR_COMPLEX_OPERATOR_DEFINITION( /, toComplex(a * c.x / n, - a * c.y / n), toComplex(c.x / a, c.y / a), toComplex((c1.x*c2.x+c1.y*c2.y)/n, (c1.x*c2.y-c1.y*c2.x)/n), float n = c.x*c.x + c.y*c.y; , ; , float n = c2.x*c2.x + c2.y*c2.y; )

	#undef MIXED_SCALAR_COMPLEX_OPERATOR_DEFINITION
	#undef MIXED_SCALAR_COMPLEX_OPERATOR

	template<typename T>
	void dynamicCopy(T* dst, const void* src, const int typeIndex, const size_t count=1)
	{
		#define SPECIAL_COPY( TypeName ) \
			case GetIndex<TypesSortedByAccuracy, TypeName>::value :	\
				for(size_t k=0; k<count; k++) \
					unprotectedComplexCopy<T>(dst[k], reinterpret_cast<const TypeName *>(src)[k]); \
				break;

		if(isComplex(typeIndex) && !TypeInfo<T>::isComplex)
			throw InvalidOperation;

		switch(typeIndex)
		{
			SPECIAL_COPY( bool )
			SPECIAL_COPY( unsigned char )
			SPECIAL_COPY( char )
			SPECIAL_COPY( signed char )
			SPECIAL_COPY( unsigned short )
			//SPECIAL_COPY( short )
			SPECIAL_COPY( signed short )
			SPECIAL_COPY( unsigned int )
			//SPECIAL_COPY( int )
			SPECIAL_COPY( signed int )
			SPECIAL_COPY( unsigned long )
			//SPECIAL_COPY( long )
			SPECIAL_COPY( signed long )
			SPECIAL_COPY( unsigned long long )
			//SPECIAL_COPY( long long )
			SPECIAL_COPY( signed long long )
			SPECIAL_COPY( float )
			SPECIAL_COPY( double )
			SPECIAL_COPY( long double )
			SPECIAL_COPY( cuFloatComplex )
			SPECIAL_COPY( cuDoubleComplex )
			default :
				throw UnknownTypeIndex;
		}
	}

} // namespace Kartet

#endif 

