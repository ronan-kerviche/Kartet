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
	\file    Traits.hpp
	\brief   Type traits.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_TRAITS__
#define __KARTET_TRAITS__

	#include <algorithm>
	#include "Core/Exceptions.hpp"
	#include "Core/Meta.hpp"
	#include "Core/Complex.hpp"
	//#include "Core/Mat.hpp"
	//#include "Core/Vec.hpp"

namespace Kartet
{
	// Traits : 
	template<typename T>
	struct Traits
	{
		typedef void SubTraits;
		typedef T BaseType;
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= false,
					isFloatingPoint	= false,
					isVector	= false,
					isMatrix	= false;
	};

	template<typename T>
	struct Traits<T[]>
	{
		typedef Traits<T> SubTraits;
		typedef typename SubTraits::BaseType BaseType;
		static const bool 	isConst 	= SubTraits::isConst,
					isArray		= true,
					isPointer 	= SubTraits::isPointer,
					isReference 	= SubTraits::isReference,
					isComplex 	= SubTraits::isComplex,
					isFloatingPoint	= SubTraits::isFloatingPoint,
					isVector	= SubTraits::isVector,
					isMatrix	= SubTraits::isMatrix;
	};

	template<typename T>
	struct Traits<T*>
	{
		typedef Traits<T> SubTraits;
		typedef typename SubTraits::BaseType BaseType;
		static const bool 	isConst 	= SubTraits::isConst,
					isArray		= SubTraits::isArray,
					isPointer 	= true,
					isReference 	= SubTraits::isReference,
					isComplex 	= SubTraits::isComplex,
					isFloatingPoint	= SubTraits::isFloatingPoint,
					isVector	= SubTraits::isVector,
					isMatrix	= SubTraits::isMatrix;
	};

	template<typename T>
	struct Traits<T&>
	{
		typedef Traits<T> SubTraits;
		typedef typename SubTraits::BaseType BaseType;
		static const bool 	isConst 	= SubTraits::isConst,
					isArray		= SubTraits::isArray,
					isPointer 	= SubTraits::isPointer,
					isReference 	= true,
					isComplex 	= SubTraits::isComplex,
					isFloatingPoint = SubTraits::isFloatingPoint,
					isVector	= SubTraits::isVector,
					isMatrix	= SubTraits::isMatrix;
	};

	template<typename T>
	struct Traits<const T>
	{
		typedef Traits<T> SubTraits;
		typedef typename SubTraits::BaseType BaseType;
		static const bool 	isConst 	= true,
					isArray		= SubTraits::isArray,	
					isPointer 	= SubTraits::isPointer,
					isReference 	= SubTraits::isReference,
					isComplex 	= SubTraits::isComplex,
					isFloatingPoint = SubTraits::isFloatingPoint,
					isVector	= SubTraits::isVector,
					isMatrix	= SubTraits::isMatrix;
	};

	template<>
	struct Traits<float>
	{
		typedef void SubTraits;
		typedef float BaseType;	
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= false,
					isFloatingPoint = true,
					isVector	= false,
					isMatrix	= false;
	};

	template<>
	struct Traits<double>
	{
		typedef void SubTraits;
		typedef double BaseType;	
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= false,
					isFloatingPoint = true,
					isVector	= false,
					isMatrix	= false;
	};

	template<>
	struct Traits<long double>
	{
		typedef void SubTraits;
		typedef long double BaseType;	
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= false,
					isFloatingPoint = true,
					isVector	= false,
					isMatrix	= false;
	};

	template<>
	struct Traits<cuFloatComplex>
	{
		typedef void SubTraits;
		typedef float BaseType;
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= true,
					isFloatingPoint = true,
					isVector	= false,
					isMatrix	= false;
	};

	template<>
	struct Traits<cuDoubleComplex>
	{
		typedef void SubTraits;
		typedef double BaseType;
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= true,
					isFloatingPoint = true,
					isVector	= false,
					isMatrix	= false;
	};

	template<typename T>
	struct Traits< Complex<T> >
	{
		typedef Traits<T> SubTraits;
		typedef T BaseType;
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= true,
					isFloatingPoint = SubTraits::isFloatingPoint,
					isVector	= false,
					isMatrix	= false;
	};
	
	// Resulting types :
	template<typename T1, typename T2>
	class ResultingType
	{
		private : 
			typedef Traits<T1> U1;
			typedef Traits<T2> U2;
			typedef typename U1::BaseType B1;
			typedef typename U2::BaseType B2;
			typedef typename StaticIf< IsSame<B1, long double>::value || IsSame<B2, long double>::value, 			long double,
				typename StaticIf< IsSame<B1, double>::value || IsSame<B2, double>::value, 				double,
				typename StaticIf< IsSame<B1, float>::value || IsSame<B2, float>::value, 				float,
				typename StaticIf< IsSame<B1, unsigned long long>::value || IsSame<B2, unsigned long long>::value, 	unsigned long long,
				typename StaticIf< IsSame<B1, long long>::value || IsSame<B2, long long>::value, 			long long,
				typename StaticIf< IsSame<B1, unsigned int>::value || IsSame<B2, unsigned int>::value, 			unsigned int,
					 int
				>::Type >::Type >::Type >::Type >::Type >::Type Br;
		public :
			typedef typename StaticIf< U1::isComplex || U2::isComplex, Complex<Br>, Br>::Type Type;
	};

	template<typename T1, typename T2>
	class ResultingType2 // Alias.
	{
		public :
			typedef typename ResultingType<T1,T2>::Type Type;
	};

	template<typename T1, typename T2, typename T3>
	class ResultingType3
	{
		public :
			typedef typename ResultingType<T1,typename ResultingType<T2,T3>::Type>::Type Type;
	};

	template<typename T1, typename T2, typename T3, typename T4>
	class ResultingType4
	{
		public :
			typedef typename ResultingType<T1,typename ResultingType<T2,typename ResultingType<T3,T4>::Type>::Type>::Type Type;
	};

	template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
	class ResultingType9
	{
		public :
			typedef typename ResultingType<T1,typename ResultingType<T2,typename ResultingType<T3,typename ResultingType<T4,typename ResultingType<T5,typename ResultingType<T6,typename ResultingType<T7,typename ResultingType<T8,T9>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type Type;
	};

	template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16>
	class ResultingType16
	{
		public :
			typedef typename ResultingType<T1,typename ResultingType<T2,typename ResultingType<T3,typename ResultingType<T4,typename ResultingType<T5,typename ResultingType<T6,typename ResultingType<T7, typename ResultingType<T8, typename ResultingType<T9,typename ResultingType<T10,typename ResultingType<T11,typename ResultingType<T12,typename ResultingType<T13,typename ResultingType<T14,typename ResultingType<T15,T16>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type>::Type Type;
	};

	// Type index/list :
	template<typename T, typename N> 
	struct TypeIndex
	{
		typedef T Type;
		typedef N Next;
	};

	typedef TypeIndex< void,
		TypeIndex< bool,
		TypeIndex< char,
		TypeIndex< unsigned char,
		TypeIndex< short,
		TypeIndex< unsigned short,
		TypeIndex< int,
		TypeIndex< unsigned int,
		TypeIndex< long long,
		TypeIndex< unsigned long long,
		TypeIndex< float,
		TypeIndex< double,
		TypeIndex< long double,
		Void
		> > > > > > > > > > > > > IntegralTypes;

	template<typename T>
	class GetTypeIndex
	{
		private :
			template<typename S, typename Q>
			struct worker;

			template<typename S>
			struct worker<S, Void>
			{
				static const int index = -1;
			};

			template<typename S, typename Q>
			struct worker<S, TypeIndex<S, Q> >
			{
				static const int index = 0;
			};

			template<typename S, typename H, typename Q>
			struct worker<S, TypeIndex<H, Q> >
			{
				static const int lower = worker<S, Q>::index;
				static const int index = lower<0 ? -1 : lower + 1;
			};
		public :
			static const int index = worker<typename Traits<T>::BaseType, IntegralTypes>::index;
	};

	__host__ inline size_t sizeOf(const int index)
	{
		switch(index)
		{
			#define CASE(x) case GetTypeIndex<x>::index : return sizeof(x);
			CASE(bool)
			CASE(char)
			CASE(unsigned char)
			CASE(short)
			CASE(unsigned short)
			CASE(int)
			CASE(unsigned int)
			CASE(long long)
			CASE(unsigned long long)
			CASE(float)
			CASE(double)
			CASE(long double)
			case GetTypeIndex<void>::index : 
			default :
				throw UnknownTypeIndex;
		}
	}

	// SourceType argument will refer to TSrc, and make use of the typedef.
	#define DYNAMIC_COPY_CASE( SourceType ) \
		switch(srcTypeIndex) \
		{ \
			case GetTypeIndex< bool >::index :			{ typedef bool TSrc; 			const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< char >::index :			{ typedef char TSrc; 			const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< unsigned char >::index :		{ typedef unsigned char TSrc; 		const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< short >::index :			{ typedef short TSrc; 			const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< unsigned short >::index :		{ typedef unsigned short TSrc; 		const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< int >::index :			{ typedef int TSrc; 			const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< unsigned int >::index :		{ typedef unsigned int TSrc; 		const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< long long >::index :			{ typedef long long TSrc; 		const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< unsigned long long >::index :	{ typedef unsigned long long TSrc; 	const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< float >::index :			{ typedef float TSrc; 			const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< double >::index :			{ typedef double TSrc; 			const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			case GetTypeIndex< long double >::index :		{ typedef long double TSrc; 		const SourceType * _src = reinterpret_cast<const SourceType *>(src); std::copy(_src, _src+count, dst); } break; \
			default : \
				throw UnknownTypeIndex; \
		}

	template<typename T>
	__host__ inline void dynamicCopy(T* dst, const void* src, const int srcTypeIndex, const bool srcIsComplex, const size_t count)
	{
		if(srcIsComplex)
			throw InvalidOperation;
		else
			DYNAMIC_COPY_CASE( TSrc )
	}

	template<typename T>
	__host__ inline void dynamicCopy(Complex<T>* dst, const void* src, const int srcTypeIndex, const bool srcIsComplex, const size_t count)
	{
		if(!srcIsComplex)
			DYNAMIC_COPY_CASE( TSrc )
		else
			DYNAMIC_COPY_CASE( Complex<TSrc> )
	}

	// Specific type tools :
	template<typename T>
	bool equalsMinusZero(const T& x)
	{
		typedef Traits<T> TypeTraits;
		typedef typename TypeTraits::BaseType BaseType;
		const BaseType minusZero = -static_cast<BaseType>(0);
		return !TypeTraits::isComplex && (IsSame<BaseType, float>::value || IsSame<BaseType, double>::value) && compareBits(x, minusZero);
	}
}

#endif

