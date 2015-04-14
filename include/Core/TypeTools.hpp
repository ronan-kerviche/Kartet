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

#ifndef __KARTET_TYPE_TOOLS__
#define __KARTET_TYPE_TOOLS__

// Includes :
	#include <limits>
	#include <complex>
	#if defined(__CUDACC__)
		#include <cufft.h>
	#endif
	#include "Core/MetaList.hpp"
	#include "Core/MetaAlgorithm.hpp"
	
namespace Kartet
{
// Tools :
	template<typename T1, typename T2>
	struct SameTypes
	{
		static const bool test = false;
	};
	
	template<typename T>
	struct SameTypes<T,T>
	{
		static const bool test = true;
	}; 

	template<typename T>
	struct TypeInfo
	{
		typedef T SubType;
		typedef T BaseType;
		typedef std::numeric_limits<BaseType> BaseLimits;
		#if defined(__CUDACC__)
			typedef cuFloatComplex ComplexType;
		#else
			typedef std::complex<float> ComplexType;
		#endif		

		static const bool 	isSigned	= BaseLimits::is_signed,
					isConst 	= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= false;
	};

	template<>
	struct TypeInfo<double>
	{
		typedef double SubType;
		typedef double BaseType;
		typedef std::numeric_limits<BaseType> BaseLimits;
		#if defined(__CUDACC__)
			typedef cuDoubleComplex ComplexType;
		#else
			typedef std::complex<double> ComplexType;
		#endif		

		static const bool 	isSigned	= BaseLimits::is_signed,
					isConst 	= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= false;
	};

	template<typename T>
	struct TypeInfo<T*>
	{
		typedef std::numeric_limits<T> Limits;
		typedef TypeInfo<T> SubType;
		typedef typename SubType::BaseType BaseType;
		typedef typename SubType::ComplexType ComplexType;
		typedef std::numeric_limits<BaseType> BaseLimits;

		static const bool 	isSigned	= BaseLimits::is_signed,
					isConst 	= SubType::isConst,
					isPointer 	= true,
					isReference 	= SubType::isReference,
					isComplex 	= SubType::isComplex;
	};

	template<typename T>
	struct TypeInfo<T&>
	{
		typedef TypeInfo<T> SubType;
		typedef typename SubType::BaseType BaseType;
		typedef typename SubType::ComplexType ComplexType;
		typedef std::numeric_limits<BaseType> BaseLimits;

		static const bool 	isSigned	= BaseLimits::is_signed,
					isConst 	= SubType::isConst,
					isPointer 	= SubType::isPointe,
					isReference 	= true,
					isComplex 	= SubType::isComplex;
	};

	template<typename T>
	struct TypeInfo<const T>
	{
		typedef TypeInfo<T> SubType;
		typedef typename SubType::BaseType BaseType;
		typedef typename SubType::ComplexType ComplexType;
		typedef std::numeric_limits<BaseType> BaseLimits;

		static const bool 	isSigned	= BaseLimits::is_signed,
					isConst 	= true,
					isPointer 	= SubType::isPointer,
					isReference 	= SubType::isReference,
					isComplex 	= SubType::isComplex;
	};

	#if defined(__CUDACC__)
		template<>
		struct TypeInfo<cuFloatComplex>
		{
			typedef float SubType;
			typedef float BaseType;
			typedef cuFloatComplex ComplexType;
			typedef std::numeric_limits<BaseType> BaseLimits;

			static const bool 	isSigned	= BaseLimits::is_signed,
						isConst 	= false,
						isPointer 	= false,
						isReference 	= false,
						isComplex 	= true;
		};

		template<>
		struct TypeInfo<cuDoubleComplex>
		{
			typedef double SubType;
			typedef double BaseType;
			typedef cuDoubleComplex ComplexType;
			typedef std::numeric_limits<BaseType> BaseLimits;

			static const bool 	isSigned	= BaseLimits::is_signed,
						isConst 	= false,
						isPointer 	= false,
						isReference 	= false,
						isComplex 	= true;
		};
	#endif

	template<typename T>
	struct RemovePointer
	{
		typedef T Type;
	};
	
	template<typename T>
	struct RemovePointer<T*>
	{
		typedef T Type;
	};

// Precision Tool :
	typedef TypeList< void,
		TypeList< bool,
		TypeList< unsigned char,
		TypeList< char,
		TypeList< signed char,
		TypeList< unsigned short,
		TypeList< short,
		TypeList< signed short,
		TypeList< unsigned int,
		TypeList< int,
		TypeList< signed int,
		TypeList< unsigned long,
		TypeList< long,
		TypeList< signed long,
		TypeList< unsigned long long,
		TypeList< long long,
		TypeList< signed long long,
		TypeList< float,
		TypeList< double,
		TypeList< long double,
		#if defined(__CUDACC__)
			TypeList< cuFloatComplex,
			TypeList< cuDoubleComplex,
			Void
			> >
		#else
			TypeList< std::complex<float>,
			TypeList< std::complex<double>,
			Void
			> >
		#endif
		> > > > > > > > > > > > > > > > > > > > TypesSortedByAccuracy;

	template<typename T>
	struct ResultingTypeOf
	{
		typedef typename RemovePointer< T >::Type Type;
	};

	template<typename T1, typename T2>
	struct ResultingTypeOf2
	{
		private :
			typedef typename RemovePointer< T1 >::Type pT1;
			typedef typename RemovePointer< T2 >::Type pT2;
			static const int i1 = GetIndex< TypesSortedByAccuracy, pT1>::value;
			static const int i2 = GetIndex< TypesSortedByAccuracy, pT2>::value;

			// Test if the types are known first : 
			StaticAssert<i1!=-1> C1;
			StaticAssert<i2!=-1> C2;

			static const bool test1 = (i1>=i2); // First type has higher accuracy.
			typedef typename MetaIf<test1, pT1, pT2>::TValue StdType;
			static const bool test2 = (TypeInfo<T1>::isComplex && SameTypes<typename RemovePointer<T2>::Type ,double>::test) || (TypeInfo<T2>::isComplex && SameTypes<typename RemovePointer<T1>::Type ,double>::test);
		public :
			#if defined(__CUDACC__)
				typedef typename MetaIf<test2, cuDoubleComplex, StdType>::TValue Type;
			#else
				typedef StdType Type;
			#endif
	};
}

#include "Core/ComplexOperators.hpp"

namespace Kartet
{
	// Dynamic tools for the types :
	inline size_t sizeOfType(int typeIndex)
	{
		#define SPECIAL_SIZE( TypeName )	case GetIndex<TypesSortedByAccuracy, TypeName>::value :	return sizeof( TypeName );

		switch(typeIndex)
		{
			SPECIAL_SIZE( bool )
			SPECIAL_SIZE( unsigned char )
			SPECIAL_SIZE( char )
			SPECIAL_SIZE( signed char )
			SPECIAL_SIZE( unsigned short )
			//SPECIAL_SIZE( short )
			SPECIAL_SIZE( signed short )
			SPECIAL_SIZE( unsigned int )
			//SPECIAL_SIZE( int )
			SPECIAL_SIZE( signed int )
			SPECIAL_SIZE( unsigned long )
			//SPECIAL_SIZE( long )
			SPECIAL_SIZE( signed long )
			SPECIAL_SIZE( unsigned long long )
			//SPECIAL_SIZE( long long )
			SPECIAL_SIZE( signed long long )
			SPECIAL_SIZE( float )
			SPECIAL_SIZE( double )
			SPECIAL_SIZE( long double )
			#if defined(__CUDACC__)
				SPECIAL_SIZE( cuFloatComplex )
				SPECIAL_SIZE( cuDoubleComplex )
			#else
				SPECIAL_SIZE( std::complex<float> )
				SPECIAL_SIZE( std::complex<double> )
			#endif
			default :
				throw UnknownTypeIndex;
		}
	}

	inline size_t isComplex(int typeIndex)
	{
		#define SPECIAL_CXTEST( TypeName , v)	case GetIndex<TypesSortedByAccuracy, TypeName>::value :	return v;

		switch(typeIndex)
		{
			SPECIAL_CXTEST( bool, false )
			SPECIAL_CXTEST( unsigned char, false )
			SPECIAL_CXTEST( char, false )
			SPECIAL_CXTEST( signed char, false )
			SPECIAL_CXTEST( unsigned short, false )
			//SPECIAL_CXTEST( short, false )
			SPECIAL_CXTEST( signed short, false )
			SPECIAL_CXTEST( unsigned int, false )
			//SPECIAL_CXTEST( int, false )
			SPECIAL_CXTEST( signed int, false )
			SPECIAL_CXTEST( unsigned long, false )
			//SPECIAL_CXTEST( long, false )
			SPECIAL_CXTEST( signed long, false )
			SPECIAL_CXTEST( unsigned long long, false )
			//SPECIAL_CXTEST( long long, false )
			SPECIAL_CXTEST( signed long long, false )
			SPECIAL_CXTEST( float, false )
			SPECIAL_CXTEST( double, false )
			SPECIAL_CXTEST( long double, false )
			#if defined(__CUDACC__)
				SPECIAL_CXTEST( cuFloatComplex, true )
				SPECIAL_CXTEST( cuDoubleComplex, true )
			#else
				SPECIAL_CXTEST( std::complex<float>, true )
				SPECIAL_CXTEST( std::complex<double>, true )
			#endif
			default :
				throw UnknownTypeIndex;
		}
	}

	template<typename T>
	void copy(T* dst, const void* src, int typeIndex, size_t count=1)
	{
		#define SPECIAL_COPY( TypeName ) \
			case GetIndex<TypesSortedByAccuracy, TypeName>::value :	\
				for(int k=0; k<count; k++) \
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
			#if defined(__CUDACC__)
				SPECIAL_COPY( cuFloatComplex )
				SPECIAL_COPY( cuDoubleComplex )
			#else
				SPECIAL_COPY( std::complex<float> )
				SPECIAL_COPY( std::complex<double> )
			#endif
			default :
				throw UnknownTypeIndex;
		}
	}

} // Namespace Kartet

	#include "ComplexOperators.hpp"

#endif

