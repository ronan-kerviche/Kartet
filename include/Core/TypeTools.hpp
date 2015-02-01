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
		
		#if defined(__CUDACC__)
			typedef cuFloatComplex ComplexType;
		#else
			typedef std::complex<float> ComplexType;
		#endif		

		static const bool 	isConst 	= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= false;
	};

	template<>
	struct TypeInfo<double>
	{
		typedef double SubType;
		typedef double BaseType;
		
		#if defined(__CUDACC__)
			typedef cuDoubleComplex ComplexType;
		#else
			typedef std::complex<double> ComplexType;
		#endif		

		static const bool 	isConst 	= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= false;
	};

	template<typename T>
	struct TypeInfo<T*>
	{
		typedef TypeInfo<T> SubType;
		typedef typename SubType::BaseType BaseType;
		typedef typename SubType::ComplexType ComplexType;

		static const bool 	isConst 	= SubType::isConst,
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

		static const bool 	isConst 	= SubType::isConst,
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

		static const bool 	isConst 	= true,
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
			static const bool 	isConst 	= false,
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
			static const bool 	isConst 	= false,
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
			TypeList< std::complex<double>,
			TypeList< cuDoubleComplex,
			Void
			> > >
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
			static const int i1 = GetIndex< TypesSortedByAccuracy, pT1>::Value;
			static const int i2 = GetIndex< TypesSortedByAccuracy, pT2>::Value;

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

} // Namespace Kartet

	#include "ComplexOperators.hpp"

#endif

