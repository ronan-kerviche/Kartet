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
	/*#include <complex>

	#if defined(__CUDACC__)
		#include <cufft.h>
	#endif*/

	#include "Core/MetaList.hpp"
	#include "Core/MetaAlgorithm.hpp"
	
namespace Kartet
{
// Prototypes :
	template<typename T>
	struct IsComplex;
	
// Tools
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
		/*TypeList< std::complex<float>
		#if defined(__CUDACC__)
			TypeList< cuFloatComplex,
			TypeList< std::complex<double>,
			TypeList< cuDoubleComplex,
			Void
			> > >
		#else
			TypeList< std::complex<double>,
			Void
			>
		#endif*/
		Void
		/*>*/ > > > > > > > > > > > > > > > > > > > > TypesSortedByAccuracy;

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
			//static const bool test2 = (IsComplex<T1>::test && SameTypes<typename RemovePointer<T2>::Type ,double>::test) || (IsComplex<T2>::test && SameTypes<typename RemovePointer<T1>::Type ,double>::test);
		public :
			#if defined(__CUDACC__)
				//typedef typename MetaIf<test2, cuDoubleComplex, StdType>::TValue Type;
				typedef StdType Type;
			#else
				typedef StdType Type;
			#endif
	};

} // Namespace Kartet

#endif

