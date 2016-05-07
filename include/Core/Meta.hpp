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
	\file    Meta.hpp
	\brief   Meta tools.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_META__
#define __KARTET_META__

namespace Kartet
{
	// Null type :
	struct Void
	{ };

	// Static assertion :
	template<bool test>
	struct StaticAssert;

	template<>
	struct StaticAssert<true>
	{ };

	// Static assertion with messages :
	template<bool test>
	struct StaticAssertVerbose;

	template<>
	struct StaticAssertVerbose<true>
	{
		enum
		{
			KARTET_STATIC_ERROR___RHS_IS_COMPLEX_BUT_LHS_IS_NOT,
			KARTET_STATIC_ERROR___LHS_AND_RHS_HAVE_INCOMPATIBLE_LOCATIONS,
			KARTET_STATIC_ERROR___ARGUMENTS_HAVE_INCOMPATIBLE_LOCATIONS,
			KARTET_STATIC_ERROR___STATIC_CONTAINER_MUST_HAVE_VOID_PARAMETER,
			KARTET_STATIC_ERROR___INVALID_LOCATION,
			KARTET_STATIC_ERROR___TYPE_NOT_SUPPORTED,
			KARTET_STATIC_ERROR___TYPE_MUST_BE_REAL,
			KARTET_STATIC_ERROR___TYPE_MUST_BE_COMPLEX,
			KARTET_STATIC_ERROR___LOCATION_NOT_SUPPORTED,
			KARTET_STATIC_ERROR___UNSUPPORTED_TYPE_SIZE,
			KARTET_STATIC_ERROR___INVALID_ARGUMENTS_LAYOUT,
			KARTET_STATIC_ERROR___ARGUMENTS_CONFLICT
		};
	};
	
	template<>
	struct StaticAssertVerbose<false>
	{ };

	template<int>
	struct StaticAssertCapsule
	{ };

	#define _JOIN(x, y) x ## y
	#define JOIN(x, y) _JOIN(x, y)
	#define STATIC_ASSERT_VERBOSE(test, message) typedef Kartet::StaticAssertCapsule<Kartet::StaticAssertVerbose<(test)>::KARTET_STATIC_ERROR___##message> JOIN(StaticAssertTest, __LINE__);

	// Static tests :
	template<bool test, typename TrueResult, typename FalseResult>
	struct StaticIf;

	template<typename TrueResult, typename FalseResult>
	struct StaticIf<true, TrueResult, FalseResult>
	{
		typedef TrueResult Type;
	};

	template<typename TrueResult, typename FalseResult>
	struct StaticIf<false, TrueResult, FalseResult>
	{
		typedef FalseResult Type;
	};

	// Type test :
	template<typename T1, typename T2>
	struct IsSame
	{
		static const bool value = false;
	};

	template<typename T>
	struct IsSame<T,T>
	{
		static const bool value = true;
	};
}

#endif

