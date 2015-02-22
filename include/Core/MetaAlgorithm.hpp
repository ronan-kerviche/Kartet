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

#ifndef __KARTET_META_ALGORITHM__
#define __KARTET_META_ALGORITHM__

namespace Kartet
{
	// Static assertion : FAIL TO COMPILE IF THE TEST IS FALSE
	template<bool test>
	struct StaticAssert;

	template<>
	struct StaticAssert<true>
	{ };

	#define STATIC_ASSERT( ... ) Kartet::StaticAssert< __VA_ARGS__ >();

	// If statement
	template<bool statement, class TrueStatement, class FalseStatement>
	struct MetaIf;

	template<class TrueStatement, class FalseStatement>
	struct MetaIf< true, TrueStatement, FalseStatement>
	{
		typedef TrueStatement TValue;
	};

	template<class TrueStatement, class FalseStatement>
	struct MetaIf< false , TrueStatement, FalseStatement>
	{
		typedef FalseStatement TValue;
	};

	// While Loop statement
	// IMPORTANT : a cycle is done after the test is false...
	/*template<bool statement, template<class Argument> class Algorithm, class Argument>
	struct MetaLoopWhile;

	template< template<class Argument> class Algorithm, class Argument>
	struct MetaLoopWhile<false, Algorithm, Argument>
	{
		typedef Argument Result;
	};

	template< template<class Argument>class Algorithm, class Argument>
	struct MetaLoopWhile<true, Algorithm, Argument>
	{
		typedef typename Algorithm<Argument>::NewArgument NewArgument;

		typedef typename MetaLoopWhile< Algorithm<Argument>::TestValue, Algorithm, NewArgument>::Result Result;
	};

	template<template<class Argument> class Algorithm, class Argument>
	struct MetaWhile
	{
		typedef typename MetaLoopWhile< Algorithm<Argument>::TestValue, Algorithm, Argument>::Result Result;
	};*/

	/* Test :
	//Useless
	//struct __TestArgument
	//{
	//    static const int dump;
	//};

	struct __TestStartArgument
	{
		static const int dump=0;
	};

	template< class __TestArgument>
	struct __TestAlgorithm
	{
		static const bool TestValue = __TestArgument::dump+1 < 32;

		struct NewArgument
		{
			static const int dump = __TestArgument::dump + 1;
		};
	};

	// test it :
	// int u = META_WHILE< __TestAlgorithm, __TestStartArgument>::Result::dump;

	// Find a root :
	template<unsigned int square>
	struct __StartArgument
	{
		static const unsigned int root=0;
		static const unsigned int sq=square;
	};

	template< class __Argument>
	struct __FindIntegerRoot
	{
		static const bool TestValue = (__Argument::root+1)*(__Argument::root+1)<__Argument::sq;

		struct NewArgument
		{
			static const unsigned int root= __Argument::root+1;
			static const unsigned int sq = __Argument::sq;
		};
	};

	//test it :
	//int root = META_WHILE< __FindIntegerRoot, __StartArgument<16> >::Result::root;
	*/

	// For Loop statement
	template<bool statement, int start, int end, template<int current, class Argument> class Algorithm, class Argument>
	struct MetaLoopFor;

	template<int start, int end, template<int current, class Argument> class Algorithm, class Argument>
	struct MetaLoopFor<false, start, end, Algorithm, Argument>
	{
		typedef Argument Result;
	};

	template<int start, int end, template<int current, class Argument> class Algorithm, class Argument>
	struct MetaLoopFor<true, start, end, Algorithm, Argument>
	{
		private: //Temp
			typedef Algorithm<start, Argument> NewStep;
			static const int tmp = NewStep::next; //calculate new index
			typedef typename NewStep::NewArgument NewArgument; // get new argument : Newargument = Algorithm(OldArgument);
		public:
			typedef typename MetaLoopFor< (tmp<end), tmp, end, Algorithm, NewArgument>::Result Result;
	};

	template<int start, int end, template<int current, class Argument> class Algorithm, class Argument>
	struct MetaFor
	{
		typedef typename MetaLoopFor< (start<end), start, end, Algorithm, Argument>::Result Result;
	};

	/*Test :
	//struct __TestArgument
	//{
	//    static const int dump;
	//    static const int count;
	//};

	struct __TestStartArgument
	{
		static const int dump=0;
		static const int count=0;
	};

	template< int current, class __TestArgument>
	struct __TestAlgorithm
	{
		static const int next = __TestArgument::dump;

		struct NewArgument
		{
			static const int dump  = __TestArgument::dump  + 1 + current;
			static const int count = __TestArgument::count + 1;
		};
	};

	// Test it :
	//int u = META_FOR< 1,1024,__TestAlgorithm, __TestStartArgument>::Result::dump;
	//int v = META_FOR< 1,1024,__TestAlgorithm, __TestStartArgument>::Result::count;

	//WARNING : here the computation is done twice, prefer :
	//typedef typename META_FOR< 1,1024,__TestAlgorithm, __TestStartArgument>::Result Res
	//int u = Res::dump;
	//int v = Res::count;
	*/

} // namespace Kartet

#endif

