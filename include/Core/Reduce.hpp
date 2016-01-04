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
	\file    Reduce.hpp
	\brief   Reduction context definitions.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_REDUCE__
#define __KARTET_REDUCE__

	#include <limits>
	#include "Core/Exceptions.hpp"
	#include "Core/TemplateSharedMemory.hpp"
	#include "Core/Array.hpp"

namespace Kartet
{
	/**
	\brief Reduction context class.

	The reduction process is carried on the side of the data (see Kartet::Location).

	A simple example of global reduction : 
	\code
	Kartet::ReduceContext reduce;
	Kartet::Array<float> A(8, 8);
	A = Kartet::IndexI() + Kartet::IndexJ();
	std::cout << "A = " << A << std::endl;
	std::cout << "sum(A) = " << reduce.sum(A) << std::endl;
	\endcode

	An example of block reduction : 
	\code
	Kartet::ReduceContext reduce;
	Kartet::Array<float> A(8, 8), B(2, 2);
	A = Kartet::IndexI() + Kartet::IndexJ();
	Kartet::ReduceContext reduce;
	reduce.sumBlock(A, B); 			// Will reduce each 4x4 block to a single value.
	// B = reduce.sumBlock(A, B) - 48;	// We could use the output directly.
	std::cout << "A = " << A << std::endl;
	std::cout << "B = " << B << std::endl;
	\endcode
	**/
	class ReduceContext
	{
		private :
			const unsigned int fillNumBlocks;
			const size_t maxMemory;			
			char	*hostPtr,
				*devicePtr;

			__host__ ReduceContext(const ReduceContext&);
		public :
			__host__ inline ReduceContext(void);
			__host__ inline ~ReduceContext(void);
			
			template<template<typename,typename> class Op, typename TOut, typename TExpr>
			__host__ TOut reduce(const Layout& layout, const TExpr& expr, const typename ExpressionEvaluation<TExpr>::ReturnType defaultValue);
			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType min(const Layout& layout, const TExpr& expr);
			template<typename T, Location l>
			__host__ T min(const Accessor<T,l>& accessor);
			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType max(const Layout& layout, const TExpr& expr);
			template<typename T, Location l>
			__host__ T max(const Accessor<T,l>& accessor);
			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType sum(const Layout& layout, const TExpr& expr);
			template<typename T, Location l>
			__host__ T sum(const Accessor<T,l>& accessor);
			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType mean(const Layout& layout, const TExpr& expr);
			template<typename T, Location l>
			__host__ T mean(const Accessor<T,l>& accessor);
			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType prod(const Layout& layout, const TExpr& expr);
			template<typename T, Location l>
			__host__ T prod(const Accessor<T,l>& accessor);
			template<typename TExpr>
			__host__ bool all(const Layout& layout, const TExpr& expr);
			template<typename T, Location l>
			__host__ bool all(const Accessor<T,l>& accessor);
			template<typename TExpr>
			__host__ bool any(const Layout& layout, const TExpr& expr);
			template<typename T, Location l>
			__host__ bool any(const Accessor<T,l>& accessor);

			template<template<typename,typename> class Op, typename TExpr, typename TOut, Location l>
			__host__ Accessor<TOut,l>& reduceBlock(const Layout& layout, const TExpr& expr, const typename ExpressionEvaluation<TExpr>::ReturnType defaultValue, Accessor<TOut,l>& output);
			template<typename TExpr, typename TOut, Location l>
			__host__ Accessor<TOut,l>& minBlock(const Layout& layout, const TExpr& expr, Accessor<TOut,l>& output);
			template<typename T, typename TOut, Location l>
			__host__ Accessor<TOut,l>& minBlock(const Accessor<T,l>& accessor, Accessor<TOut,l>& output);
			template<typename TExpr, typename TOut, Location l>
			__host__ Accessor<TOut,l>& maxBlock(const Layout& layout, const TExpr& expr, Accessor<TOut,l>& output);
			template<typename T, typename TOut, Location l>
			__host__ Accessor<TOut,l>& maxBlock(const Accessor<T,l>& accessor, Accessor<TOut,l>& output);
			template<typename TExpr, typename TOut, Location l>
			__host__ Accessor<TOut,l>& sumBlock(const Layout& layout, const TExpr& expr, Accessor<TOut,l>& output);
			template<typename T, typename TOut, Location l>
			__host__ Accessor<TOut,l>& sumBlock(const Accessor<T,l>& accessor, Accessor<TOut,l>& output);
			template<typename TExpr, typename TOut, Location l>
			__host__ Accessor<TOut,l>& prodBlock(const Layout& layout, const TExpr& expr, Accessor<TOut,l>& output);
			template<typename T, typename TOut, Location l>
			__host__ Accessor<TOut,l>& prodBlock(const Accessor<T,l>& accessor, Accessor<TOut,l>& output);
			template<typename TExpr, typename TOut, Location l>
			__host__ Accessor<TOut,l>& allBlock(const Layout& layout, const TExpr& expr, Accessor<TOut,l>& output);
			template<typename T, typename TOut, Location l>
			__host__ Accessor<TOut,l>& allBlock(const Accessor<T,l>& accessor, Accessor<TOut,l>& output);
			template<typename TExpr, typename TOut, Location l>
			__host__ Accessor<TOut,l>& anyBlock(const Layout& layout, const TExpr& expr, Accessor<TOut,l>& output);
			template<typename T, typename TOut, Location l>
			__host__ Accessor<TOut,l>& anyBlock(const Accessor<T,l>& accessor, Accessor<TOut,l>& output);
	};

} // namespace Kartet

	#include "ReduceTools.hpp"

#endif

