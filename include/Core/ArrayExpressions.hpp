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

#ifndef __KARTET_ARRAY_EXPRESSIONS__
#define __KARTET_ARRAY_EXPRESSIONS__

	#include "Core/LibTools.hpp"

namespace Kartet
{
// Evaluation tools :
	template<typename T>
	struct ExpressionEvaluation
	{
		// Default expression mechanism (transparent).
		typedef T ReturnType;
		static const Location location = AnySide;

		__host__ __device__ inline static ReturnType evaluate(const T& expr, const Layout& layout, const index_t& i, const index_t& j, const index_t& k)
		{
			UNUSED_PARAMETER(expr)
			UNUSED_PARAMETER(layout)
			UNUSED_PARAMETER(i)
			UNUSED_PARAMETER(j)
			UNUSED_PARAMETER(k)
			return expr;
		}
	};

	template<typename T>
	struct ExpressionEvaluation<T*>;

	template<typename T>
	struct ExpressionEvaluation< ExpressionContainer<T> >
	{
		// Mechanism for a ExpressionContainer Object :
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType;
		static const Location location = ExpressionEvaluation<T>::location;

		__host__ __device__ inline static ReturnType evaluate(const ExpressionContainer<T>& container, const Layout& layout, const index_t& i, const index_t& j, const index_t& k)
		{
			UNUSED_PARAMETER(container)
			UNUSED_PARAMETER(layout)
			UNUSED_PARAMETER(i)
			UNUSED_PARAMETER(j)
			UNUSED_PARAMETER(k)
			return ExpressionEvaluation<T>::evaluate(container.expr, layout, i, j, k);
		}
	};

	template<typename T, Location l>
	struct ExpressionEvaluation< Accessor<T,l> >
	{
		// Mechanism for a Accessor Object :
		typedef T ReturnType;
		static const Location location = l;
		
		__host__ __device__ inline static ReturnType evaluate(const Accessor<T,l>& accessor, const Layout& layout, const index_t& i, const index_t& j, const index_t& k)
		{
			UNUSED_PARAMETER(layout)
			UNUSED_PARAMETER(i)
			UNUSED_PARAMETER(j)
			UNUSED_PARAMETER(k)
			return accessor.data(i, j, k);
		}
	};

	template<typename T, Location l>
	struct ExpressionEvaluation< Array<T,l> >
	{
		// Mechanism for a Array Object :
		typedef T ReturnType;
		static const Location location = l;
		
		__host__ __device__ inline static ReturnType evaluate(const Array<T,l>& accessor, const Layout& layout, const index_t& i, const index_t& j, const index_t& k)
		{
			UNUSED_PARAMETER(layout)
			UNUSED_PARAMETER(i)
			UNUSED_PARAMETER(j)
			UNUSED_PARAMETER(k)
			return accessor.data(i, j, k);
		}
	};

	template<class Op>
	struct ExpressionEvaluation< NullaryExpression<Op> >
	{
		// Mechanism for a NullaryExpression :
		typedef Op Operator;
		typedef typename Op::ReturnType ReturnType;
		static const Location location = AnySide;
		
		__host__ __device__ inline static ReturnType evaluate(const NullaryExpression<Op>& nullaryExpression, const Layout& layout, const index_t& i, const index_t& j, const index_t& k)
		{
			UNUSED_PARAMETER(nullaryExpression)
			return Operator::apply(layout, i, j, k);
		}
	};

	template< typename T, template<typename> class Op >
	struct ExpressionEvaluation< UnaryExpression<T, Op> >
	{
		// Mechanism for a UnaryExpression :
		typedef Op<		typename ExpressionEvaluation<T>::ReturnType 
					> Operator;
		typedef typename Op< 	typename ExpressionEvaluation<T>::ReturnType 
					>::ReturnType ReturnType;
		static const Location location = ExpressionEvaluation<T>::location;
		
		__host__ __device__ inline static ReturnType evaluate(const UnaryExpression<T, Op>& unaryExpression, const Layout& layout, const index_t& i, const index_t& j, const index_t& k)
		{
			return Operator::apply(	ExpressionEvaluation<T>::evaluate(unaryExpression.a, layout, i, j, k) );
		}
	};

	template< typename T, class Op >
	struct ExpressionEvaluation< TransformExpression<T, Op> >
	{
		// Mechanism for a TransformExpression :
		typedef Op Operator;
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType;
		static const Location location = ExpressionEvaluation<T>::location;
		
		__host__ __device__ inline static ReturnType evaluate(const TransformExpression<T, Op>& transfomExpression, const Layout& layout, index_t i, index_t j, index_t k)
		{
			Operator::apply(layout, i, j, k); // Change in shape
			return ExpressionEvaluation<T>::evaluate(transfomExpression.a, layout, i, j, k); // Use sub type with new coordinates
		}
	};

	template< typename T, class Op >
	struct ExpressionEvaluation< LayoutReinterpretationExpression<T, Op> >
	{
		// Mechanism for a LayoutReinterpretationExpression :
		typedef Op Operator;
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType;
		static const Location location = ExpressionEvaluation<T>::location;
		
		__host__ __device__ inline static ReturnType evaluate(const LayoutReinterpretationExpression<T, Op>& layoutReinterpretationExpression, const Layout& layout, index_t i, index_t j, index_t k)
		{
			Operator::apply(layout, layoutReinterpretationExpression.layout, i, j, k); // Change in shape according to a new layout
			return ExpressionEvaluation<T>::evaluate(layoutReinterpretationExpression.a, layoutReinterpretationExpression.layout, i, j, k); // Use sub type with new coordinates AND new layout
		}
	};

	template< typename T1, typename T2, template<typename,typename> class Op >
	struct ExpressionEvaluation< BinaryExpression<T1, T2, Op> >
	{
		// Mechanism for a BinaryExpression :
		typedef Op< 		typename ExpressionEvaluation<T1>::ReturnType, 
					typename ExpressionEvaluation<T2>::ReturnType 
					> Operator;
		typedef typename Op< 	typename ExpressionEvaluation<T1>::ReturnType, 
					typename ExpressionEvaluation<T2>::ReturnType 
					>::ReturnType ReturnType;
		static const Location 	location = StaticIf<ExpressionEvaluation<T1>::location!=AnySide, ExpressionEvaluation<T1>, ExpressionEvaluation<T2> >::TValue::location;

		__host__ __device__ inline static ReturnType evaluate(const BinaryExpression<T1, T2, Op>& binaryExpression, const Layout& layout, const index_t& i, const index_t& j, const index_t& k)
		{
			// Both branches must be on the same side.
			StaticAssert<	(ExpressionEvaluation<T1>::location==ExpressionEvaluation<T2>::location) ||
					(ExpressionEvaluation<T1>::location==AnySide) ||
					(ExpressionEvaluation<T2>::location==AnySide)>(); 

			return Operator::apply(		ExpressionEvaluation<T1>::evaluate(binaryExpression.a, layout, i, j, k), 	
							ExpressionEvaluation<T2>::evaluate(binaryExpression.b, layout, i, j, k)
						);
		}
	};

	template< typename TIndex, typename TData, template<typename> class Op >
	struct ExpressionEvaluation< ShuffleExpression<TIndex, TData, Op> >
	{
		// Mechanism for a ShuffleExpression :
		typedef Op<TIndex> Operator;
		typedef typename ExpressionEvaluation<TData>::ReturnType ReturnType;
		static const Location location = ExpressionEvaluation<TData>::location;
		
		__host__ __device__ inline static ReturnType evaluate(const ShuffleExpression<TIndex, TData, Op>& shuffleExpression, const Layout& layout, index_t i, index_t j, index_t k)
		{
			Operator::apply(shuffleExpression.index, layout, i, j, k); // Change in shape from data
			return ExpressionEvaluation<TData>::evaluate(shuffleExpression.data, layout, i, j, k); // Use sub type with new coordinates
		}
	};

	template< typename T1, typename T2, typename T3, template<typename,typename,typename> class Op >
	struct ExpressionEvaluation< TernaryExpression<T1, T2, T3, Op> >
	{
		// Mechanism for a TernaryExpression :

		typedef Op< 		typename ExpressionEvaluation<T1>::ReturnType, 
					typename ExpressionEvaluation<T2>::ReturnType, 
					typename ExpressionEvaluation<T3>::ReturnType 
					> Operator;
		typedef typename Op< 	typename ExpressionEvaluation<T1>::ReturnType, 
					typename ExpressionEvaluation<T2>::ReturnType, 
					typename ExpressionEvaluation<T3>::ReturnType >::ReturnType 
					ReturnType;
		static const Location location = StaticIf<ExpressionEvaluation<T1>::location!=AnySide, ExpressionEvaluation<T1>, 
						 StaticIf<ExpressionEvaluation<T2>::location!=AnySide, ExpressionEvaluation<T2>, ExpressionEvaluation<T3> > >::TValue::location;
		
		__host__ __device__ inline static ReturnType evaluate(const TernaryExpression<T1, T2, T3, Op>& ternaryExpression, const Layout& layout, const index_t& i, const index_t& j, const index_t& k)
		{
			// All branches must be on the same side.
			StaticAssert<	(ExpressionEvaluation<T1>::location==ExpressionEvaluation<T2>::location ||
					 ExpressionEvaluation<T1>::location==AnySide ||
					 ExpressionEvaluation<T2>::location==AnySide) &&
					(ExpressionEvaluation<T2>::location==ExpressionEvaluation<T3>::location ||
					 ExpressionEvaluation<T2>::location==AnySide ||
					 ExpressionEvaluation<T3>::location==AnySide) &&
					(ExpressionEvaluation<T1>::location==ExpressionEvaluation<T3>::location ||
					 ExpressionEvaluation<T1>::location==AnySide ||
					 ExpressionEvaluation<T3>::location==AnySide)	>();

			return Operator::apply(		ExpressionEvaluation<T1>::evaluate(ternaryExpression.a, layout, i, j, k), 
							ExpressionEvaluation<T2>::evaluate(ternaryExpression.b, layout, i, j, k),
							ExpressionEvaluation<T3>::evaluate(ternaryExpression.c, layout, i, j, k)
						);
		}
	};

// Expression Container (transparent expression, used as an expression filter) :
	template<typename T>
	struct ExpressionContainer
	{
		T expr;

		__host__ ExpressionContainer(const T& e)
		 : expr(e)
		{ }

		__host__ __device__ ExpressionContainer(const ExpressionContainer<T>& e)
		 : expr(e.expr)
		{ }
	};

// Nullary Expression 
	template<class Op>
	struct NullaryExpression
	{
		__host__ NullaryExpression(void)
		{ }

		__host__ __device__ NullaryExpression(const NullaryExpression<Op>& e)
		{
			UNUSED_PARAMETER(e)
		}
	};

// Unary Expression
	template<typename T, template<typename> class Op >
	struct UnaryExpression
	{
		T a;

		__host__ UnaryExpression(const T& _a)
		 : a(_a)
		{ }

		__host__ __device__ UnaryExpression(const UnaryExpression<T, Op>& e)
		 : a(e.a)
		{ }
	};

// Transform Expression
	template<typename T, class Op >
	struct TransformExpression
	{
		T a;

		__host__ TransformExpression(const T& _a)
		 : a(_a)
		{ }

		__host__ __device__ TransformExpression(const TransformExpression<T, Op>& e)
		 : a(e.a)
		{ }
	};

// Layout Reinterpretation Expression
	template<typename T, class Op >
	struct LayoutReinterpretationExpression
	{
		Layout layout;
		T a;
		
		__host__ LayoutReinterpretationExpression(const Layout& _layout, const T& _a)
		 : layout(_layout), a(_a) 
		{ }

		__host__ __device__ LayoutReinterpretationExpression(const LayoutReinterpretationExpression<T, Op>& e)
		 : layout(e.layout), a(e.a) 
		{ }
	};

// Binary Operators
	template<typename T1, typename T2, template<typename,typename> class Op >
	struct BinaryExpression
	{
		T1 a;
		T2 b;

		__host__ BinaryExpression(const T1& _a, const T2& _b)
		 : a(_a), b(_b)
		{ }

		__host__ __device__ BinaryExpression(const BinaryExpression<T1,T2,Op>& e)
		 : a(e.a), b(e.b)
		{ }
	};

// Shuffle Expression
	template<typename TIndex, typename TData, template<typename> class Op >
	struct ShuffleExpression
	{
		TIndex index;
		TData data;

		__host__ ShuffleExpression(const TIndex& _index, const TData& _data)
		 : index(_index), data(_data)
		{ }

		__host__ __device__ ShuffleExpression(const ShuffleExpression<TIndex, TData, Op>& e)
		 : index(e.index), data(e.data)
		{ }
	};

// Ternary Operators :
	template<typename T1, typename T2, typename T3, template<typename, typename, typename> class Op >
	struct TernaryExpression
	{
		T1 a;
		T2 b;
		T3 c;
		
		__host__ TernaryExpression(const T1& _a, const T2& _b, const T3& _c)
		 : a(_a), b(_b), c(_c)
		{ }

		__host__ __device__ TernaryExpression(const TernaryExpression<T1,T2,T3,Op>& e)
		 : a(e.a), b(e.b), c(e.c)
		{ }
	};

// Evaluation functions :
#ifdef __CUDACC__
	template<typename T, Location l, typename TExpr>
	__global__ void evaluateExpression(const Accessor<T,l> array, const TExpr expr)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;

		index_t i = array.getI(),
			j = array.getJ(),
			k = array.getK();

		if(array.isInside(i, j, k))
		{
			ReturnType t = ExpressionEvaluation<TExpr>::evaluate(expr, array, i, j, k);

			T buffer;
			complexCopy(buffer, t);
			array.data() = buffer;
		}
	}
#endif

	template<typename T, Location l, typename TExpr>
	__host__ void evaluateExpressionOverLayout(const Accessor<T,l>& array, const TExpr& expr)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
		
		for(index_t k=0, j=0, i=0, q=0; q<array.numElements(); q++)
		{
			ReturnType t = ExpressionEvaluation<TExpr>::evaluate(expr, array, i, j, k);

			T buffer;
			complexCopy(buffer, t);
			array.data(i, j, k) = buffer;

			array.moveToNext(i, j, k);
		}
	}

#ifdef __CUDACC__
	template<typename T, typename TExprMask, typename TExpr>
	__global__ void evaluateExpressionWithMask(const Accessor<T> array, const TExprMask exprMask, const TExpr expr)
	{
		typedef typename ExpressionEvaluation<TExprMask>::ReturnType MaskType;
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;

		index_t i = array.getI(),
			j = array.getJ(),
			k = array.getK();

		if(array.isInside(i, j, k))
		{
			const MaskType test = ExpressionEvaluation<TExprMask>::evaluate(exprMask, array, i, j, k);
			if(test)
			{
				ReturnType t = ExpressionEvaluation<TExpr>::evaluate(expr, array, i, j, k);
				
				T buffer;
				complexCopy(buffer, t);
				array.data() = buffer;
			}
		}
	}
#endif

	template<typename T, Location l, typename TExprMask, typename TExpr>
	__host__ void evaluateExpressionWithMaskOverLayout(const Accessor<T,l>& array, const TExprMask& exprMask, const TExpr& expr)
	{
		typedef typename ExpressionEvaluation<TExprMask>::ReturnType MaskType;
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;

		for(index_t k=0, j=0, i=0, q=0; q<array.numElements(); q++)
		{
			const MaskType test = ExpressionEvaluation<TExprMask>::evaluate(exprMask, array, i, j, k);
			if(test)
			{
				ReturnType t = ExpressionEvaluation<TExpr>::evaluate(expr, array, i, j, k);

				T buffer;
				complexCopy(buffer, t);
				array.data(i, j, k) = buffer;
			}
			array.moveToNext(i, j, k);
		}
	}

} // Namespace Kartet

#endif

