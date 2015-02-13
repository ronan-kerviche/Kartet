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
		
		__device__ inline static ReturnType evaluate(const T& expr, const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k)
		{
			return expr;
		}
	};

	template<typename T>
	struct ExpressionEvaluation< ExpressionContainer<T> >
	{
		// Mechanism for a ExpressionContainer Object :
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType;

		__device__ inline static ReturnType evaluate(const ExpressionContainer<T>& container, const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k)
		{
			return ExpressionEvaluation<T>::evaluate(container.expr, l, p, i, j, k);
		}
	};

	template<typename T>
	struct ExpressionEvaluation< Accessor<T> >
	{
		// Mechanism for a Accessor Object :
		typedef T ReturnType;
		
		__device__ inline static ReturnType evaluate(const Accessor<T>& accessor, const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k)
		{
			return accessor.data(p);
		}
	};

	template<typename T>
	struct ExpressionEvaluation< Array<T> >
	{
		// Mechanism for a Array Object :
		typedef T ReturnType;
		
		__device__ inline static ReturnType evaluate(const Array<T>& accessor, const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k)
		{
			return accessor.data(p);
		}
	};

	template<class Op>
	struct ExpressionEvaluation< NullaryExpression<Op> >
	{
		// Mechanism for a NullaryExpression :
		typedef Op Operator;
		typedef typename Op::ReturnType ReturnType;
		
		__device__ inline static ReturnType evaluate(const NullaryExpression<Op>& nullaryExpression, const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k)
		{
			return Operator::apply(l, p, i, j, k);
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
		
		__device__ inline static ReturnType evaluate(const UnaryExpression<T, Op>& unaryExpression, const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k)
		{
			return Operator::apply(		ExpressionEvaluation<T>::evaluate(unaryExpression.a, l, p, i, j, k) );
		}
	};

	template< typename T, class Op >
	struct ExpressionEvaluation< TransformExpression<T, Op> >
	{
		// Mechanism for a TransformExpression :
		typedef Op Operator;
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType;
		
		__device__ inline static ReturnType evaluate(const TransformExpression<T, Op>& transfomExpression, const Layout& l, index_t p, index_t i, index_t j, index_t k)
		{
			Operator::apply(l, p, i, j, k); // Change in shape
			return ExpressionEvaluation<T>::evaluate(transfomExpression.a, l, p, i, j, k); // Use sub type with new coordinates
		}
	};

	template< typename T, class Op >
	struct ExpressionEvaluation< LayoutReinterpretationExpression<T, Op> >
	{
		// Mechanism for a LayoutReinterpretationExpression :
		typedef Op Operator;
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType;
		
		__device__ inline static ReturnType evaluate(const LayoutReinterpretationExpression<T, Op>& layoutReinterpretationExpression, const Layout& l, index_t p, index_t i, index_t j, index_t k)
		{
			Operator::apply(l, layoutReinterpretationExpression.layout, p, i, j, k); // Change in shape according to a new layout
			return ExpressionEvaluation<T>::evaluate(layoutReinterpretationExpression.a, layoutReinterpretationExpression.layout, p, i, j, k); // Use sub type with new coordinates AND new layout
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
		
		__device__ inline static ReturnType evaluate(const BinaryExpression<T1, T2, Op>& binaryExpression, const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k)
		{
			return Operator::apply(		ExpressionEvaluation<T1>::evaluate(binaryExpression.a, l, p, i, j, k), 	
							ExpressionEvaluation<T2>::evaluate(binaryExpression.b, l, p, i, j, k)
						);
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
		
		__device__ inline static ReturnType evaluate(const TernaryExpression<T1, T2, T3, Op>& ternaryExpression, const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k)
		{
			return Operator::apply(		ExpressionEvaluation<T1>::evaluate(ternaryExpression.a, l, p, i, j, k), 
							ExpressionEvaluation<T2>::evaluate(ternaryExpression.b, l, p, i, j, k),
							ExpressionEvaluation<T3>::evaluate(ternaryExpression.c, l, p, i, j, k)
						);
		}
	};

// Expression Container (transparent expression) :
	template<typename T>
	struct ExpressionContainer
	{
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType;

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
		typedef typename Op::ReturnType ReturnType; 

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
		typedef typename Op< typename ExpressionEvaluation<T>::ReturnType >::ReturnType ReturnType; 

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
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType; 

		T a;

		__host__ TransformExpression(const T& _a)
		 : a(_a)
		{ }

		__host__ __device__ TransformExpression(const TransformExpression<T, Op>& e)
		 : a(e.a)
		{ }
	};

// Transform Expression
	template<typename T, class Op >
	struct LayoutReinterpretationExpression
	{
		typedef typename ExpressionEvaluation<T>::ReturnType ReturnType; 

		T a;
		Layout layout;

		__host__ LayoutReinterpretationExpression(const T& _a, const Layout& l)
		 : a(_a), layout(l)
		{ }

		__host__ __device__ LayoutReinterpretationExpression(const LayoutReinterpretationExpression<T, Op>& e)
		 : a(e.a), layout(e.layout)
		{ }
	};

// Binary Operators
	template<typename T1, typename T2, template<typename,typename> class Op >
	struct BinaryExpression
	{
		typedef typename Op< 	typename ExpressionEvaluation<T1>::ReturnType, 
					typename ExpressionEvaluation<T2>::ReturnType 
					>::ReturnType ReturnType; 
		
		T1 a;
		T2 b;

		__host__ BinaryExpression(const T1& _a, const T2& _b)
		 : a(_a), b(_b)
		{ }

		__host__ __device__ BinaryExpression(const BinaryExpression<T1,T2,Op>& e)
		 : a(e.a), b(e.b)
		{ }
	};

// Ternary Operators :
	template<typename T1, typename T2, typename T3, template<typename, typename, typename> class Op >
	struct TernaryExpression
	{
		typedef typename Op<	typename ExpressionEvaluation<T1>::ReturnType, 
					typename ExpressionEvaluation<T2>::ReturnType,
					typename ExpressionEvaluation<T3>::ReturnType
					>::ReturnType ReturnType; 

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
	template<typename T, typename TExpr>
	__global__ void evaluateExpression(const Accessor<T> array, const TExpr expr)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;

		index_t i = array.getI();

		if(array.validRowIndex(i))
		{
			index_t j = array.getJ(),
				k = array.getK(),
				p = array.getIndex();

			ReturnType t = ExpressionEvaluation<TExpr>::evaluate(expr, array, p, i, j, k);

			T buffer;
			complexCopy(buffer, t);
			array.data() = buffer;
		}
	}

	template<typename T, typename TExprMask, typename TExpr>
	__global__ void evaluateExpressionWithMask(const Accessor<T> array, const TExprMask exprMask, const TExpr expr)
	{
		typedef typename ExpressionEvaluation<TExprMask>::ReturnType MaskType;
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;

		index_t i = array.getI();

		if(array.validRowIndex(i))
		{
			index_t j = array.getJ(),
				k = array.getK(),
				p = array.getIndex();

			MaskType test = ExpressionEvaluation<TExprMask>::evaluate(exprMask, array, p, i, j, k);

			if(test)
			{
				ReturnType t = ExpressionEvaluation<TExpr>::evaluate(expr, array, p, i, j, k);
				
				T buffer;
				complexCopy(buffer, t);
				array.data() = buffer;
			}
		}
	}

} // Namespace Kartet

#endif

