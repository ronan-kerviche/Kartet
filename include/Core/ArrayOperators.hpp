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

#ifndef __KARTET_ARRAY_OPERATORS__
#define __KARTET_ARRAY_OPERATORS__

namespace Kartet
{
// Assignement operators :
	template<typename T>
	template<typename TExpr>
	Accessor<T>& Accessor<T>::assign(const TExpr& expr, cudaStream_t stream)
	{
		// Make sure we are not computing complex numbers to store in a real array :
		STATIC_ASSERT( !(TypeInfo< typename ExpressionEvaluation<TExpr>::ReturnType >::isComplex && !TypeInfo<T>::isComplex) )
		evaluateExpression COMPUTE_LAYOUT(*this) (*this, expr);
		return *this;
	}

	template<typename T>
	Accessor<T>& Accessor<T>::assign(const Accessor<T>& a, cudaStream_t stream)
	{
		evaluateExpression COMPUTE_LAYOUT(*this) (*this, a);
		return *this;
	}

	template<typename T>
	template<typename TExpr>
	Accessor<T>& Accessor<T>::operator=(const TExpr& expr)
	{		
		return assign(expr);
	}

	template<typename T>
	Accessor<T>& Accessor<T>::operator=(const Accessor<T>& a)
	{
		return assign(a);
	}

// Masked assignements : 
	template<typename T>
	template<typename TExprMask, typename TExpr>
	Accessor<T>& Accessor<T>::maskedAssignment(const TExprMask& exprMask, const TExpr& expr)
	{
		// Make sure we are not computing complex numbers to store in a real array :
		STATIC_ASSERT( !(TypeInfo< typename ExpressionEvaluation<TExpr>::ReturnType >::isComplex && !TypeInfo<T>::isComplex) )
		evaluateExpressionWithMask COMPUTE_LAYOUT(*this) (*this, exprMask, expr);
		return *this;
	}

// Standard operators tool :
	/*
		Given types cannot be expressions.
		To change arity, set type of ith argument to void.
	*/
	template<typename T1, typename T2, typename T3, bool preferComplexOutput, bool preferRealOutput, bool inputMustBeComplex, bool inputMustBeReal, typename forceReturnType, typename enforceT1, typename enforceT2, typename enforceT3>
	struct StandardOperatorTypeToolbox
	{
		private :
			// Test for correct order of arguments :
			static const StaticAssert< 	(SameTypes<T3,void>::test) ? 
							( // No third argument
								!(SameTypes<T1,void>::test) // whatever 2nd arg is, there must be at least one.
							)
							:
							(
								(!SameTypes<T1,void>::test) && (!SameTypes<T2,void>::test)
							)
						> 										test1;

			// Test for conflict free requests :
			static const StaticAssert<	!(preferComplexOutput && preferRealOutput) > 				test2;
			static const StaticAssert<	!(inputMustBeReal && inputMustBeComplex) > 				test3;

			// Test for type enforcing :
			static const StaticAssert<	SameTypes<enforceT1,void>::test || SameTypes<enforceT1,T1>::test>	test4;
			static const StaticAssert<	SameTypes<enforceT2,void>::test || SameTypes<enforceT2,T2>::test>	test5;
			static const StaticAssert<	SameTypes<enforceT3,void>::test || SameTypes<enforceT3,T3>::test>	test6;

			// Test validity :
			static const StaticAssert<	!(inputMustBeReal && TypeInfo<T1>::isComplex) >				test_ir1;
			static const StaticAssert<	!(inputMustBeReal && TypeInfo<T2>::isComplex) >				test_ir2;
			static const StaticAssert<	!(inputMustBeReal && TypeInfo<T3>::isComplex) >				test_ir3;

			static const StaticAssert<	!(inputMustBeComplex && !TypeInfo<T1>::isComplex) >			test_ic1;
			static const StaticAssert<	!(inputMustBeComplex && !TypeInfo<T2>::isComplex) >			test_ic2;
			static const StaticAssert<	!(inputMustBeComplex && !TypeInfo<T3>::isComplex) >			test_ic3;
			
			// Test for dimensionality (value, pointers, etc.) :

			// Choose best output type :
			typedef typename ResultingTypeOf2<T1,T2>::Type 								Type1;
			typedef typename ResultingTypeOf2<Type1,T3>::Type 							Type2;

			// Force output type according to request :
			typedef typename MetaIf<preferRealOutput, typename TypeInfo<Type2>::ComplexType, Type2 >::TValue	Type3;
			typedef typename MetaIf<preferComplexOutput, typename TypeInfo<Type3>::BaseType, Type3 >::TValue	Type4;
			StaticAssert<!preferComplexOutput> C1;

			typedef typename MetaIf<SameTypes<forceReturnType,void>::test, Type4, forceReturnType >::TValue	Type5;
		public :
			static const int arity = 	(SameTypes<T1,void>::test) ? 0 : (
							(SameTypes<T2,void>::test) ? 1 : (
							(SameTypes<T3,void>::test) ? 2 : 3));

			typedef Type5 ReturnType;
	};

// Nullary Standard Maths Operators :
	#define STANDARD_NULLARY_OPERATOR_OBJECT(objName, operation, outputType) \
		struct objName \
		{ \
			typedef outputType ReturnType; \
			\
			__device__ inline static ReturnType apply(const Layout& l, const index_t& p, const index_t& i, const index_t& j, const index_t& k) \
			{ \
				return operation ; \
			} \
		};

	// Keep the inline here to avoid redefinition errors at client compile time.
	#define STANDARD_NULLARY_FUNCTION_INTERFACE(funcName, opName) \
		inline ExpressionContainer< NullaryExpression< opName > > funcName (void) \
		{ \
			return ExpressionContainer< NullaryExpression< opName > >( NullaryExpression< opName >() ); \
		}

	#define STANDARD_NULLARY_OPERATOR_DEFINITION( objName, funcName, outputType, operation) \
		STANDARD_NULLARY_OPERATOR_OBJECT( objName, operation, outputType) \
		STANDARD_NULLARY_FUNCTION_INTERFACE( funcName, objName)

// Unary Standard Maths Operators :
	#define STANDARD_UNARY_OPERATOR_OBJECT(objName, operation) \
		template<typename T> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T, void, void, false, false, false, false, void, void, void, void>::ReturnType ReturnType; \
			\
			__device__ inline static ReturnType apply(const T& a) \
			{ \
				return (operation) ; \
			} \
		};

	#define C2R_UNARY_OPERATOR_OBJECT(objName, operation) \
		template<typename T> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T, void, void, false, true, false, false, void, void, void, void>::ReturnType ReturnType; \
			\
			__device__ inline static ReturnType apply(const T& a) \
			{ \
				return (operation) ; \
			} \
		};

	#define R2C_UNARY_OPERATOR_OBJECT(objName, operation) \
		template<typename T> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T, void, void, true, false, false, true, void, void, void, void>::ReturnType ReturnType; \
			\
			__device__ inline static ReturnType apply(const T& a) \
			{ \
				return (operation) ; \
			} \
		};

	#define CAST_UNARY_OPERATOR(objName, operation) \
		template<typename TOut> \
		struct objName \
		{ \
			template<typename T> \
			struct Sub \
			{ \
				typedef typename StandardOperatorTypeToolbox<TOut, void, void, false, false, false, false, void, void, void, void>::ReturnType ReturnType; \
				\
				__device__ inline static ReturnType apply(const T& a) \
				{ \
					return static_cast<TOut>(operation) ; \
				} \
			}; \
		};

	#define STANDARD_UNARY_FUNCTION_INTERFACE(funcName, opName) \
		template<typename T> \
		ExpressionContainer< UnaryExpression< Accessor<T>, opName > > funcName (const Accessor<T>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< Accessor<T>, opName > >( UnaryExpression< Accessor<T>, opName >(a) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< UnaryExpression< Accessor<T>, opName > > funcName (const Array<T>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< Accessor<T>, opName > >( UnaryExpression< Accessor<T>, opName >(a) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< UnaryExpression< ExpressionContainer<T>, opName > > funcName (const ExpressionContainer<T>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< ExpressionContainer<T>, opName > >( UnaryExpression< ExpressionContainer<T>, opName >(a) ); \
		}

	#define CAST_UNARY_FUNCTION_INTERFACE(funcName, opName) \
		template<typename TOut, typename T> \
		ExpressionContainer< UnaryExpression< Accessor<T>, opName <TOut>::template Sub > > funcName (const Accessor<T>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< Accessor<T>, opName <TOut>::template Sub > >( UnaryExpression< Accessor<T>, opName <TOut>::template Sub >(a) ); \
		} \
		\
		template<typename TOut, typename T> \
		ExpressionContainer< UnaryExpression< Accessor<T>, opName <TOut>::template Sub > > funcName (const Array<T>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< Accessor<T>, opName <TOut>::template Sub > >( UnaryExpression< Accessor<T>, opName <TOut>::template Sub >(a) ); \
		} \
		\
		template<typename TOut, typename T> \
		ExpressionContainer< UnaryExpression< ExpressionContainer<T>, opName <TOut>::template Sub > > funcName (const ExpressionContainer<T>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< ExpressionContainer<T>, opName <TOut>::template Sub > >( UnaryExpression< ExpressionContainer<T>, opName <TOut>::template Sub >(a) ); \
		}

	#define STANDARD_UNARY_OPERATOR_DEFINITION(objName, funcName, operation) \
		STANDARD_UNARY_OPERATOR_OBJECT(objName, operation) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	#define C2R_UNARY_OPERATOR_DEFINITION(objName, funcName, operation) \
		C2R_UNARY_OPERATOR_OBJECT(objName, operation) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	#define R2C_UNARY_OPERATOR_DEFINITION(objName, funcName, operation) \
		R2C_UNARY_OPERATOR_OBJECT(objName, operation) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	#define CAST_UNARY_OPERATOR_DEFINITION(objName, funcName, operation) \
		CAST_UNARY_OPERATOR(objName, operation) \
		CAST_UNARY_FUNCTION_INTERFACE(funcName, objName)

// Standard Transform Operators :
	#define STANDARD_TRANSFORM_OPERATOR_OBJECT(objName, operations) \
		struct objName \
		{ \
			typedef void ReturnType; \
			\
			__device__ inline static ReturnType apply(const Layout& l, index_t& p, index_t& i, index_t& j, index_t& k) \
			{ \
				operations \
			} \
		};

	#define STANDARD_TRANSFORM_FUNCTION_INTERFACE(funcName, opName) \
		template<typename T> \
		ExpressionContainer< TransformExpression< Accessor<T>, opName > > funcName (const Accessor<T>& a) \
		{ \
			return ExpressionContainer< TransformExpression< Accessor<T>, opName > >( TransformExpression< Accessor<T>, opName >(a) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< TransformExpression< Accessor<T>, opName > > funcName (const Array<T>& a) \
		{ \
			return ExpressionContainer< TransformExpression< Accessor<T>, opName > >( TransformExpression< Accessor<T>, opName >(a) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< TransformExpression< ExpressionContainer<T>, opName > > funcName (const ExpressionContainer<T>& a) \
		{ \
			return ExpressionContainer< TransformExpression< ExpressionContainer<T>, opName > >( TransformExpression< ExpressionContainer<T>, opName >(a) ); \
		} \

	#define STANDARD_TRANSFORM_OPERATOR_DEFINITION( objName, funcName, operation) \
		STANDARD_TRANSFORM_OPERATOR_OBJECT( objName, operation) \
		STANDARD_TRANSFORM_FUNCTION_INTERFACE( funcName, objName)

// Standard Layout Reinterpretation Operators :
	#define STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_OBJECT(objName, operations) \
		struct objName \
		{ \
			typedef void ReturnType; \
			\
			__device__ inline static ReturnType apply(const Layout& l, const Layout& lnew, index_t& p, index_t& i, index_t& j, index_t& k) \
			{ \
				operations \
			} \
		};

	#define STANDARD_LAYOUT_REINTERPRETATION_FUNCTION_INTERFACE(funcName, opName) \
		template<typename T> \
		ExpressionContainer< LayoutReinterpretationExpression< Accessor<T>, opName > > funcName (const Accessor<T>& a, const Layout& l) \
		{ \
			return ExpressionContainer< LayoutReinterpretationExpression< Accessor<T>, opName > >( LayoutReinterpretationExpression< Accessor<T>, opName >(a, l) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< LayoutReinterpretationExpression< Accessor<T>, opName > > funcName (const Array<T>& a, const Layout& l) \
		{ \
			return ExpressionContainer< LayoutReinterpretationExpression< Accessor<T>, opName > >( LayoutReinterpretationExpression< Accessor<T>, opName >(a, l) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< LayoutReinterpretationExpression< ExpressionContainer<T>, opName > > funcName (const ExpressionContainer<T>& a, const Layout& l) \
		{ \
			return ExpressionContainer< LayoutReinterpretationExpression< ExpressionContainer<T>, opName > >( LayoutReinterpretationExpression< ExpressionContainer<T>, opName >(a, l) ); \
		} \

	#define STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( objName, funcName, operation) \
		STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_OBJECT( objName, operation) \
		STANDARD_LAYOUT_REINTERPRETATION_FUNCTION_INTERFACE( funcName, objName)

// Binary Operators Tools :
	#define STANDARD_BINARY_OPERATOR_OBJECT(objName, operation) \
		template<typename T1, typename T2> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T1, T2, void, false, false, false, false, void, void, void, void>::ReturnType ReturnType; \
			 \
			__device__ inline static ReturnType apply(T1 a, T2 b) \
			{ \
				return (operation) ; \
			} \
		};

	#define COMPARISON_BINARY_OPERATOR_OBJECT(objName, operation) \
		template<typename T1, typename T2> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T1, T2, void, false, false, false, false, bool, void, void, void>::ReturnType ReturnType; \
			 \
			__device__ inline static ReturnType apply(T1 a, T2 b) \
			{ \
				return (operation) ; \
			} \
		};

	#define R2C_BINARY_OPERATOR_OBJECT(objName, operation) \
		template<typename T1, typename T2> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T1, T2, void, true, false, false, true, void, void, void, void>::ReturnType ReturnType; \
			 \
			__device__ inline static ReturnType apply(T1 a, T2 b) \
			{ \
				return (operation) ; \
			} \
		};

	#define STANDARD_BINARY_FUNCTION_INTERFACE(funcName, opName) \
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1>, T2, opName > > funcName (const Accessor<T1>& a, const T2& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1>, T2, opName > >( BinaryExpression< Accessor<T1>, T2, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1>, Accessor<T2>, opName > > funcName (const Accessor<T1>& a, const Array<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1>, Accessor<T2>, opName > >( BinaryExpression< Accessor<T1>, Accessor<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< T1, Accessor<T2>, opName > > funcName (const T1& a, const Accessor<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< T1, Accessor<T2>, opName > >( BinaryExpression< T1, Accessor<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1>, Accessor<T2>, opName > > funcName (const Array<T1>& a, const Accessor<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1>, Accessor<T2>, opName > >( BinaryExpression< Accessor<T1>, Accessor<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1>, Accessor<T2>, opName > > funcName (const Accessor<T1>& a, const Accessor<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1>, Accessor<T2>, opName > >( BinaryExpression< Accessor<T1>, Accessor<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1>, Accessor<T2>, opName > > funcName (const Array<T1>& a, const Array<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1>, Accessor<T2>, opName > >( BinaryExpression< Accessor<T1>, Accessor<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, T2, opName > > funcName (const ExpressionContainer<T1>& a, const T2& b) \
		{ \
			return ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, T2, opName > >( BinaryExpression< ExpressionContainer<T1>, T2, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, Accessor<T2>, opName > > funcName (const ExpressionContainer<T1>& a, const Array<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, Accessor<T2>, opName > >( BinaryExpression< ExpressionContainer<T1>, Accessor<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< T1, ExpressionContainer<T2>, opName > > funcName (const T1& a, const ExpressionContainer<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< T1, ExpressionContainer<T2>, opName > >( BinaryExpression< T1, ExpressionContainer<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1>, ExpressionContainer<T2>, opName > > funcName (const Array<T1>& a, const ExpressionContainer<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1>, ExpressionContainer<T2>, opName > >( BinaryExpression< Accessor<T1>, ExpressionContainer<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, Accessor<T2>, opName > > funcName (const ExpressionContainer<T1>& a, const Accessor<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, Accessor<T2>, opName > >( BinaryExpression< ExpressionContainer<T1>, Accessor<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1>, ExpressionContainer<T2>, opName > > funcName (const Accessor<T1>& a, const ExpressionContainer<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1>, ExpressionContainer<T2>, opName > >( BinaryExpression< Accessor<T1>, ExpressionContainer<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, ExpressionContainer<T2>, opName > > funcName (const ExpressionContainer<T1>& a, const ExpressionContainer<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, ExpressionContainer<T2>, opName > >( BinaryExpression< ExpressionContainer<T1>, ExpressionContainer<T2>, opName >(a, b) ); \
		}

	#define STANDARD_BINARY_OPERATOR_DEFINITION( objName, funcName, operation) \
		STANDARD_BINARY_OPERATOR_OBJECT( objName, operation) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

	#define COMPARISON_BINARY_OPERATOR_DEFINITION( objName, funcName, operation) \
		COMPARISON_BINARY_OPERATOR_OBJECT( objName, operation) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

	#define R2C_BINARY_OPERATOR_DEFINITION( objName, funcName, operation) \
		R2C_BINARY_OPERATOR_OBJECT( objName, operation) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

// Standard operators : 
	// Unary : 
		STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_Minus,		operator-,	-a)
		STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_Not,		operator!,	!a)
		STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_BinaryCompl,	operator~,	~a)

	// Binary :
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Plus, 		operator+, 	a+b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Minus, 		operator-, 	a-b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Times, 		operator*, 	a*b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Divide, 		operator/, 	a/b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Modulo, 		operator%, 	a%b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_BinaryAnd, 	operator&, 	a&b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_BinaryOr, 	operator|, 	a|b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_And, 		operator&&, 	a && b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Or, 		operator||, 	a || b )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Xor, 		operator^, 	a^b )

		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_Equal, 		operator==, 	a==b )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_Different, 	operator!=, 	a!=b )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_GreaterStrict, 	operator>, 	a>b )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_GreaterOrEqual, 	operator>=, 	a>=b )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_SmallerStrict, 	operator<, 	a<b )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_SmallerOrEqual, 	operator<=, 	a<=b )

} // Namespace Kartet

// Unary and Binary functions :
	#include "Core/ArrayFunctions.hpp"
	
// Clean the mess :
	// #undef ?

#endif

