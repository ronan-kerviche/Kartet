/* ************************************************************************************************************* */
/*                                                                                                               */
/*     Kartet                                                                                                    */
/*     A Simple C++ Array Library for CUDA                                                                       */
/*                                                                                                               */
/*     LICENSE : The MIT License                                                                                 */
/*     Copyright (c) 2015-2017 Ronan Kerviche                                                                    */
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
	\file    ArrayOperators.hpp
	\brief   Array classes operators implementations and macros.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_ARRAY_OPERATORS__
#define __KARTET_ARRAY_OPERATORS__

namespace Kartet
{
// Assignement operators :
	/**
	\brief Evaluate an expression and store the result.
	\param expr Expression.
	\param stream Stream to be used for the computation (if the location is set to Kartet::DeviceSide).

	See \ref OperatorsGroup and \ref FunctionsGroup.

	\return Reference to this.
	**/
	template<typename T, Location l>
	template<typename TExpr>
	Accessor<T,l>& Accessor<T,l>::assign(const TExpr& expr, cudaStream_t stream)
	{
		#ifndef __CUDACC__
			UNUSED_PARAMETER(stream)
		#endif
		// Make sure we are not computing complex numbers to store in a real array :
		STATIC_ASSERT_VERBOSE(!Traits<typename ExpressionEvaluation<TExpr>::ReturnType>::isComplex || Traits<T>::isComplex, RHS_IS_COMPLEX_BUT_LHS_IS_NOT)
		// Make sure the expression is on the same side :
		STATIC_ASSERT_VERBOSE(ExpressionEvaluation<TExpr>::location==l || ExpressionEvaluation<TExpr>::location==Kartet::AnySide, LHS_AND_RHS_HAVE_INCOMPATIBLE_LOCATIONS)
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				if(sizeof(typename ExpressionEvaluation<TExpr>::ReturnType)<=sizeof(float))
				{
					// With repetition (better for single precision) :
					dim3 blockSize, numBlocks, blockRepetition;
					computeLayout(blockSize, numBlocks, blockRepetition);
					evaluateExpressionRepeat <<<numBlocks, blockSize, 0, stream>>> (*this, expr, blockRepetition);
				}
				else // Without repetition (better for double precision) :
					evaluateExpression COMPUTE_LAYOUT_STREAM(*this, stream) (*this, expr);
				cudaError_t err = cudaGetLastError();
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);
			#else
				throw NotSupported;
			#endif
		}
		else
			evaluateExpressionOverLayout(*this, expr);
		return *this;
	}

	/**
	\brief Copy data.
	\param a Other accessor.
	\param stream Stream to be used for the copy (if the location is set to Kartet::DeviceSide).
	\return Reference to this.
	**/
	template<typename T, Location l>
	Accessor<T,l>& Accessor<T,l>::assign(const Accessor<T,l>& a, cudaStream_t stream)
	{
		#ifndef __CUDACC__
			UNUSED_PARAMETER(stream)
		#endif
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				if(sizeof(typename ExpressionEvaluation<T>::ReturnType)<=sizeof(float))
				{
					// With repetition (better for single precision) :
					dim3 blockSize, numBlocks, blockRepetition;
					computeLayout(blockSize, numBlocks, blockRepetition);
					evaluateExpressionRepeat <<<numBlocks, blockSize, 0, stream>>> (*this, a, blockRepetition);
				}
				else // Without repetition (better for double precision) :
					evaluateExpression COMPUTE_LAYOUT_STREAM(*this, stream) (*this, a);
				cudaError_t err = cudaGetLastError();
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);
			#else
				throw NotSupported;
			#endif
		}
		else
			evaluateExpressionOverLayout(*this, a);
		return *this;
	}

	#ifdef __CUDACC__
		template<typename T>
		struct MemCpyDualToolBox
		{			
			const cudaMemcpyKind direction;
			cudaStream_t stream;

			__host__ MemCpyDualToolBox(const cudaMemcpyKind _d, cudaStream_t _stream = NULL)
			 :	direction(_d),
				stream(_stream)
			{ }

			__host__ MemCpyDualToolBox(const MemCpyDualToolBox& m)
			 :	direction(m.direction),
				stream(m.stream)
			{ }
			
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, T* dst, T* src, size_t offsetDst, size_t offsetSrc, index_t i, index_t j, index_t k) const
			{
				UNUSED_PARAMETER(mainLayout)
				UNUSED_PARAMETER(i)
				UNUSED_PARAMETER(j)
				UNUSED_PARAMETER(k)

				cudaError_t err = cudaSuccess;
				if(stream!=NULL)
					err = cudaMemcpy((dst + offsetDst), (src + offsetSrc), currentAccessLayout.numElements()*sizeof(T), direction);
				else
					err = cudaMemcpyAsync((dst + offsetDst), (src + offsetSrc), currentAccessLayout.numElements()*sizeof(T), direction, stream);

				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);
			}
		};
	#endif

	/**
	\brief Copy data.
	\param a Other accessor.
	\param stream Stream to be used for the copy (if one of the location is set to Kartet::DeviceSide).
	\return Reference to this.
	**/
	template<typename T, Location l>
	template<Location l2>
	Accessor<T,l>& Accessor<T,l>::assign(const Accessor<T,l2>& a, cudaStream_t stream)
	{
		STATIC_ASSERT_VERBOSE(l!=l2, LHS_AND_RHS_HAVE_INCOMPATIBLE_LOCATIONS)
		#ifdef __CUDACC__
			MemCpyDualToolBox<T> op((l==DeviceSide) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream);
			dualScan(*this, dataPtr(), a, a.dataPtr(), op);
			return *this;
		#else
			// This is/should never (be) happening if device-side is not defined.
			UNUSED_PARAMETER(a)
			UNUSED_PARAMETER(stream)
			throw NotSupported;
		#endif
	}

	/**
	\brief Copy data.
	\param a Other array.
	\param stream Stream to be used for the copy (if the location is set to Kartet::DeviceSide).
	\return Reference to this.
	**/
	template<typename T, Location l>
	Accessor<T,l>& Accessor<T,l>::assign(const Array<T,l>& a, cudaStream_t stream)
	{
		return assign(a.accessor(), stream);
	}

	/**
	\brief Copy data.
	\param a Other array.
	\param stream Stream to be used for the copy (if one of the location is set to Kartet::DeviceSide).
	\return Reference to this.
	**/
	template<typename T, Location l>
	template<Location l2>
	Accessor<T,l>& Accessor<T,l>::assign(const Array<T,l2>& a, cudaStream_t stream)
	{
		return assign(a.accessor(), stream);
	}

	/**
	\brief Assignment operator.
	\param expr Expression.
	
	See \ref OperatorsGroup and \ref FunctionsGroup.

	\return Reference to this.
	**/
	template<typename T, Location l>
	template<typename TExpr>
	Accessor<T,l>& Accessor<T,l>::operator=(const TExpr& expr)
	{
		return assign(expr);
	}

	/**
	\brief Assignment operator.
	\param a Accessor.
	\return Reference to this.
	**/
	template<typename T, Location l>
	Accessor<T,l>& Accessor<T,l>::operator=(const Accessor<T,l>& a)
	{
		return assign(a);
	}

	/**
	\brief Assignment operator.
	\param a Accessor.
	\return Reference to this.
	**/
	template<typename T, Location l>
	template<Location l2>
	Accessor<T,l>& Accessor<T,l>::operator=(const Accessor<T,l2>& a)
	{
		return assign(a);
	}

	template<typename T, Location l>
	Accessor<T,l>& Accessor<T,l>::operator=(const Array<T,l>& a)
	{
		return assign(a);
	}

	/**
	\brief Assignment operator.
	\param a Array.
	\return Reference to this.
	**/
	template<typename T, Location l>
	template<Location l2>
	Accessor<T,l>& Accessor<T,l>::operator=(const Array<T,l2>& a)
	{
		return assign(a);
	}

	#define ACCESSOR_COMPOUND_ASSIGNMENT( operatorName, ... ) \
		template<typename T, Location l> \
		template<typename TExpr> \
		Accessor<T,l>& Accessor<T,l>:: operatorName (const TExpr& expr) \
		{ \
			(*this) = (*this) __VA_ARGS__ expr; \
			return (*this); \
		} \
		 \
		template<typename T, Location l> \
		Accessor<T,l>& Accessor<T,l>:: operatorName (const Accessor<T,l>& a) \
		{ \
			(*this) = (*this) __VA_ARGS__ a; \
			return (*this); \
		} \
		 \
		template<typename T, Location l> \
		template<Location l2> \
		Accessor<T,l>& Accessor<T,l>:: operatorName (const Accessor<T,l2>& a) \
		{ \
			(*this) = (*this) __VA_ARGS__ a; \
			return (*this); \
		} \
		 \
		template<typename T, Location l> \
		Accessor<T,l>& Accessor<T,l>:: operatorName (const Array<T,l>& a) \
		{ \
			(*this) = (*this) __VA_ARGS__ a; \
			return (*this); \
		} \
		 \
		template<typename T, Location l> \
		template<Location l2> \
		Accessor<T,l>& Accessor<T,l>:: operatorName (const Array<T,l2>& a) \
		{ \
			(*this) = (*this) __VA_ARGS__ a; \
			return (*this); \
		}

		ACCESSOR_COMPOUND_ASSIGNMENT( operator+=,  + )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator-=,  - )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator*=,  * )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator/=,  / )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator%=,  % )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator&=,  & )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator|=,  | )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator^=,  ^ )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator<<=, << )
		ACCESSOR_COMPOUND_ASSIGNMENT( operator>>=, >> )

	#undef ACCESSOR_COMPOUND_ASSIGNMENT

	/**
	\brief Assignment operator.
	\param expr Expression.

	See \ref OperatorsGroup and \ref FunctionsGroup.

	\return Reference to this.
	**/
	template<typename T, Location l>
	template<typename TExpr>
	Array<T,l>& Array<T,l>::operator=(const TExpr& expr)
	{
		assign(expr);
		return (*this);
	}

	/**
	\brief Assignment operator.
	\param a Accessor.
	\return Reference to this.
	**/
	template<typename T, Location l>
	Array<T,l>& Array<T,l>::operator=(const Accessor<T,l>& a)
	{
		assign(a);
		return (*this);
	}

	/**
	\brief Assignment operator.
	\param a Accessor.
	\return Reference to this.
	**/
	template<typename T, Location l>
	template<Location l2>
	Array<T,l>& Array<T,l>::operator=(const Accessor<T,l2>& a)
	{
		assign(a);
		return (*this);
	}

	/**
	\brief Assignment operator.
	\param a Array.
	\return Reference to this.
	**/
	template<typename T, Location l>
	Array<T,l>& Array<T,l>::operator=(const Array<T,l>& a)
	{
		assign(a);
		return (*this);
	}

	/**
	\brief Assignment operator.
	\param a Array.
	\return Reference to this.
	**/
	template<typename T, Location l>
	template<Location l2>
	Array<T,l>& Array<T,l>::operator=(const Array<T,l2>& a)
	{
		assign(a);
		return (*this);
	}

	#define ARRAY_COMPOUND_ASSIGNMENT( operatorName, ... ) \
		template<typename T, Location l> \
		template<typename TExpr> \
		Array<T,l>& Array<T,l>:: operatorName (const TExpr& expr) \
		{ \
			(*this) = (*this) __VA_ARGS__ expr; \
			return (*this); \
		} \
		 \
		template<typename T, Location l> \
		Array<T,l>& Array<T,l>:: operatorName (const Accessor<T,l>& a) \
		{ \
			(*this) = (*this) __VA_ARGS__ a; \
			return (*this); \
		} \
		 \
		template<typename T, Location l> \
		template<Location l2> \
		Array<T,l>& Array<T,l>:: operatorName (const Accessor<T,l2>& a) \
		{ \
			(*this) = (*this) __VA_ARGS__ a; \
			return (*this); \
		} \
		 \
		template<typename T, Location l> \
		Array<T,l>& Array<T,l>:: operatorName (const Array<T,l>& a) \
		{ \
			(*this) = (*this) __VA_ARGS__ a; \
			return (*this); \
		} \
		 \
		template<typename T, Location l> \
		template<Location l2> \
		Array<T,l>& Array<T,l>:: operatorName (const Array<T,l2>& a) \
		{ \
			(*this) = (*this) __VA_ARGS__ a; \
			return (*this); \
		}

		ARRAY_COMPOUND_ASSIGNMENT( operator+=,  + )
		ARRAY_COMPOUND_ASSIGNMENT( operator-=,  - )
		ARRAY_COMPOUND_ASSIGNMENT( operator*=,  * )
		ARRAY_COMPOUND_ASSIGNMENT( operator/=,  / )
		ARRAY_COMPOUND_ASSIGNMENT( operator%=,  % )
		ARRAY_COMPOUND_ASSIGNMENT( operator&=,  & )
		ARRAY_COMPOUND_ASSIGNMENT( operator|=,  | )
		ARRAY_COMPOUND_ASSIGNMENT( operator^=,  ^ )
		ARRAY_COMPOUND_ASSIGNMENT( operator<<=, << )
		ARRAY_COMPOUND_ASSIGNMENT( operator>>=, >> )

	#undef ARRAY_COMPOUND_ASSIGNMENT

// Masked assignements : 
	template<typename T, Location l>
	template<typename TExprMask, typename TExpr>
	Accessor<T,l>& Accessor<T,l>::maskedAssignment(const TExprMask& exprMask, const TExpr& expr, cudaStream_t stream)
	{
		#ifndef __CUDACC__
			UNUSED_PARAMETER(stream)
		#endif

		// Make sure we are not computing complex numbers to store in a real array :
		STATIC_ASSERT_VERBOSE(!Traits<typename ExpressionEvaluation<TExpr>::ReturnType>::isComplex || Traits<T>::isComplex, RHS_IS_COMPLEX_BUT_LHS_IS_NOT)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				if(sizeof(typename ExpressionEvaluation<TExprMask>::ReturnType)<=sizeof(float) && sizeof(typename ExpressionEvaluation<TExpr>::ReturnType)<=sizeof(float))
				{
					// With repetition (better for single precision) :
					dim3 blockSize, numBlocks, blockRepetition;
					computeLayout(blockSize, numBlocks, blockRepetition);
					evaluateExpressionWithMaskRepeat <<<numBlocks, blockSize, 0, stream>>> (*this, exprMask, expr, blockRepetition);
				}
				else // Without repetition (better for double precision) :
					evaluateExpressionWithMask COMPUTE_LAYOUT_STREAM(*this, stream) (*this, exprMask, expr);

				cudaError_t err = cudaGetLastError();
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);
			#else
				throw NotSupported;
			#endif
		}
		else
			evaluateExpressionWithMaskOverLayout(*this, exprMask, expr);

		return *this;
	}

// Standard operators tool :
	/*
		Given types cannot be expressions.
		To change arity, set the type of the i'th argument to void.
	*/
	template<typename T1, typename T2, typename T3, bool preferComplexOutput, bool preferRealOutput, bool inputMustBeComplex, bool inputMustBeReal, typename forceReturnType, typename enforceT1, typename enforceT2, typename enforceT3>
	struct StandardOperatorTypeToolbox
	{
		private :
			// Test for correct order of arguments :
			STATIC_ASSERT_VERBOSE( 	(IsSame<T3,void>::value) ? // No third argument :
								!(IsSame<T1,void>::value) // whatever 2nd arg is, there must be at least one.
							:	(!IsSame<T1,void>::value) && (!IsSame<T2,void>::value), INVALID_ARGUMENTS_LAYOUT)

			// Test for conflict free requests :
			STATIC_ASSERT_VERBOSE( !(preferComplexOutput && preferRealOutput), ARGUMENTS_CONFLICT)
			STATIC_ASSERT_VERBOSE( !(inputMustBeReal && inputMustBeComplex), ARGUMENTS_CONFLICT)

			// Test for type enforcing :
			STATIC_ASSERT_VERBOSE( (IsSame<enforceT1,void>::value || IsSame<enforceT1,T1>::value), TYPE_NOT_SUPPORTED)
			STATIC_ASSERT_VERBOSE( (IsSame<enforceT2,void>::value || IsSame<enforceT2,T2>::value), TYPE_NOT_SUPPORTED)
			STATIC_ASSERT_VERBOSE( (IsSame<enforceT3,void>::value || IsSame<enforceT3,T3>::value), TYPE_NOT_SUPPORTED)

			// Test validity :
			STATIC_ASSERT_VERBOSE( !(inputMustBeReal && Traits<T1>::isComplex), TYPE_MUST_BE_REAL)
			STATIC_ASSERT_VERBOSE( !(inputMustBeReal && Traits<T2>::isComplex), TYPE_MUST_BE_REAL)
			STATIC_ASSERT_VERBOSE( !(inputMustBeReal && Traits<T3>::isComplex), TYPE_MUST_BE_REAL)
			STATIC_ASSERT_VERBOSE( !(inputMustBeComplex && !Traits<T1>::isComplex), TYPE_MUST_BE_COMPLEX)
			STATIC_ASSERT_VERBOSE( !(inputMustBeComplex && !Traits<T2>::isComplex), TYPE_MUST_BE_COMPLEX)
			STATIC_ASSERT_VERBOSE( !(inputMustBeComplex && !Traits<T3>::isComplex), TYPE_MUST_BE_COMPLEX)
			
			// Choose best output type :
			typedef typename ResultingType<T1,T2>::Type								Type1;
			typedef typename ResultingType<Type1,T3>::Type								Type2;

			// Force output type according to request :
			typedef typename StaticIf<preferRealOutput, typename Traits<Type2>::BaseType, Type2 >::Type		Type3;
			typedef typename StaticIf<preferComplexOutput, Complex<Type3>, Type3 >::Type				Type4;

			typedef typename StaticIf<IsSame<forceReturnType,void>::value, Type4, forceReturnType >::Type		Type5;
		public :
			static const int arity = 	(IsSame<T1,void>::value) ? 0 : (
							(IsSame<T2,void>::value) ? 1 : (
							(IsSame<T3,void>::value) ? 2 : 3));

			typedef Type5 ReturnType;
	};

} // Namespace Kartet

// Nullary Standard Maths Operators :
	#define STANDARD_NULLARY_OPERATOR_OBJECT(objName, outputType, ...) \
	namespace Kartet \
	{ \
		struct objName \
		{ \
			typedef outputType ReturnType; \
			\
			__host__ __device__ inline static ReturnType apply(const Layout& l, const index_t& i, const index_t& j, const index_t& k) \
			{ \
				UNUSED_PARAMETER(l) \
				UNUSED_PARAMETER(i) \
				UNUSED_PARAMETER(j) \
				UNUSED_PARAMETER(k) \
				__VA_ARGS__ \
			} \
		}; \
	}

	// Keep the inline here to avoid redefinition errors at client compile time.
	#define STANDARD_NULLARY_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		inline ExpressionContainer< NullaryExpression< opName > > funcName (void) \
		{ \
			return ExpressionContainer< NullaryExpression< opName > >( NullaryExpression< opName >() ); \
		} \
	}

	#define STANDARD_NULLARY_OPERATOR_DEFINITION( objName, funcName, outputType, ...) \
		STANDARD_NULLARY_OPERATOR_OBJECT( objName, outputType, __VA_ARGS__) \
		STANDARD_NULLARY_FUNCTION_INTERFACE( funcName, objName)

// Unary Standard Maths Operators :
	#define STANDARD_UNARY_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename T> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T, void, void, false, false, false, false, void, void, void, void>::ReturnType ReturnType; \
			\
			__host__ __device__ inline static ReturnType apply(const T& a) \
			{ \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define BOOLEAN_UNARY_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename T> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T, void, void, false, false, false, false, bool, void, void, void>::ReturnType ReturnType; \
			\
			__host__ __device__ inline static ReturnType apply(const T& a) \
			{ \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define C2R_UNARY_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename T> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T, void, void, false, true, false, false, void, void, void, void>::ReturnType ReturnType; \
			\
			__host__ __device__ inline static ReturnType apply(const T& a) \
			{ \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define R2C_UNARY_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename T> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T, void, void, true, false, false, true, void, void, void, void>::ReturnType ReturnType; \
			\
			__host__ __device__ inline static ReturnType apply(const T& a) \
			{ \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define CAST_UNARY_OPERATOR(objName, ...) \
	namespace Kartet \
	{ \
		template<typename TOut> \
		struct objName \
		{ \
			template<typename T> \
			struct Sub \
			{ \
				typedef typename StandardOperatorTypeToolbox<T, void, void, false, false, false, false, TOut, void, void, void>::ReturnType ReturnType; \
				\
				__host__ __device__ inline static ReturnType apply(const T& a) \
				{ \
					__VA_ARGS__ \
				} \
			}; \
		}; \
	}

	#define STANDARD_UNARY_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename T, Location l> \
		ExpressionContainer< UnaryExpression< Accessor<T,l>, opName > > funcName (const Accessor<T,l>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< Accessor<T,l>, opName > >( UnaryExpression< Accessor<T,l>, opName >(a) ); \
		} \
		\
		template<typename T, Location l> \
		ExpressionContainer< UnaryExpression< Accessor<T,l>, opName > > funcName (const Array<T,l>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< Accessor<T,l>, opName > >( UnaryExpression< Accessor<T,l>, opName >(a) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< UnaryExpression< ExpressionContainer<T>, opName > > funcName (const ExpressionContainer<T>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< ExpressionContainer<T>, opName > >( UnaryExpression< ExpressionContainer<T>, opName >(a) ); \
		} \
	}

	#define EXTRA_UNARY_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename T> \
		__host__ __device__ typename opName <T>::ReturnType funcName (const T& a) \
		{ \
			return opName <T>::apply(a); \
		} \
	}

	#define CAST_UNARY_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename TOut, typename T, Location l> \
		ExpressionContainer< UnaryExpression< Accessor<T,l>, opName <TOut>::template Sub > > funcName (const Accessor<T,l>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< Accessor<T,l>, opName <TOut>::template Sub > >( UnaryExpression< Accessor<T,l>, opName <TOut>::template Sub >(a) ); \
		} \
		\
		template<typename TOut, typename T, Location l> \
		ExpressionContainer< UnaryExpression< Accessor<T,l>, opName <TOut>::template Sub > > funcName (const Array<T,l>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< Accessor<T,l>, opName <TOut>::template Sub > >( UnaryExpression< Accessor<T,l>, opName <TOut>::template Sub >(a) ); \
		} \
		\
		template<typename TOut, typename T> \
		ExpressionContainer< UnaryExpression< ExpressionContainer<T>, opName <TOut>::template Sub > > funcName (const ExpressionContainer<T>& a) \
		{ \
			return ExpressionContainer< UnaryExpression< ExpressionContainer<T>, opName <TOut>::template Sub > >( UnaryExpression< ExpressionContainer<T>, opName <TOut>::template Sub >(a) ); \
		} \
	}

	#define EXTRA_CAST_UNARY_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename TOut, typename T> \
		__host__ __device__ typename opName <TOut>::template Sub<T>::ReturnType funcName (const T& a) \
		{ \
			return opName <TOut>::template Sub<T>::apply(a); \
		} \
	}

	/**
	\ingroup FunctionsGroup
	\brief Create a standard unary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_UNARY_OPERATOR_DEFINITION instead).

	Declaration example :
	\code
	STANDARD_UNARY_OPERATOR_DEFINITION(UnOp_cosDiv2, cosDiv2, return ::cos(a/static_cast<T>(2)); )
	\endcode

	Call example :
	\code
	Array<...> A, B;
	B = cosDiv2(4*A);
	\endcode
	**/
	#define STANDARD_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		STANDARD_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a unary boolean operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_UNARY_OPERATOR_DEFINITION instead). The result is guaranteed to be boolean.

	Declaration example :
	\code
	BOOLEAN_UNARY_OPERATOR_DEFINITION(UnOp_mod2Test, mod2Test, return ::mod(a,static_cast<T>(2))==static_cast<T>(0); )
	\endcode

	Call example :
	\code
	Array<...> A, B;
	B = mod2Test(A);
	\endcode
	**/
	#define BOOLEAN_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		BOOLEAN_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a Complex To Real unary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_C2R_UNARY_OPERATOR_DEFINITION instead). The inputs are either real or complex and the output is guaranteed to be real.

	Declaration example :
	\code
	C2R_UNARY_OPERATOR_DEFINITION(UnOp_cxl1, cxl1, return real(a)+imag(a); )
	\endcode

	Call example :
	\code
	Array<...> A, B;
	A = cxl1( polar(4*B) );
	\endcode
	**/
	#define C2R_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		C2R_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a Real To Complex unary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_R2C_UNARY_OPERATOR_DEFINITION instead). It will enforce that the inputs are real and the output is guaranteed to be complex.

	Declaration example :
	\code
	R2C_UNARY_OPERATOR_DEFINITION(UnOp_expPol, expPol, return ::exp(I()*a+static_cast<T>(1)); )
	\endcode

	Call example :
	\code
	Array<...> A;
	A = expPol( IndexI()+IndexJ() );
	\endcode
	**/
	#define R2C_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		R2C_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a type casting unary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T' and the return type is 'TOut'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_CAST_UNARY_OPERATOR_DEFINITION instead). It will enforce that the return type is TOut.

	Declaration example :
	\code
	CAST_UNARY_OPERATOR_DEFINITION(UnOp_castFilter, castFilter, if(Kartet::IsSame<T,TOut>::value) return a; else return static_cast<TOut>(0); )
	\endcode

	Call example :
	\code
	Array<...> A, B;
	A = castFilter<double>(5.0);  // 5.0 everywhere.
	B = castFilter<double>(7.0f); // 0.0 everywhere.
	\endcode
	**/
	#define CAST_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		CAST_UNARY_OPERATOR(objName, __VA_ARGS__) \
		CAST_UNARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a standard unary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T'.

	This declaration creates an operator which is part of the Kartet namespace.

	Declaration example :
	\code
	EXTRA_UNARY_OPERATOR_DEFINITION(UnOp_cosDiv2, cosDiv2, return ::cos(a/static_cast<T>(2)); )
	\endcode

	Call example :
	\code
	std::cout << cosDiv2(1.0) << std::endl;
	\endcode
	**/
	#define EXTRA_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		STANDARD_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName) \
		EXTRA_UNARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a unary boolean operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T'.

	This declaration creates an operator which is part of the Kartet namespace. The result is guaranteed to be boolean.

	Declaration example :
	\code
	EXTRA_BOOLEAN_UNARY_OPERATOR_DEFINITION(UnOp_mod2Test, mod2Test, return ::mod(a,static_cast<T>(2))==static_cast<T>(0); )
	\endcode

	Call example :
	\code
	std::cout << mod2Test(1) << std::endl;
	\endcode
	**/
	#define EXTRA_BOOLEAN_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		BOOLEAN_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName) \
		EXTRA_UNARY_FUNCTION_INTERFACE(funcName, objName)
	
	/**
	\ingroup FunctionsGroup
	\brief Create a Complex To Real unary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T'.

	This declaration creates an operator which is part of the Kartet namespace. The inputs are either real or complex and the output is guaranteed to be real.

	Declaration example :
	\code
	EXTRA_C2R_UNARY_OPERATOR_DEFINITION(UnOp_cxl1, cxl1, return real(a)+imag(a); )
	\endcode

	Call example :
	\code
	std::cout << cxl1( polar(4.0) ) << std::endl;
	\endcode
	**/
	#define EXTRA_C2R_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		C2R_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName) \
		EXTRA_UNARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a Real To Complex unary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T'.

	This declaration creates an operator which is part of the Kartet namespace. It will enforce that the inputs are real and the output is guaranteed to be complex.

	Declaration example :
	\code
	EXTRA_R2C_UNARY_OPERATOR_DEFINITION(UnOp_expPol, expPol, return ::exp(I()*a+static_cast<T>(1)); )
	\endcode

	Call example :
	\code
	std::cout << expPol(5.0) << std::endl;
	\endcode
	**/
	#define EXTRA_R2C_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		R2C_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName) \
		EXTRA_UNARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a type casting unary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input argument is 'a', its type is 'T' and the return type is 'TOut'.

	This declaration creates an operator which is part of the Kartet namespace. It will enforce that the return type is TOut.

	Declaration example :
	\code
	EXTRA_CAST_UNARY_OPERATOR_DEFINITION(UnOp_castFilter, castFilter, if(Kartet::IsSame<T,TOut>::value) return a; else return static_cast<TOut>(0); )
	\endcode

	Call example :
	\code
	std::cout << castFilter<double>(5.0) << " or " << castFilter<double>(7.0f) << std::endl;
	\endcode
	**/
	#define EXTRA_CAST_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		CAST_UNARY_OPERATOR(objName, __VA_ARGS__) \
		CAST_UNARY_FUNCTION_INTERFACE(funcName, objName) \
		EXTRA_CAST_UNARY_FUNCTION_INTERFACE(funcName, objName)

// Standard Transform Operators :
	#define STANDARD_TRANSFORM_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		struct objName \
		{ \
			typedef void ReturnType; \
			\
			__host__ __device__ inline static ReturnType apply(const Layout& l, index_t& i, index_t& j, index_t& k) \
			{ \
				UNUSED_PARAMETER(l) \
				UNUSED_PARAMETER(i) \
				UNUSED_PARAMETER(j) \
				UNUSED_PARAMETER(k) \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define STANDARD_TRANSFORM_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename T, Location l> \
		ExpressionContainer< TransformExpression< Accessor<T,l>, opName > > funcName (const Accessor<T,l>& a) \
		{ \
			return ExpressionContainer< TransformExpression< Accessor<T,l>, opName > >( TransformExpression< Accessor<T,l>, opName >(a) ); \
		} \
		\
		template<typename T, Location l> \
		ExpressionContainer< TransformExpression< Accessor<T,l>, opName > > funcName (const Array<T,l>& a) \
		{ \
			return ExpressionContainer< TransformExpression< Accessor<T,l>, opName > >( TransformExpression< Accessor<T,l>, opName >(a) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< TransformExpression< ExpressionContainer<T>, opName > > funcName (const ExpressionContainer<T>& a) \
		{ \
			return ExpressionContainer< TransformExpression< ExpressionContainer<T>, opName > >( TransformExpression< ExpressionContainer<T>, opName >(a) ); \
		} \
	}

	/**
	\ingroup FunctionsGroup
	\brief Create an array transform operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The layout is 'l' (constant) and the coordinates are 'i', 'j' and 'k'.

	This declaration creates an operator which is part of the Kartet namespace and can only be used in expressions. It can change the coordinates of the sub expression it contains.

	Declaration example :
	\code
	STANDARD_TRANSFORM_OPERATOR_DEFINITION(UnOp_shiftRow2, shiftRow2, i=(i+2)%l.numRows(); )
	\endcode

	Call example :
	\code
	Array<...> A, B;
	B = shiftRow2(A); // Cannot be used in place, A and B must have the same size.
	\endcode
	**/
	#define STANDARD_TRANSFORM_OPERATOR_DEFINITION( objName, funcName, ...) \
		STANDARD_TRANSFORM_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_TRANSFORM_FUNCTION_INTERFACE( funcName, objName)

// Standard Layout Reinterpretation Operators :
	#define STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		struct objName \
		{ \
			typedef void ReturnType; \
			\
			__host__ __device__ inline static ReturnType apply(const Layout& l, const Layout& lnew, index_t& i, index_t& j, index_t& k) \
			{ \
				UNUSED_PARAMETER(l) \
				UNUSED_PARAMETER(lnew) \
				UNUSED_PARAMETER(i) \
				UNUSED_PARAMETER(j) \
				UNUSED_PARAMETER(k) \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define STANDARD_LAYOUT_REINTERPRETATION_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename T, Location l> \
		ExpressionContainer< LayoutReinterpretationExpression< Accessor<T,l>, opName > > funcName (const Accessor<T,l>& a) \
		{ \
			return ExpressionContainer< LayoutReinterpretationExpression< Accessor<T,l>, opName > >( LayoutReinterpretationExpression< Accessor<T,l>, opName >(a.layout(), a) ); \
		} \
		\
		template<typename T, Location l> \
		ExpressionContainer< LayoutReinterpretationExpression< Accessor<T,l>, opName > > funcName (const Array<T,l>& a) \
		{ \
			return ExpressionContainer< LayoutReinterpretationExpression< Accessor<T,l>, opName > >( LayoutReinterpretationExpression< Accessor<T,l>, opName >(a.layout(), a) ); \
		} \
		\
		template<typename T, Location l> \
		ExpressionContainer< LayoutReinterpretationExpression< Accessor<T,l>, opName > > funcName (const Layout& layout, const Accessor<T,l>& a) \
		{ \
			return ExpressionContainer< LayoutReinterpretationExpression< Accessor<T,l>, opName > >( LayoutReinterpretationExpression< Accessor<T,l>, opName >(layout, a) ); \
		} \
		\
		template<typename T, Location l> \
		ExpressionContainer< LayoutReinterpretationExpression< Accessor<T,l>, opName > > funcName (const Layout& layout, const Array<T,l>& a) \
		{ \
			return ExpressionContainer< LayoutReinterpretationExpression< Accessor<T,l>, opName > >( LayoutReinterpretationExpression< Accessor<T,l>, opName >(layout, a) ); \
		} \
		\
		template<typename T> \
		ExpressionContainer< LayoutReinterpretationExpression< ExpressionContainer<T>, opName > > funcName (const Layout& layout, const ExpressionContainer<T>& a) \
		{ \
			return ExpressionContainer< LayoutReinterpretationExpression< ExpressionContainer<T>, opName > >( LayoutReinterpretationExpression< ExpressionContainer<T>, opName >(layout, a) ); \
		} \
	}

	/**
	\ingroup FunctionsGroup
	\brief Create an layout reinterpretation operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The (external) layout is 'l' (constant), the (internal) layout is 'lnew' and the coordinates are 'i', 'j' and 'k'.

	This declaration creates an operator which is part of the Kartet namespace and can only be used in expressions. It can change the coordinates and layout of the sub expression it contains.

	Declaration example :
	\code
	STANDARD_TRANSFORM_OPERATOR_DEFINITION(UnOp_clampShift2, clampShift2, 	i = lnew.getIClamped(i+2);
										j = lnew.getJClamped(j+2);
										k = lnew.getKClamped(k+2); )
	\endcode

	Call example :
	\code
	Array<...> A, B;
	B = clampShift2(A); // Cannot be used in place, A and B do not necessarily have the same size.
	B = clampShift2(A.layout(), 2.0*cos(A)); // For an expression, we must supply the internal layout.
	\endcode
	**/
	#define STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( objName, funcName, ...) \
		STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_LAYOUT_REINTERPRETATION_FUNCTION_INTERFACE( funcName, objName)

// Binary Operators Tools :
	#define STANDARD_BINARY_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename T1, typename T2> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T1, T2, void, false, false, false, false, void, void, void, void>::ReturnType ReturnType; \
			 \
			__host__ __device__ inline static ReturnType apply(const T1& a, const T2& b) \
			{ \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define COMPARISON_BINARY_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename T1, typename T2> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T1, T2, void, false, false, false, false, bool, void, void, void>::ReturnType ReturnType; \
			 \
			__host__ __device__ inline static ReturnType apply(const T1& a, const T2& b) \
			{ \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define C2R_BINARY_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename T1, typename T2> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T1, T2, void, false, true, false, false, void, void, void, void>::ReturnType ReturnType; \
			 \
			__host__ __device__ inline static ReturnType apply(const T1& a, const T2& b) \
			{ \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define R2C_BINARY_OPERATOR_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename T1, typename T2> \
		struct objName \
		{ \
			typedef typename StandardOperatorTypeToolbox<T1, T2, void, true, false, false, true, void, void, void, void>::ReturnType ReturnType; \
			 \
			__host__ __device__ inline static ReturnType apply(const T1& a, const T2& b) \
			{ \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define STANDARD_BINARY_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename T1, Location l1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1,l1>, T2, opName > > funcName (const Accessor<T1,l1>& a, const T2& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1,l1>, T2, opName > >( BinaryExpression< Accessor<T1,l1>, T2, opName >(a, b) ); \
		} \
		\
		template<typename T1, Location l1, typename T2, Location l2> \
		ExpressionContainer< BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName > > funcName (const Accessor<T1,l1>& a, const Array<T2,l2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName > >( BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2, Location l2> \
		ExpressionContainer< BinaryExpression< T1, Accessor<T2,l2>, opName > > funcName (const T1& a, const Accessor<T2,l2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< T1, Accessor<T2,l2>, opName > >( BinaryExpression< T1, Accessor<T2,l2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, Location l1, typename T2, Location l2> \
		ExpressionContainer< BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName > > funcName (const Array<T1,l1>& a, const Accessor<T2,l2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName > >( BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, Location l1, typename T2, Location l2> \
		ExpressionContainer< BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName > > funcName (const Accessor<T1,l1>& a, const Accessor<T2,l2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName > >( BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, Location l1, typename T2, Location l2> \
		ExpressionContainer< BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName > > funcName (const Array<T1,l1>& a, const Array<T2,l2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName > >( BinaryExpression< Accessor<T1,l1>, Accessor<T2,l2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, T2, opName > > funcName (const ExpressionContainer<T1>& a, const T2& b) \
		{ \
			return ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, T2, opName > >( BinaryExpression< ExpressionContainer<T1>, T2, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2, Location l2> \
		ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, Accessor<T2,l2>, opName > > funcName (const ExpressionContainer<T1>& a, const Array<T2,l2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, Accessor<T2,l2>, opName > >( BinaryExpression< ExpressionContainer<T1>, Accessor<T2,l2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< T1, ExpressionContainer<T2>, opName > > funcName (const T1& a, const ExpressionContainer<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< T1, ExpressionContainer<T2>, opName > >( BinaryExpression< T1, ExpressionContainer<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, Location l1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1,l1>, ExpressionContainer<T2>, opName > > funcName (const Array<T1,l1>& a, const ExpressionContainer<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1,l1>, ExpressionContainer<T2>, opName > >( BinaryExpression< Accessor<T1,l1>, ExpressionContainer<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2, Location l2> \
		ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, Accessor<T2,l2>, opName > > funcName (const ExpressionContainer<T1>& a, const Accessor<T2,l2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, Accessor<T2,l2>, opName > >( BinaryExpression< ExpressionContainer<T1>, Accessor<T2,l2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, Location l1, typename T2> \
		ExpressionContainer< BinaryExpression< Accessor<T1,l1>, ExpressionContainer<T2>, opName > > funcName (const Accessor<T1,l1>& a, const ExpressionContainer<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< Accessor<T1,l1>, ExpressionContainer<T2>, opName > >( BinaryExpression< Accessor<T1,l1>, ExpressionContainer<T2>, opName >(a, b) ); \
		} \
		\
		template<typename T1, typename T2> \
		ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, ExpressionContainer<T2>, opName > > funcName (const ExpressionContainer<T1>& a, const ExpressionContainer<T2>& b) \
		{ \
			return ExpressionContainer< BinaryExpression< ExpressionContainer<T1>, ExpressionContainer<T2>, opName > >( BinaryExpression< ExpressionContainer<T1>, ExpressionContainer<T2>, opName >(a, b) ); \
		} \
	}

	#define EXTRA_BINARY_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename T1, typename T2> \
		__host__ __device__ typename opName <T1,T2>::ReturnType funcName (const T1& a, const T2& b) \
		{ \
			return opName <T1,T2>::apply(a,b); \
		} \
	}

	/**
	\ingroup FunctionsGroup
	\brief Create a standard binary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input arguments are 'a' and 'b', their respective types are 'T1' and 'T2'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_BINARY_OPERATOR_DEFINITION instead).

	Declaration example :
	\code
	STANDARD_BINARY_OPERATOR_DEFINITION(UnOp_cosSum, cosSum, return ::cos(a+b); )
	\endcode

	Call example :
	\code
	Array<...> A, B, C;
	C = cosSum(A, B);
	\endcode
	**/
	#define STANDARD_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		STANDARD_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a binary comparison operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input arguments are 'a' and 'b', their respective types are 'T1' and 'T2'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_COMPARISON_BINARY_OPERATOR_DEFINITION instead). The result is guaranteed to be boolean.

	Declaration example :
	\code
	COMPARISON_BINARY_OPERATOR_DEFINITION(BinOp_equalPlusOne, equalPlusOne, return a==b+1; )
	\endcode

	Call example :
	\code
	Array<...> A, B, C;
	C = equalPlusOne(A, B);
	\endcode
	**/
	#define COMPARISON_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		COMPARISON_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a Complex to Real binary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input arguments are 'a' and 'b', their respective types are 'T1' and 'T2'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_C2R_BINARY_OPERATOR_DEFINITIONinstead). The inputs are either real or complex and the output is guaranteed to be real.

	Declaration example :
	\code
	C2R_BINARY_OPERATOR_DEFINITION(BinOp_cxle, cxle, return real(a*b)+imag(a+b); )
	\endcode

	Call example :
	\code
	Array<...> A, B, C;
	C = cxle(A, B);
	\endcode
	**/
	#define C2R_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		C2R_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a Real To Complex binary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input arguments are 'a' and 'b', their respective types are 'T1' and 'T2'.

	This declaration creates an operator which is part of the Kartet namespace and <b>can only be used in expressions</b> (prefer EXTRA_R2C_BINARY_OPERATOR_DEFINITION). It will enforce that the inputs are real and the output is guaranteed to be complex.

	Declaration example :
	\code
	EXTRA_R2C_BINARY_OPERATOR_DEFINITION(BinOp_pole, pole, return polar(a*b+a); )
	\endcode

	Call example :
	\code
	Array<...> A, B, C;
	C = pole(A, B);
	\endcode
	**/
	#define R2C_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		R2C_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a standard binary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input arguments are 'a' and 'b', their respective types are 'T1' and 'T2'.

	This declaration creates an operator which is part of the Kartet namespace.

	Declaration example :
	\code
	STANDARD_BINARY_OPERATOR_DEFINITION(UnOp_cosSum, cosSum, return ::cos(a+b); )
	\endcode

	Call example :
	\code
	std::cout << cosSum(K_PI, K_PI) << std::endl;
	\endcode
	**/
	#define EXTRA_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		STANDARD_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName) \
		EXTRA_BINARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a binary comparison operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input arguments are 'a' and 'b', their respective types are 'T1' and 'T2'.

	This declaration creates an operator which is part of the Kartet namespace. The result is guaranteed to be boolean.

	Declaration example :
	\code
	COMPARISON_BINARY_OPERATOR_DEFINITION(BinOp_equalPlusOne, equalPlusOne, return a==b+1; )
	\endcode

	Call example :
	\code
	std::cout << equalPlusOne(2.0, 1.0) << std::endl;
	\endcode
	**/
	#define EXTRA_COMPARISON_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		COMPARISON_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName) \
		EXTRA_BINARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a Complex to Real binary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input arguments are 'a' and 'b', their respective types are 'T1' and 'T2'.

	This declaration creates an operator which is part of the Kartet namespace. The inputs are either real or complex and the output is guaranteed to be real.

	Declaration example :
	\code
	C2R_BINARY_OPERATOR_DEFINITION(BinOp_cxle, cxle, return real(a*b)+imag(a+b); )
	\endcode

	Call example :
	\code
	std::cout << cxle(I(), 1.0+I()) << std::end;
	\endcode
	**/
	#define EXTRA_C2R_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		C2R_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName) \
		EXTRA_BINARY_FUNCTION_INTERFACE(funcName, objName)

	/**
	\ingroup FunctionsGroup
	\brief Create a Real To Complex binary operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The input arguments are 'a' and 'b', their respective types are 'T1' and 'T2'.

	This declaration creates an operator which is part of the Kartet namespace. It will enforce that the inputs are real and the output is guaranteed to be complex.

	Declaration example :
	\code
	EXTRA_R2C_BINARY_OPERATOR_DEFINITION(BinOp_pole, pole, return polar(a*b+a); )
	\endcode

	Call example :
	\code
	std::cout << pole(K_PI, 1.0) << std::endl;
	\endcode
	**/
	#define EXTRA_R2C_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		R2C_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName) \
		EXTRA_BINARY_FUNCTION_INTERFACE(funcName, objName)

// Shuffle Operator Tools (can make use of v variable from the index data) :
	#define STANDARD_SHUFFLE_FUNCTION_OBJECT(objName, ...) \
	namespace Kartet \
	{ \
		template<typename TIndex> \
		struct objName \
		{ \
			typedef void ReturnType; \
			\
			__host__ __device__ inline static ReturnType apply(const TIndex& index, const Layout& l, index_t& i, index_t& j, index_t& k) \
			{ \
				UNUSED_PARAMETER(l) \
				UNUSED_PARAMETER(i) \
				UNUSED_PARAMETER(j) \
				UNUSED_PARAMETER(k) \
				typedef typename ExpressionEvaluation<TIndex>::ReturnType IndexType; \
				const IndexType v = ExpressionEvaluation<TIndex>::evaluate(index, l, i, j, k); \
				__VA_ARGS__ \
			} \
		}; \
	}

	#define SHUFFLE_FUNCTION_INTERFACE(funcName, opName) \
	namespace Kartet \
	{ \
		template<typename TIndex, typename TData, Location lData> \
		ExpressionContainer< ShuffleExpression< TIndex, Array<TData,lData>, opName > > funcName (const TIndex& index, const Array<TData,lData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression<TIndex, Array<TData,lData>, opName > >( ShuffleExpression<TIndex, Array<TData,lData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, typename TData, Location lData> \
		ExpressionContainer< ShuffleExpression< TIndex, Accessor<TData,lData>, opName > > funcName (const TIndex& index, const Accessor<TData,lData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression<TIndex, Accessor<TData,lData>, opName > >( ShuffleExpression<TIndex, Accessor<TData,lData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, typename TData> \
		ExpressionContainer< ShuffleExpression< TIndex, ExpressionContainer<TData>, opName > > funcName (const TIndex& index, const ExpressionContainer<TData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression<TIndex, ExpressionContainer<TData>, opName > >( ShuffleExpression<TIndex, ExpressionContainer<TData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, Location lIndex, typename TData> \
		ExpressionContainer< ShuffleExpression< Accessor<TIndex,lIndex>, TData, opName > > funcName (const Accessor<TIndex,lIndex>& index, const TData& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< Accessor<TIndex,lIndex>, TData, opName > >( ShuffleExpression< Accessor<TIndex,lIndex>, TData, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, Location lIndex, typename TData> \
		ExpressionContainer< ShuffleExpression< Array<TIndex,lIndex>, TData, opName > > funcName (const Array<TIndex,lIndex>& index, const TData& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< Array<TIndex,lIndex>, TData, opName > >( ShuffleExpression< Array<TIndex,lIndex>, TData, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, typename TData> \
		ExpressionContainer< ShuffleExpression< ExpressionContainer<TIndex>, TData, opName > > funcName (const ExpressionContainer<TIndex>& index, const TData& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< ExpressionContainer<TIndex>, TData, opName > >( ShuffleExpression< ExpressionContainer<TIndex>, TData, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, Location lIndex, typename TData, Location lData> \
		ExpressionContainer< ShuffleExpression< Accessor<TIndex,lIndex>, Array<TData,lData>, opName > > funcName (const Accessor<TIndex,lIndex>& index, const Array<TData,lData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< Accessor<TIndex,lIndex>, Array<TData,lData>, opName > >( ShuffleExpression< Accessor<TIndex,lIndex>, Array<TData,lData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, Location lIndex, typename TData, Location lData> \
		ExpressionContainer< ShuffleExpression< Array<TIndex,lIndex>, Accessor<TData,lData>, opName > > funcName (const Array<TIndex,lIndex>& index, const Accessor<TData,lData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< Array<TIndex,lIndex>, Accessor<TData,lData>, opName > >( ShuffleExpression< Array<TIndex,lIndex>, Accessor<TData,lData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, Location lIndex, typename TData, Location lData> \
		ExpressionContainer< ShuffleExpression< Accessor<TIndex,lIndex>, Accessor<TData,lData>, opName > > funcName (const Accessor<TIndex,lIndex>& index, const Accessor<TData,lData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< Accessor<TIndex,lIndex>, Accessor<TData,lData>, opName > >( ShuffleExpression< Accessor<TIndex,lIndex>, Accessor<TData,lData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, Location lIndex, typename TData, Location lData> \
		ExpressionContainer< ShuffleExpression< Array<TIndex,lIndex>, Array<TData,lData>, opName > > funcName (const Array<TIndex,lIndex>& index, const Array<TData,lData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< Array<TIndex,lIndex>, Array<TData,lData>, opName > >( ShuffleExpression< Array<TIndex,lIndex>, Array<TData,lData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, typename TData, Location lData> \
		ExpressionContainer< ShuffleExpression< ExpressionContainer<TIndex>, Array<TData,lData>, opName > > funcName (const ExpressionContainer<TIndex>& index, const Array<TData,lData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< ExpressionContainer<TIndex>, Array<TData,lData>, opName > >( ShuffleExpression< ExpressionContainer<TIndex>, Array<TData,lData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, Location lIndex, typename TData> \
		ExpressionContainer< ShuffleExpression< Array<TIndex,lIndex>, ExpressionContainer<TData>, opName > > funcName (const Array<TIndex,lIndex>& index, const ExpressionContainer<TData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< Array<TIndex,lIndex>, ExpressionContainer<TData>, opName > >( ShuffleExpression< Array<TIndex,lIndex>, ExpressionContainer<TData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, typename TData, Location lData> \
		ExpressionContainer< ShuffleExpression< ExpressionContainer<TIndex>, Accessor<TData,lData>, opName > > funcName (const ExpressionContainer<TIndex>& index, const Accessor<TData,lData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< ExpressionContainer<TIndex>, Accessor<TData,lData>, opName > >( ShuffleExpression< ExpressionContainer<TIndex>, Accessor<TData,lData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, Location lIndex, typename TData> \
		ExpressionContainer< ShuffleExpression< Accessor<TIndex,lIndex>, ExpressionContainer<TData>, opName > > funcName (const Accessor<TIndex,lIndex>& index, const ExpressionContainer<TData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< Accessor<TIndex,lIndex>, ExpressionContainer<TData>, opName > >( ShuffleExpression< Accessor<TIndex,lIndex>, ExpressionContainer<TData>, opName >(index, data) ); \
		} \
		\
		template<typename TIndex, typename TData> \
		ExpressionContainer< ShuffleExpression< ExpressionContainer<TIndex>, ExpressionContainer<TData>, opName > > funcName (const ExpressionContainer<TIndex>& index, const ExpressionContainer<TData>& data) \
		{ \
			return ExpressionContainer< ShuffleExpression< ExpressionContainer<TIndex>, ExpressionContainer<TData>, opName > >( ShuffleExpression< ExpressionContainer<TIndex>, ExpressionContainer<TData>, opName >(index, data) ); \
		} \
	}

	/**
	\ingroup FunctionsGroup
	\brief Create an shuffling operator.
	\param objName Name of the operator object.
	\param funcName Name of the function.
	\param ... Function body. The new index is 'index', the layout is 'l' (constant) and the coordinates are 'i', 'j' and 'k'.

	This declaration creates an operator which is part of the Kartet namespace and can only be used in expressions. It can change the coordinates for the sub-expression it contains via another expression.

	Declaration example :
	\code
	STANDARD_TRANSFORM_OPERATOR_DEFINITION(UnOp_clampShift2, clampShift2, 	i = lnew.getIClamped(i+2);
										j = lnew.getJClamped(j+2);
										k = lnew.getKClamped(k+2); )
	\endcode

	Call example :
	\code
	Array<...> A, B;
	B = clampShift2(A); // Cannot be used in place, A and B do not necessarily have the same size.
	B = clampShift2(A.layout(), 2.0*cos(A)); // For an expression, we must supply the internal layout.
	\endcode
	**/
	#define STANDARD_SHUFFLE_FUNCTION_DEFINITION( objName, funcName, ...) \
		STANDARD_SHUFFLE_FUNCTION_OBJECT( objName, __VA_ARGS__) \
		SHUFFLE_FUNCTION_INTERFACE( funcName, objName)

// Standard operators : 
	// Unary : 
		STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_Minus,		operator-,	return -a;)
		BOOLEAN_UNARY_OPERATOR_DEFINITION(	UnOp_Not,		operator!,	return !a;)
		STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_BinaryCompl,	operator~,	return ~a;)

	// Binary :
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Plus, 		operator+, 	return a+b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Minus, 		operator-, 	return a-b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Times, 		operator*, 	return a*b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Divide, 		operator/, 	return a/b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Remainder, 	operator%, 	return a%b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_BinaryAnd, 	operator&, 	return a&b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_BinaryOr, 	operator|, 	return a|b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_And, 		operator&&, 	return a && b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Or, 		operator||, 	return a || b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Xor, 		operator^, 	return a^b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_LeftShift,	operator<<,	return a << b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_RightShift,	operator>>,	return a >> b; )

		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_Equal, 		operator==, 	return a==b; )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_Different, 	operator!=, 	return a!=b; )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_GreaterStrict, 	operator>, 	return a>b; )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_GreaterOrEqual, 	operator>=, 	return a>=b; )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_SmallerStrict, 	operator<, 	return a<b; )
		COMPARISON_BINARY_OPERATOR_DEFINITION( 	BinOp_SmallerOrEqual, 	operator<=, 	return a<=b; )

// Unary and Binary functions :
	#include "Core/ArrayFunctions.hpp"
	
// Clean the mess :
	// #undef ?

#endif

