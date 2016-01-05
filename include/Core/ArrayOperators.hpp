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
		StaticAssert<!(Traits< typename ExpressionEvaluation<TExpr>::ReturnType >::isComplex && !Traits<T>::isComplex)>();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
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
		StaticAssert<l!=l2>(); // The two accessors are on different sides.

		#ifdef __CUDACC__
			MemCpyDualToolBox<T> op((l==DeviceSide) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream);
			dualScan(*this, dataPtr(), a, a.dataPtr(), op);
			
			return *this;
		#else
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
		StaticAssert<!(Traits< typename ExpressionEvaluation<TExpr>::ReturnType >::isComplex && !Traits<T>::isComplex)>();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
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
		To change arity, set type of ith argument to void.
	*/
	template<typename T1, typename T2, typename T3, bool preferComplexOutput, bool preferRealOutput, bool inputMustBeComplex, bool inputMustBeReal, typename forceReturnType, typename enforceT1, typename enforceT2, typename enforceT3>
	struct StandardOperatorTypeToolbox
	{
		private :
			// Test for correct order of arguments :
			static const StaticAssert< 	(IsSame<T3,void>::value) ? 
							( // No third argument
								!(IsSame<T1,void>::value) // whatever 2nd arg is, there must be at least one.
							)
							:
							(
								(!IsSame<T1,void>::value) && (!IsSame<T2,void>::value)
							)
						> 										test1;

			// Test for conflict free requests :
			static const StaticAssert<	!(preferComplexOutput && preferRealOutput) > 				test2;
			static const StaticAssert<	!(inputMustBeReal && inputMustBeComplex) > 				test3;

			// Test for type enforcing :
			static const StaticAssert<	IsSame<enforceT1,void>::value || IsSame<enforceT1,T1>::value>		test4;
			static const StaticAssert<	IsSame<enforceT2,void>::value || IsSame<enforceT2,T2>::value>		test5;
			static const StaticAssert<	IsSame<enforceT3,void>::value || IsSame<enforceT3,T3>::value>		test6;

			// Test validity :
			static const StaticAssert<	!(inputMustBeReal && Traits<T1>::isComplex) >				test_ir1;
			static const StaticAssert<	!(inputMustBeReal && Traits<T2>::isComplex) >				test_ir2;
			static const StaticAssert<	!(inputMustBeReal && Traits<T3>::isComplex) >				test_ir3;
			static const StaticAssert<	!(inputMustBeComplex && !Traits<T1>::isComplex) >			test_ic1;
			static const StaticAssert<	!(inputMustBeComplex && !Traits<T2>::isComplex) >			test_ic2;
			static const StaticAssert<	!(inputMustBeComplex && !Traits<T3>::isComplex) >			test_ic3;
			
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
				typedef typename StandardOperatorTypeToolbox<TOut, void, void, false, false, false, false, void, void, void, void>::ReturnType ReturnType; \
				\
				__host__ __device__ inline static ReturnType apply(const T& a) \
				{ \
					return static_cast<TOut>(__VA_ARGS__) ; \
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

	#define STANDARD_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		STANDARD_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	#define C2R_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		C2R_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	#define R2C_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		R2C_UNARY_OPERATOR_OBJECT(objName, __VA_ARGS__) \
		STANDARD_UNARY_FUNCTION_INTERFACE(funcName, objName)

	#define CAST_UNARY_OPERATOR_DEFINITION(objName, funcName, ...) \
		CAST_UNARY_OPERATOR(objName, __VA_ARGS__) \
		CAST_UNARY_FUNCTION_INTERFACE(funcName, objName)

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
			__host__ __device__ inline static ReturnType apply(T1 a, T2 b) \
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
			__host__ __device__ inline static ReturnType apply(T1 a, T2 b) \
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
			__host__ __device__ inline static ReturnType apply(T1 a, T2 b) \
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

	#define STANDARD_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		STANDARD_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

	#define COMPARISON_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		COMPARISON_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

	#define R2C_BINARY_OPERATOR_DEFINITION( objName, funcName, ...) \
		R2C_BINARY_OPERATOR_OBJECT( objName, __VA_ARGS__) \
		STANDARD_BINARY_FUNCTION_INTERFACE( funcName, objName)

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

	#define STANDARD_SHUFFLE_FUNCTION_DEFINITION( objName, funcName, ...) \
		STANDARD_SHUFFLE_FUNCTION_OBJECT( objName, __VA_ARGS__) \
		SHUFFLE_FUNCTION_INTERFACE( funcName, objName)

// Standard operators : 
	// Unary : 
		STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_Minus,		operator-,	return -a;)
		STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_Not,		operator!,	return !a;)
		STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_BinaryCompl,	operator~,	return ~a;)

	// Binary :
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Plus, 		operator+, 	return a+b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Minus, 		operator-, 	return a-b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Times, 		operator*, 	return a*b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Divide, 		operator/, 	return a/b; )
		STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_Modulo, 		operator%, 	return a%b; )
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

