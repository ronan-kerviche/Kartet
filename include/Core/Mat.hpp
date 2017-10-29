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
	\file    Mat.hpp
	\brief   Matrix maths tools.
	\author  R. Kerviche
	\date    June 26th 2017
**/

#ifndef __KARTET_MATRIX_MATHS_TOOLS__
#define __KARTET_MATRIX_MATHS_TOOLS__

	#include "Core/LibTools.hpp"
	#include "Core/Meta.hpp"
	#include "Core/Complex.hpp"
	#include "Core/Traits.hpp"

namespace Kartet
{
	/**
	\brief Small matrix object (of static size).
	\tparam r Number of rows.
	\tparam c Number of columns.
	\tparam T Type of the matrix.

	Small matrix object with static size :
	\code
	Kartet::Mat<3,4,double> m1; // Unitialized value.
	Kartet::Mat<2,4,float> m2(0.0); // Cleared to 0.
	Mat3f m3 = Mat3f::identity(); // 3x3 <float> identity.
	Mat4z m4 = Mat4d::identity(); // 4x4 Complex<double> identity, initialized from a double identity.
	\endcode
	**/
	template<int r, int c, typename T>
	struct Mat
	{
		STATIC_ASSERT_VERBOSE((r*c)>1, INVALID_DIMENSION)
		typedef T BaseType;
		static const int	rows = r,
					cols = c,
					dim = r*c;
		T m[r*c];

		__host__ __device__ inline Mat(void);
		__host__ __device__ inline Mat(const T& val);
		__host__ __device__ inline Mat(const Mat<r, c, T>& o);
		template<typename U>
		__host__ __device__ inline Mat(const Mat<r, c, U>& o);
		template<typename U>
		__host__ __device__ inline Mat(const U* ptr);
		__host__ __device__ inline const T& operator()(const int& i, const int& j) const;
		__host__ __device__ inline T& operator()(const int& i, const int& j);
		__host__ __device__ inline Mat<r,1,T> col(const int& j);
		__host__ __device__ inline Mat<r,c,T>& operator=(const Mat<r, c, T>& o);
		template<typename U>
		__host__ __device__ inline Mat<r,c,T>& operator=(const Mat<r, c, U>& o);
		__host__ __device__ inline Mat<r,c,T>& operator+=(const Mat<r, c, T>& o);
		template<typename U>
		__host__ __device__ inline Mat<r,c,T>& operator+=(const Mat<r, c, U>& o);
		__host__ __device__ inline Mat<r,c,T>& operator-=(const Mat<r, c, T>& o);
		template<typename U>
		__host__ __device__ inline Mat<r,c,T>& operator-=(const Mat<r, c, U>& o);
		template<typename U>
		__host__ __device__ inline Mat<r,c,T>& operator*=(const U& o);
		template<int rb, int cb, typename U>
		__host__ __device__ inline Mat<r,c,T>& operator*=(const Mat<rb, cb, U>& m);
		template<typename U>
		__host__ __device__ inline Mat<r,c,T>& operator/=(const U& o);
		__host__ __device__ inline Mat<r,c,T>& clear(const T& val);
		template<typename U>
		__host__ __device__ inline Mat<r,c,T>& set(const U* ptr);

		// Tools :
		__host__ __device__ inline static Mat<r,c,T> identity(void);
	};

	// Type aliases :
	/**
	\typedef Mat2f
	\brief Alias to Mat<2,2,float> type, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat3f
	\brief Alias to Mat<3,3,float>, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat4f
	\brief Alias to Mat<4,4,float>, see Kartet::Mat for more information.
	\relatedalso Kartet::Mat
	\typedef Mat2c
	\brief Alias to Mat<2,2,Complex<float> > type, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat3c
	\brief Alias to Mat<3,3,Complex<float> >, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat4c
	\brief Alias to Mat<4,4,Complex<float> >, see Kartet::Mat for more information.
	\relatedalso Kartet::Mat
	\typedef Mat2d
	\brief Alias to Mat<2,2,double> type, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat3d
	\brief Alias to Mat<3,3,double>, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat4d
	\brief Alias to Mat<4,4,double>, see Kartet::Mat for more information.
	\relatedalso Kartet::Mat
	\typedef Mat2z
	\brief Alias to Mat<2,2,Complex<double> > type, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat3z
	\brief Alias to Mat<3,3,Complex<double> >, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat4z
	\brief Alias to Mat<4,4,Complex<double> >, see Kartet::Mat for more information.
	\relatedalso Kartet::Mat
	**/
	typedef Mat<2,2,float> Mat2f;
	typedef Mat<3,3,float> Mat3f;
	typedef Mat<4,4,float> Mat4f;
	typedef Mat<2,2,Complex<float> > Mat2c;
	typedef Mat<3,3,Complex<float> > Mat3c;
	typedef Mat<4,4,Complex<float> > Mat4c;
	typedef Mat<2,2,double> Mat2d;
	typedef Mat<3,3,double> Mat3d;
	typedef Mat<4,4,double> Mat4d;
	typedef Mat<2,2,Complex<double> > Mat2z;
	typedef Mat<3,3,Complex<double> > Mat3z;
	typedef Mat<4,4,Complex<double> > Mat4z;

	// Traits :
	template<int r, int c, typename T>
	struct Traits<Mat<r,c,T> >
	{
		typedef Traits<T> SubTraits;
		typedef typename SubTraits::BaseType BaseType;
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= SubTraits::isComplex,
					isFloatingPoint = SubTraits::isFloatingPoint,
					isMatrix	= true,
					isVector	= (c==1);
	};

	// Functions :
	/**
	\brief Default matrix constructor.
	The elements are left unitialized.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(void)
	{ }

	/**
	\brief Matrix constructor.
	\param val Fill value.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(const T& val)
	{
		clear(val);
	}

	/**
	\brief Matrix copy constructor.
	\param o The matrix to be copied.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(const Mat<r,c,T>& o)
	{
		metaUnaryEqual<dim>(m, o.m);
	}

	/**
	\brief Matrix copy constructor.
	\param o The matrix to be copied.
	**/
	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>::Mat(const Mat<r,c,U>& o)
	{
		metaUnaryEqual<dim>(m, o.m);
	}

	/**
	\brief Matrix constructor.
	\param ptr Pointer to the array to be copied. It must contain r*c elements.
	**/
	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>::Mat(const U* ptr)
	{
		metaUnaryEqual<dim>(m, ptr);
	}

	/**
	\brief Read a matrix element.
	\param i Row index.
	\param j Column index.
	\return A reference to the value of the matrix at the index \f$(i,j)\f$.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline const T& Mat<r,c,T>::operator()(const int& i, const int& j) const
	{
		return m[j*r+i];
	}

	/**
	\brief Read/write a matrix element.
	\param i Row index.
	\param j Column index.
	\return A reference to the value of the matrix at the index \f$(i,j)\f$.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline T& Mat<r,c,T>::operator()(const int& i, const int& j)
	{
		return m[j*r+i];
	}

	/**
	\brief Copy a matrix column.
	\param j Column index.
	\return A matrix containing a copy of column j.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,1,T> Mat<r,c,T>::col(const int& j)
	{
		Mat<r,1,T> v(m+j*r);
		return v;
	}

	/**
	\brief Copy operator.
	\param o The matrix to be copied.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator=(const Mat<r,c,T>& o)
	{
		metaUnaryEqual<dim>(this->m, o.m);
		return *this;
	}

	/**
	\brief Copy operator.
	\param o The matrix to be copied.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator=(const Mat<r,c,U>& o)
	{
		metaUnaryEqual<dim>(this->m, o.m);
		return *this;
	}

	/**
	\brief Coumpound addition operator.
	\param o The matrix to be added to this.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator+=(const Mat<r,c,T>& o)
	{
		metaBinaryPlus<dim>(this->m, reinterpret_cast<const T*>(this->m), o.m);
		return *this;
	}

	/**
	\brief Coumpound addition operator.
	\param o The matrix to be added to this.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator+=(const Mat<r,c,U>& o)
	{
		metaBinaryPlus<dim>(this->m, reinterpret_cast<const T*>(this->m), o.m);
		return *this;
	}

	/**
	\brief Coumpound substraction operator.
	\param o The matrix to be removed to this.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator-=(const Mat<r,c,T>& o)
	{
		metaBinaryMinus<dim>(this->m, reinterpret_cast<const T*>(this->m), o.m);
		return *this;
	}

	/**
	\brief Coumpound substraction operator.
	\param o The matrix to be removed to this.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator-=(const Mat<r,c,U>& o)
	{
		metaBinaryMinus<dim>(this->m, reinterpret_cast<const T*>(this->m), o.m);
		return *this;
	}

	/**
	\brief Coumpound product operator.
	\param o The scalar to be multiplied with this.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator*=(const U& o)
	{
		metaBinaryProduct<dim>(this->m, this->m, o);
		return *this;
	}

	/**
	\brief Coumpound matrix product operator.
	\param b The matrix to be multiplied (on the right hand-side) with this.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	template<int rb, int cb, typename Tb>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator*=(const Mat<rb,cb,Tb>& b)
	{
		STATIC_ASSERT_VERBOSE(c==rb, INVALID_DIMENSION)
		Mat<r,c,T> t = (*this);
		for(int j=0; j<cb; j++)
			for(int i=0; i<r; i++)
				(*this)(i,j) = metaBinaryProductSum<c, typename ResultingType<T,Tb>::Type>(t.m+i, b.m+j*rb, r, 1);
		return *this;
	}

	/**
	\brief Coumpound quotient operator.
	\param o The scalar to be multiplied with this.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::operator/=(const U& o)
	{
		metaBinaryQuotient<dim>(this->m, this->m, o);
		return (*this);
	}

	/**
	\brief Clear a matrix with a fill value.
	\param val The fill value.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::clear(const T& val)
	{
		metaUnaryEqual<Mat<r,c,T>::dim>(this->m, val);
		return (*this);
	}

	/**
	\brief Set matrix components.
	\param ptr A pointer to the array of r*c elements.
	\return A reference to this.
	**/
	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>& Mat<r,c,T>::set(const U* ptr)
	{
		metaUnaryEqual<dim>(m, ptr);
		return (*this);
	}

	/**
	\brief Create a pseudo-identity matrix.
	\return A pseudo-identity matrix.

	If the matrix is squarred then this is a correct identity. Otherwise, the matrix will have one on all the elements for which \f$i==j\f$.
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T> Mat<r,c,T>::identity(void)
	{
		Mat<r,c,T> res;
		for(int k=0; k<Mat<r,c,T>::dim; k++)
			res.m[k] = (k%(r+1)==0) ? static_cast<T>(1) : static_cast<T>(0);
		return res;
	}

// Non-members :
	/**
	\brief Unary minus operator.
	\param a Input matrix.
	\return A new matrix set to \f$-a\f$.
	\related Mat
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T> operator-(const Mat<r,c,T>& a)
	{
		Mat<r,c,T> res;
		metaUnaryMinus<Mat<r,c,T>::dim>(res.m, a.m);
		return res;
	}

	/**
	\brief Binary addition operator.
	\param a Input matrix.
	\param b Input matrix.
	\return A new matrix set to \f$a+b\f$.
	\related Mat
	**/
	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator+(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryPlus<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	/**
	\brief Binary substraction operator.
	\param a Input matrix.
	\param b Input matrix.
	\return A new matrix set to \f$a-b\f$.
	\related Mat
	**/
	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator-(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryMinus<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	/**
	\brief Binary scalar product operator.
	\param a Input matrix.
	\param b Input scalar.
	\return A new vector set to \f$ab\f$.
	\related Mat
	**/
	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator*(const Mat<r,c,T>& a, const U& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a.m, b);
		return res;
	}

	/**
	\brief Binary scalar product operator.
	\param a Input scalar.
	\param b Input matrix.
	\return A new matrix set to \f$ab\f$.
	\related Mat
	**/
	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator*(const T& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a, b.m);
		return res;
	}

	/**
	\brief Binary element-wise product operator.
	\param a Input matrix.
	\param b Input matrix.
	\return A new matrix set to \f$a.b\f$.
	\related Mat
	**/
	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> modulate(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	/**
	\brief Binary matrix product operator.
	\param a Input matrix.
	\param b Input matrix.
	\return A new matrix set to \f$a \times b\f$.
	\related Mat
	**/
	template<int ra, int ca, typename Ta, int rb, int cb, typename Tb>
	__host__ __device__ inline Mat<ra,cb,typename ResultingType<Ta,Tb>::Type> operator*(const Mat<ra,ca,Ta>& a, const Mat<rb,cb,Tb>& b)
	{
		STATIC_ASSERT_VERBOSE(ca==rb, INVALID_DIMENSION)
		Mat<ra,cb,typename ResultingType<Ta,Tb>::Type> res;
		for(int j=0; j<cb; j++)
			for(int i=0; i<ra; i++)
				res(i,j) = metaBinaryProductSum<ca, typename ResultingType<Ta,Tb>::Type>(a.m+i, b.m+j*rb, ra, 1);
		return res;
	}

	/**
	\brief Binary scalar quotient operator.
	\param a Input matrix.
	\param b Input scalar.
	\return A new matrix set to \f$a/b\f$.
	\related Mat
	**/
	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator/(const Mat<r,c,T>& a, const U& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryQuotient<Mat<r,c,T>::dim>(res.m, a.m, b);
		return res;
	}

	/**
	\brief Trace operator.
	\param a Square input matrix.
	\return The trace of the input matrix : \f$Tr(a)\f$ (sum of the elements on the main diagonal).
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline T trace(const Mat<r,r,T>& a)
	{
		return metaUnaryPlusSum<r, T>(a.m, r+1);
	}

	/**
	\brief Transpose operator.
	\param a Input matrix.
	\return A new matrix, being the transpose of the input matrix : \f$a^\intercal\f$ (\f$(i,j)\rightarrow (j,i)\f$).
	\related Mat
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<c,r,T> transpose(const Mat<r,c,T>& a)
	{	
		Mat<c,r,T> res;
		for(int j=0; j<r; j++)
			for(int i=0; i<c; i++)
				res(i,j) = a(j,i);
		return res;
	}

	/**
	\brief In place transpose operator.
	\param a Square input matrix.
	\return A reference to the input matrix, after transposition : \f$a^\intercal\f$ (\f$(i,j)\rightarrow (j,i)\f$).
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline Mat<r,r,T>& transposeInPlace(Mat<r,r,T>& a)
	{	
		for(int j=0; j<r; j++)
			for(int i=0; i<r; i++)
				swap(a(i,j),a(j,i));
		return a;
	}

	/**
	\brief Hermitian transpose operator.
	\param a Input matrix.
	\return A new matrix, being the hermitian transpose of the input matrix : \f$a^\dagger\f$ (\f$\overline{(i,j)}\rightarrow (j,i)\f$).
	\related Mat
	**/
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<c,r,T> hermTranspose(const Mat<r,c,T>& a)
	{	
		Mat<c,r,T> res;
		for(int j=0; j<r; j++)
			for(int i=0; i<c; i++)
				res(i,j) = conj(a(j,i));
		return res;
	}

	/**
	\brief Compute the determinant of a 2x2 matrix.
	\param m Input pointer.
	\return The determinant.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline T determinant2x2(const T* m)
	{
		return m[0]*m[3] - m[1]*m[2];
	}

	/**
	\brief Compute the determinant of a 2x2 matrix.
	\param a Input matrix.
	\return The determinant.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline T determinant2x2(const Mat<2,2,T>& a)
	{
		return determinant2x2(a.m);
	}

	/**
	\brief Compute the determinant of a 3x3 matrix.
	\param m Input pointer.
	\return The determinant.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline T determinant3x3(const T* m)
	{
		return m[0]*m[4]*m[8] + m[1]*m[5]*m[6] + m[2]*m[3]*m[7] - m[2]*m[4]*m[6] - m[1]*m[3]*m[8] - m[0]*m[5]*m[7];
	}

	/**
	\brief Compute the determinant of a 3x3 matrix.
	\param a Input matrix.
	\return The determinant.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline T determinant3x3(const Mat<3,3,T>& a)
	{
		return determinant3x3(a.m);
	}

	template<int r, typename T>
	__host__ __device__ inline T determinant(const T* a)
	{
		if(r==1)
			return *a;
		else if(r==2)
			return determinant2x2(a);
		else if(r==3)
			return determinant3x3(a);
		else
			throw InvalidOperation; // To be implemented.
	}

	template<int r, typename T>
	__host__ __device__ inline T determinant(const Mat<r,r,T>& a)
	{
		return determinant<r>(a.m);
	}

	/**
	\brief Compute the dot product between two vectors.
	\param a Input vector.
	\param b Input vector.
	\return The dot product \f$ a \cdot b = \sum_k a_k b_k\f$.
	\related Mat
	**/
	template<int r, typename T, typename U>
	__host__ __device__ inline typename ResultingType<T,U>::Type dot(const Mat<r,1,T>& a, const Mat<r,1,U>& b)
	{
		return metaBinaryProductSum<r, typename ResultingType<T,U>::Type>(a.m, b.m);
	}

	/**
	\brief Compute the hermitian dot product between two vectors.
	\param a Input vector.
	\param b Input vector.
	\return The hermitian dot product \f$ a \cdot b = \sum_k \overline{a_k} b_k \f$.
	\related Mat
	**/
	template<int r, typename T, typename U>
	__host__ __device__ inline typename ResultingType<T,U>::Type hermDot(const Mat<r,1,T>& a, const Mat<r,1,U>& b)
	{
		return metaBinaryHermProductSum<r, typename ResultingType<T,U>::Type>(a.m, b.m);
	}

	/**
	\brief Compute the Frobenius norm of the square matrix.
	\param a Square input matrix.
	\return The Froebenius norm of the matrix : \f$ \sqrt{Tr\left[a a^\dagger\right]} \f$.
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline T frobenius(const Mat<r,r,T>& a)
	{
		T val = static_cast<T>(0);
		for(int i=0; i<r; i++)
			val += metaBinaryHermProductSum<r, typename ResultingType<T,T>::Type>(a.m+i, a.m+i*r, r, 1);
		return ::sqrt(val);
	}

	/**
	\brief Compute the \f$\mathcal{L}_2\f$ norm squared of the vector.
	\param v Input vector.
	\return The \f$\mathcal{L}_2\f$ norm squared of the vector.
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType normSquared(const Mat<r,1,T>& v)
	{
		return metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m);
	}

	/**
	\brief Compute the \f$\mathcal{L}_2\f$ norm squared of the vector.
	\param v Input vector.
	\return The \f$\mathcal{L}_2\f$ norm squared of the vector.
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType lengthSquared(const Mat<r,1,T>& v) // alias normSquared
	{
		return normSquared(v);
	}

	/**
	\brief Compute the \f$\mathcal{L}_2\f$ norm of the vector.
	\param v Input vector.
	\return The \f$\mathcal{L}_2\f$ norm of the vector.
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType norm(const Mat<r,1,T>& v)
	{
		return ::sqrt(metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m));
	}

	/**
	\brief Compute the inverse of the \f$\mathcal{L}_2\f$ norm of the vector.
	\param v Input vector.
	\return The inverse of the \f$\mathcal{L}_2\f$ norm of the vector : \f$ 1/||v|| \f$.
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType rnorm(const Mat<r,1,T>& v)
	{
		#ifdef __CUDACC__
			return ::rsqrt(metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m));
		#else
			return static_cast<typename Traits<T>::BaseType>(1)/::sqrt(metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m));
		#endif
	}

	/**
	\brief Compute the \f$\mathcal{L}_2\f$ norm of the vector.
	\param v Input vector.
	\return The \f$\mathcal{L}_2\f$ norm of the vector.
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType length(const Mat<r,1,T>& v) // alias norm
	{
		return norm(v);
	}

	/**
	\brief Compute the inverse of the \f$\mathcal{L}_2\f$ norm of the vector.
	\param v Input vector.
	\return The inverse of the \f$\mathcal{L}_2\f$ norm of the vector : \f$ 1/||v|| \f$.
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType rlength(const Mat<r,1,T>& v) // alias norm
	{
		return rnorm(v);
	}

	/**
	\brief Normalize the vector by its \f$\mathcal{L}_2\f$ norm.
	\param v Input vector.
	\return A vector of same direction as a but of unit length : \f$ a/||a|| \f$.
	\related Mat
	**/
	template<int r, typename T>
	__host__ __device__ inline Mat<r,1,T> normalize(const Mat<r,1,T>& v)
	{
		#ifdef __CUDACC__
			return v*::rsqrt(metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m));
		#else
			return v/norm(v);
		#endif
	}

	/**
	\brief Reflect the incoming vector dir around a normal (symmetry).
	\param dir Input vector, direction.
	\param normal Input vector, normal. This vector must be unitary, this constraint is not tested.
	\return The vector dir reflected around the normal as a physical bounce : \f$ 2 (dir \cdot normal) normal - dir \f$.

	Note that the result vector is opposite in direction to that of Kartet::rebound().

	\related Mat
	**/
	template<int r, typename T, typename U>
	__host__ __device__ inline Mat<r,1,typename ResultingType<T,U>::Type> reflect(const Mat<r,1,T>& dir, const Mat<r,1,U>& normal)
	{
		return (static_cast<typename ResultingType<T,U>::Type>(2)*dot(dir, normal))*normal - dir;
	}

	/**
	\brief Reflect the incoming vector dir around a normal (physical rebound).
	\param dir Input vector, direction.
	\param normal Input vector, normal. This vector must be unitary, this constraint is not tested.
	\return The vector dir reflected around the normal as a physical bounce : \f$ dir - 2 (dir \cdot normal) normal \f$.

	Note that the result vector is opposite in direction to that of Kartet::reflect().

	\related Mat
	**/
	template<int r, typename T, typename U>
	__host__ __device__ inline Mat<r,1,typename ResultingType<T,U>::Type> rebound(const Mat<r,1,T>& dir, const Mat<r,1,U>& normal)
	{
		return dir - (static_cast<typename ResultingType<T,U>::Type>(2)*dot(dir, normal))*normal;
	}

// Dimension specific :
	/**
	\brief Make a 2x2 matrix.
	\param m0 Element at \f$(0,0)\f$.
	\param m1 Element at \f$(1,0)\f$.
	\param m2 Element at \f$(0,1)\f$.
	\param m3 Element at \f$(1,1)\f$.
	\return A 2x2 matrix with the given elements and the highest rated type.
	\related Mat
	**/
	template<typename T0, typename T1, typename T2, typename T3>
	__host__ __device__ inline Mat<2,2,typename ResultingType4<T0,T1,T2,T3>::Type> mat2(const T0& m0, const T1& m2, const T2& m1, const T3& m3)
	{
		Mat<2,2,typename ResultingType4<T0,T1,T2,T3>::Type> m;
		m(0,0) = m0;
		m(1,0) = m1;
		m(0,1) = m2;
		m(1,1) = m3;
		return m;
	}

	/**
	\brief Make a 3x3 matrix.
	\param m0 Element at \f$(0,0)\f$.
	\param m1 Element at \f$(1,0)\f$.
	\param m2 Element at \f$(2,0)\f$.
	\param m3 Element at \f$(0,1)\f$.
	\param m4 Element at \f$(1,1)\f$.
	\param m5 Element at \f$(2,1)\f$.
	\param m6 Element at \f$(0,2)\f$.
	\param m7 Element at \f$(1,2)\f$.
	\param m8 Element at \f$(2,2)\f$.
	\return A 3x3 matrix with the given elements and the highest rated type.
	\related Mat
	**/
	template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
	__host__ __device__ inline Mat<3,3,typename ResultingType9<T0,T1,T2,T3,T4,T5,T6,T7,T8>::Type> mat3(const T0& m0, const T1& m2, const T2& m1, const T3& m3, const T4& m4, const T5& m5, const T6& m6, const T7& m7, const T8& m8)
	{
		Mat<3,3,typename ResultingType9<T0,T1,T2,T3,T4,T5,T6,T7,T8>::Type> m;
		m(0,0) = m0;
		m(1,0) = m1;
		m(2,0) = m2;
		m(0,1) = m3;
		m(1,1) = m4;
		m(2,1) = m5;
		m(0,2) = m6;
		m(1,2) = m7;
		m(2,2) = m8;
		return m;
	}

	/**
	\brief Make a 4x4 matrix.
	\param m0 Element at \f$(0,0)\f$.
	\param m1 Element at \f$(1,0)\f$.
	\param m2 Element at \f$(2,0)\f$.
	\param m3 Element at \f$(3,0)\f$.
	\param m4 Element at \f$(0,1)\f$.
	\param m5 Element at \f$(1,1)\f$.
	\param m6 Element at \f$(2,1)\f$.
	\param m7 Element at \f$(3,1)\f$.
	\param m8 Element at \f$(0,2)\f$.
	\param m9 Element at \f$(1,2)\f$.
	\param m10 Element at \f$(2,2)\f$.
	\param m11 Element at \f$(3,2)\f$.
	\param m12 Element at \f$(0,3)\f$.
	\param m13 Element at \f$(1,3)\f$.
	\param m14 Element at \f$(2,3)\f$.
	\param m15 Element at \f$(3,3)\f$.
	\return A 4x4 matrix with the given elements and the highest rated type.
	\related Mat
	**/
	template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15>
	__host__ __device__ inline Mat<4,4,typename ResultingType16<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15>::Type> mat4(const T0& m0, const T1& m2, const T2& m1, const T3& m3, const T4& m4, const T5& m5, const T6& m6, const T7& m7, const T8& m8,const T9& m9, const T10& m10, const T11& m11, const T12& m12, const T13& m13, const T14& m14, const T15& m15)
	{
		Mat<4,4,typename ResultingType16<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15>::Type> m;
		m(0,0) = m0;
		m(1,0) = m1;
		m(2,0) = m2;
		m(3,0) = m3;
		m(0,1) = m4;
		m(1,1) = m5;
		m(2,1) = m6;
		m(3,1) = m7;
		m(0,2) = m8;
		m(1,2) = m9;
		m(2,2) = m10;
		m(3,2) = m11;
		m(0,3) = m12;
		m(1,3) = m13;
		m(2,3) = m14;
		m(3,3) = m15;
		return m;
	}

	/**
	\brief Make the 2x2 identity matrix.
	\return The 2x2 identity matrix.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline Mat<2,2,T> identity2(void)
	{
		return Mat<2,2,T>::identity();
	}

	/**
	\brief Make the 3x3 identity matrix.
	\return The 3x3 identity matrix.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline Mat<3,3,T> identity3(void)
	{
		return Mat<3,3,T>::identity();
	}

	/**
	\brief Make the 4x4 identity matrix.
	\return The 4x4 identity matrix.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline Mat<4,4,T> identity4(void)
	{
		return Mat<4,4,T>::identity();
	}

	/**
	\brief Make a 2x2 rotation matrix.
	\param angle Rotation angle, in radians.
	\return The 2x2 rotation matrix for the given angle.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline Mat<2,2,T> rot2(const T& angle)
	{
		Mat<2,2,T> res;
		const T c = ::cos(angle),
			s = ::sin(angle);
		res(0,0) = c;
		res(1,0) = -s;
		res(0,1) = s;
		res(1,1) = c;
		return res;
	}

	/**
	\brief Make a 3x3 rotation matrix around the X axis.
	\param angle Rotation angle, in radians.
	\return The 3x3 rotation matrix for the given angle, around the X axis.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline Mat<3,3,T> rot3x(const T& angle)
	{
		Mat<3,3,T> res(static_cast<T>(0));
		const T	c = ::cos(angle),
			s = ::sin(angle);
		res(0,0) = static_cast<T>(1);
		res(1,1) = c;
		res(2,1) = -s;
		res(1,2) = s;
		res(2,2) = c;
		return res;
	}

	/**
	\brief Make a 3x3 rotation matrix around the Y axis.
	\param angle Rotation angle, in radians.
	\return The 3x3 rotation matrix for the given angle, around the Y axis.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline Mat<3,3,T> rot3y(const T& angle)
	{
		Mat<3,3,T> res(static_cast<T>(0));
		const T	c = ::cos(angle),
			s = ::sin(angle);
		res(0,0) = c;
		res(2,0) = -s;	
		res(1,1) = static_cast<T>(1);
		res(0,2) = s;
		res(2,2) = c;
		return res;
	}

	/**
	\brief Make a 3x3 rotation matrix around the Z axis.
	\param angle Rotation angle, in radians.
	\return The 3x3 rotation matrix for the given angle, around the Z axis.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline Mat<3,3,T> rot3z(const T& angle)
	{
		Mat<3,3,T> res(static_cast<T>(0));
		const T	c = ::cos(angle),
			s = ::sin(angle);
		res(0,0) = c;
		res(1,0) = -s;
		res(0,1) = s;
		res(1,1) = c;	
		res(2,2) = static_cast<T>(1);
		return res;
	}

	/**
	\brief Make a 3x3 rotation matrix around an arbitrary axis.
	\param axis Rotation axis, this vector must be normalized and this condition is not tested.
	\param angle Rotation angle, in radians.
	\return The 3x3 rotation matrix for the given angle, around the given axis.
	\related Mat
	**/
	template<typename T, typename U>
	__host__ __device__ inline Mat<3,3,typename ResultingType<T,U>::Type> rot3(const Mat<3,1,T>& axis, const U& angle)
	{
		Mat<3,3,typename ResultingType<T,U>::Type> res;
		const T	c = ::cos(angle),
			s = ::sin(angle),
			m = static_cast<T>(1)-c;
		const T& x = axis(0,0),
			 y = axis(1,0),
			 z = axis(2,0);
		res(0,0) = c+x*x*m;
		res(1,0) = x*y*m+z*s;
		res(2,0) = z*x*m-y*s;
		res(0,1) = x*y*m-z*s;
		res(1,1) = c+y*y*m;
		res(2,1) = z*y*m+x*s;
		res(0,2) = x*z*m+y*s;
		res(1,2) = y*z*m-x*s;
		res(2,2) = c+z*z*m;
		return res;
	}

	/**
	\brief Compute the rotation axis from a 3x3 rotation matrix.
	\param m A 3x3 rotation matrix (not verified).
	\return The rotation axis vector of length \f$ 2\sin(\theta) \f$.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline Mat<3, 1, T> rotationAxis(const Mat<3,3,T>& m)
	{
		Mat<3, 1, T> res;
		res(0,0) = m(2,1)-m(1,2);
		res(1,0) = m(0,2)-m(2,0);
		res(2,0) = m(1,0)-m(0,1);
		return res;
	}

	/**
	\brief Compute the rotation angle from a 3x3 rotation matrix.
	\param m A 3x3 rotation matrix (not verified).
	\return The rotation angle, in radians.
	\related Mat
	**/
	template<typename T>
	__host__ __device__ inline T rotationAngle(const Mat<3,3,T>& m)
	{
		return ::acos((trace(m)-static_cast<T>(1))/static_cast<T>(2));
	}

	/**
	\brief Compute the cross-product between two vectors.
	\param a Input vector.
	\param b Input vector.
	\return The cross-product vector \f$a \times b\f$.
	\related Mat
	**/
	template<typename T, typename U>
	__host__ __device__ inline Mat<3,1,typename ResultingType<T,U>::Type> cross(const Mat<3,1,T>& a, const Mat<3,1,U>& b)
	{
		Mat<3,1,typename ResultingType<T,U>::Type> c;
		c.m[0] = a.m[1]*b.m[2]-a.m[2]*b.m[1];
		c.m[1] = a.m[2]*b.m[0]-a.m[0]*b.m[2];
		c.m[2] = a.m[0]*b.m[1]-a.m[1]*b.m[0];
		return c;
	}

// Misc. :
	/**
	\brief Print the matrix to a standard stream.
	\param os Output standard stream.
	\param v Input matrix.
	\return A reference to the output stream.
	\related Mat

	Example :
	\code
		Mat4d m = Mat2d::identity();
		std::cout << "4x4 Identity : " << std::endl;
		std::cout << m << std::endl;
	\endcode
	**/
	template<int r, int c, typename T>
	__host__ std::ostream& operator<<(std::ostream& os, const Mat<r,c,T>& v)
	{
		const T zero = static_cast<T>(0),
			minusZero = -static_cast<T>(0);
		// Change the floatting point format if not specified :
		const std::ios_base::fmtflags originalFlags = os.flags();
		const bool forcedFloattingFormat = (os.flags() & std::ios_base::floatfield)!=0;
		if(!forcedFloattingFormat)
			os.setf(std::ios_base::scientific);

		const int precision = forcedFloattingFormat? os.precision() : 3;
		const int originalPrecision = os.precision(precision);
		const char fillCharacter = ' ';
		const char originalFill = os.fill(fillCharacter);
		const bool flag = !(os.flags() & std::ios_base::showpos);
		for(int i=0; i<r; i++)
		{
			os << " |";
			for(int j=0; j<(c-1); j++)
			{
				const T val = v(i,j);
				const bool f = (!Traits<T>::isComplex && softLargerEqual(val,zero)) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(val, minusZero));
				if(flag && f)
					os << fillCharacter;
				os << val << ", ";
			}
			const T val = v(i, c-1);
			const bool f = (!Traits<T>::isComplex && softLargerEqual(val,zero)) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(val, minusZero));
			if(flag && f)
				os << ' ';
			os << val << " | " << std::endl;
		}
		// Reset :
		os.precision(originalPrecision);
		os.fill(originalFill);
		os.flags(originalFlags);
		return os;
	}
}

#endif

