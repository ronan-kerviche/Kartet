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
	\file    Container.hpp
	\brief   Pointer Tools.
	\author  R. Kerviche
	\date    April 29th 2016
**/

#ifndef __KARTET_CONTAINER__
#define __KARTET_CONTAINER__

	#include "Core/Exceptions.hpp"
	#include "Core/Traits.hpp"

/// Kartet main namespace.
namespace Kartet
{
	/**
	\brief Container object.

	Contains a dynamically allocated object and assumes its ownership similarly to C++11's unique_ptr. <b>However, it does not support the allocation of arrays</b> and will always perform the de-allocation with delete (not delete[]).

\code
Kartet::Container<Kartet::Array<double> > a;
// Populate with a pointer :
a = new Kartet::Array<double>(5,5);
// Use as a typical pointer :
*a = Kartet::IndexI();
std::cout << "Layout : " << a->layout() << std::endl;
// Replace, the previous object will be de-allocated automatically :
a = new Kartet::Array<double>(10,10);
// The object is automatically cleared at the end of the scope.
\endcode
	**/
	template<typename T>
	class Container
	{
		private :
			T* ptr;
		public :
			__host__ Container(void);
			__host__ Container(T* _ptr);
			__host__ ~Container(void);

			__host__ void reset(void);
			__host__ void replace(Container<T>& c);
			__host__ T* take(void);
			__host__ operator bool(void) const;
			__host__ bool isNull(void) const;
			__host__ Container<T>& operator=(T* _ptr);
			__host__ Container<T>& operator=(Container<T>& c);
			__host__ T& operator*(void) const;
			__host__ T* operator->(void) const;
	};

	/**
	\brief Container constructer.
	**/
	template<typename T>
	__host__ Container<T>::Container(void)
	 : ptr(NULL)
	{ }

	/**
	\brief Container constructor.
	\param _ptr Pointer to data. The ownership will be transfered to the container.
	**/
	template<typename T>
	__host__ Container<T>::Container(T* _ptr)
	 : ptr(_ptr)
	{ }

	/**
	\brief Container destructor.
	**/
	template<typename T>
	__host__ Container<T>::~Container(void)
	{
		reset();
	}

	/**
	\brief Release the data and reset the pointer.
	**/
	template<typename T>
	__host__ void Container<T>::reset(void)
	{
		delete ptr;
		ptr = NULL;
	}

	/**
	\brief Take ownership of data from another container. The previous data is released.
	**/
	template<typename T>
	__host__ void Container<T>::replace(Container<T>& c)
	{
		reset();
		ptr = c.take();
	}

	/**
	\brief Relieve the container from the ownership of the data. The pointer is reset.
	\return The pointer to the data.
	**/
	template<typename T>
	__host__ T* Container<T>::take(void)
	{
		T* r = ptr;
		ptr = NULL;
		return r;
	}	

	/**
	\brief Test if the pointer is NULL.
	\return True if the pointer is NULL.
	**/
	template<typename T>
	__host__ bool Container<T>::isNull(void) const
	{
		return (ptr==NULL);
	}

	/**
	\brief Test if the pointer is not NULL.
	\return True if the pointer is not NULL.
	**/
	template<typename T>
	__host__ Container<T>::operator bool(void) const
	{
		return (ptr!=NULL);
	}

	/**
	\brief Set the data tracked by this container. The previous data will be released.
	\param _ptr Replacement pointer. This container will take its ownership.
	\return This.
	**/
	template<typename T>
	__host__ Container<T>& Container<T>::operator=(T* _ptr)
	{
		reset();
		ptr = _ptr;
		return (*this);
	}

	/**
	\brief Dereference the data tracked by this container.
	\throw Kartet::NullPointer if the the pointer is invalid.
	\return A reference to the tracked object.
	**/
	template<typename T>
	__host__ T& Container<T>::operator*(void) const
	{
		if(ptr==NULL)
			throw NullPointer;
		return (*ptr);
	}

	/**
	\brief Get the pointer to the data.
	\throw Kartet::NullPointer if the the pointer is invalid.
	\return The pointer to the data.
	**/
	template<typename T>
	__host__ T* Container<T>::operator->(void) const
	{
		if(ptr==NULL)
			throw NullPointer;
		return ptr;
	}

	/**
	\brief Swap the content of two containers.
	\param a First container.
	\param b Second container.
	**/
	template<typename T>
	__host__ void swap(Container<T>& a, Container<T>& b)
	{
		T *aPtr = a.take(),
		  *bPtr = b.take();
		a = bPtr;
		b = aPtr;
	}

} // namespace Kartet

#endif

