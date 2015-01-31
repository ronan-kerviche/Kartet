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

#ifndef __KARTET_META_LIST__
#define __KARTET_META_LIST__

	// Symbols : 
	struct Void
	{};

	// Lists of types :
	template< typename H, typename Q>
	struct TypeList
	{
		typedef H Head;
		typedef Q Queue;
	};

	typedef TypeList<Void,Void> TypeListVoid;

	// Length :
	template <class>
	struct Length;

	template <>
	struct Length<Void>
	{
		enum
		{
			Value = 0
		};
	};

	template <class Head, class Queue>
	struct Length< TypeList<Head, Queue> >
	{
		enum
		{
			Value = 1 + Length<Queue>::Value
		};
	};

	// Get a type by its index :
	template <class TList, unsigned int i>
	struct TypeAtIndex;

	template <class Head, class Queue>
	struct TypeAtIndex< TypeList<Head,Queue>, 0 >
	{
		typedef Head TValue;
	};

	template <class Tete, class Queue, unsigned int i>
	struct TypeAtIndex<TypeList<Tete,Queue>,i>
	{
		typedef typename TypeAtIndex<Queue, i-1>::TValue TValue;
	};

	//Find the index of an element :
	template <class TList, class H>
	struct GetIndex;

	template <class H>
	struct GetIndex <Void, H>
	{
		enum
		{
			Value = -1
		};
	};

	template <class Queue, class H>
	struct GetIndex <TypeList<H, Queue>, H>
	{
		enum
		{
			Value = 0
		};
	};

	template <class Head, class Queue, class H>
	struct GetIndex <TypeList<Head, Queue>, H>
	{
		private: // temporary value
			enum
			{
				Tmp = GetIndex<Queue, H>::Value
			};
		public: // Index Value :
			enum
			{
				Value = Tmp == -1? -1 : 1 + Tmp // if(not found) -1 else 1+GetIndex(Queue)
			};
	};

	// Test the presence :
	template <class TList, class H>
	struct Belongs;

	template <class Head, class Queue, class H>
	struct Belongs <TypeList<Head, Queue>, H>
	{
		static const bool Value = GetIndex< TypeList<Head, Queue>, H>::Value != -1;
	};

	// Add an element at the back list : 
	template <class TList, class T>
	struct PushElement;

	template <class T>
	struct PushElement <Void, T>
	{
		typedef TypeList<T,Void> TValue;
	};

	template <class Head, class Queue, class T>
	struct PushElement < TypeList<Head, Queue>, T>
	{
		typedef typename PushElement<Queue, T>::TValue Tmp;
		typedef TypeList<Head, Tmp > TValue;
	};

	// Union of two lists :
	template <class TList1, class TList2>
	struct Union;

	template <class Head, class Queue>
	struct Union <Void, TypeList<Head, Queue> >
	{
		typedef TypeList<Head, Queue> TValue;
	};

	template <class Head, class Queue, class TList>
	struct Union< TypeList<Head, Queue>, TList>
	{
		typedef typename Union<Queue, TList>::TValue Tmp;
		typedef TypeList<Head, Tmp > TValue;
	};

	// Add an element in the front list : 
	template <class TList, class T>
	struct PushFrontElement
	{
		typedef TypeList<T,TList> TValue;
	};

	// Single version :
	template <class TList, class T>
	struct PushElementSingle;

	template <class T>
	struct PushElementSingle <Void, T>
	{
		typedef TypeList<T,Void> TValue;
	};

	template <class Head, class Queue, class T>
	struct PushElementSingle < TypeList<Head, Queue>, T>
	{
		typedef typename  PushElementSingle<Queue, T>::TValue Tmp;
		typedef TypeList<Head, Tmp > TValue;
	};

	template <class Queue, class T>
	struct PushElementSingle < TypeList<T, Queue>, T>
	{
		typedef TypeList<T, Queue> TValue;
	};

	// Drop front :
	template <class TList>
	struct DropFront;

	template <class Head, class Queue>
	struct DropFront < TypeList<Head, Queue> >
	{
		typedef Queue TValue;
	};

	// Drop back :
	template <class TList>
	struct DropBack;

	template <class Head, class Queue>
	struct DropBack < TypeList<Head, Queue> >
	{
		typedef typename DropBack<Queue>::TValue Tmp;
		typedef TypeList<Head, Tmp > TValue;
	};

	template <class Head>
	struct DropBack < TypeList<Head, Void> >
	{
		typedef Void TValue;
	};

	// Drop first element like :
	template <class TList, class Elmt>
	struct DropFirstElement;

	template <class Head, class Queue, class Elmt>
	struct DropFirstElement< TypeList<Head, Queue>, Elmt>
	{
		typedef typename DropFirstElement<Queue, Elmt>::TValue Tmp;
		typedef TypeList<Head, Tmp > TValue;
	};

	template < class Queue, class Elmt>
	struct DropFirstElement< TypeList<Elmt, Queue>, Elmt>
	{
		typedef Queue TValue;
	};

	// Drop all elements like :
	template <class TList, class Elmt>
	struct DropAllElements;

	template <class Head, class Queue, class Elmt>
	struct DropAllElements< TypeList<Head, Queue>, Elmt>
	{
		typedef typename DropFirstElement<Queue, Elmt>::TValue Tmp;
		typedef TypeList<Head, Tmp > TValue;
	};

	template < class Queue, class Elmt>
	struct DropAllElements< TypeList<Elmt, Queue>, Elmt>
	{
		typedef DropAllElements<Queue, Elmt> TValue;
	};

#endif

