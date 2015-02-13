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

#ifndef __KARTET_MAIN_ARRAY__
#define __KARTET_MAIN_ARRAY__

// Includes :
	#include <iostream>
	#include <iomanip>
	#include <vector>
	#include <fstream>
	#include <string>	
	#include <cstring>
	#include "Exceptions.hpp"
	#include "TemplateSharedMemory.hpp"
	#include "MetaAlgorithm.hpp"
	#include "TypeTools.hpp"

namespace Kartet
{
	// Prototypes : 
		template<typename T>
		class Array;

		template<typename T>
		struct ExpressionContainer;

		template<class Op>
		struct NullaryExpression;

		template<typename T, template<typename> class Op >
		struct UnaryExpression;

		template<typename T, class Op >
		struct TransformExpression;

		template<typename T, class Op >
		struct LayoutReinterpretationExpression;

		template<typename T1, typename T2, template<typename,typename> class Op >
		struct BinaryExpression;

		template<typename T1, typename T2, typename T3, template<typename,typename,typename> class Op >
		struct TernaryExpression;

	typedef signed long long index_t;

	class Layout
	{
		private :
			index_t	numRows,		// Also Height
				numColumns,		// Also Width
				numSlices,		// Also Depth
				leadingColumns,		// Number of elements between the start of each column.
				leadingSlices,		// Number of elements between the start of each slice.
				offset;			// Starting point from the original position (informative parameter, not decisive).
							//  It will not be taken into account in most of the transformations.

		public :
			template<typename T>
			struct StaticContainer
			{
				typedef StaticAssert< SameTypes<void,T>::test > TestAssertion; // Must use the void type to access the container.
				static index_t numThreads;
				static const char fileHeader[];
			};

			// Constructors :
				__host__ __device__ inline Layout(index_t r, index_t c=1, index_t s=1, index_t lc=0, index_t ls=0, index_t o=0);
				__host__ __device__ inline Layout(const Layout& l);

			// Dimensions :
				__host__ __device__ inline index_t getNumElements(void) const;
				__host__ __device__ inline index_t getNumElementsPerSlice(void) const;
				__host__ __device__ inline index_t getNumRows(void) const;
				__host__ __device__ inline index_t getNumColumns(void) const;
				__host__ __device__ inline index_t getNumSlices(void) const;
				__host__ __device__ inline index_t getWidth(void) const;		// For convenience ?
				__host__ __device__ inline index_t getHeight(void) const;		// For convenience ?
				__host__ __device__ inline index_t getDepth(void) const;		// For convenience ?
				__host__ __device__ inline index_t getLeadingColumns(void) const;
				__host__ __device__ inline index_t getLeadingSlices(void) const;
				__host__ __device__ inline index_t getOffset(void) const;
				__host__ __device__ inline index_t setOffset(index_t newOffset);
				__host__ __device__ inline dim3 getDimensions(void) const;
				__host__ __device__ inline bool isMonolithic(void) const;
				__host__ __device__ inline bool isSliceMonolithic(void) const;
				__host__            inline void reinterpretLayout(index_t r, index_t c=1, index_t s=1);
				__host__            inline void reinterpretLayout(const Layout& other);
				__host__            inline void flatten(void);
				__host__            inline void vectorize(void);
				__host__ 	    inline std::vector<Layout> splitLayoutPages(index_t jBegin, index_t numVectors) const;
				__host__ __device__ inline bool sameLayoutAs(const Layout& other) const;
				__host__ __device__ inline bool sameSliceLayoutAs(const Layout& other) const;

			// Position tools :
			static		 __device__ inline index_t getI(void);
			static 		 __device__ inline index_t getJ(void);
			static		 __device__ inline index_t getK(void);
					template<typename TOut>
					 __device__ inline TOut getINorm(index_t i) const; // exclusive, from 0 to 1, (1 NOT INCLUDED)
					template<typename TOut>
					 __device__ inline TOut getJNorm(index_t j) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getKNorm(index_t k) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getINorm(void) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getJNorm(void) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getKNorm(void) const; // exclusive 
					 template<typename TOut>
					__device__ inline TOut getINormIncl(index_t i) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getJNormIncl(index_t j) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getKNormIncl(index_t k) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getINormIncl(void) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getJNormIncl(void) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getKNormIncl(void) const; // inclusive
				__host__ __device__ inline index_t getIClamped(index_t i) const;
				__host__ __device__ inline index_t getJClamped(index_t j) const;
				__host__ __device__ inline index_t getKClamped(index_t k) const;
				__host__ __device__ inline index_t getIWrapped(index_t i) const;
				__host__ __device__ inline index_t getJWrapped(index_t j) const;
				__host__ __device__ inline index_t getKWrapped(index_t k) const;
				__host__ __device__ inline index_t getIndex(index_t i, index_t j=0, index_t k=0) const;
					 __device__ inline index_t getIndex(void) const;
				__host__ __device__ inline index_t getIndicesFFTShift(index_t& i, index_t& j, index_t& k) const;
				__host__ __device__ inline index_t getIndexFFTShift(index_t i, index_t j, index_t k=0) const;
				__host__ __device__ inline index_t getIndicesFFTInverseShift(index_t& i, index_t& j, index_t& k) const;
				__host__ __device__ inline index_t getIndexFFTInverseShift(index_t i, index_t j, index_t k=0) const;
				__host__ __device__ inline index_t getIndexClampedToEdge(index_t i, index_t j, index_t k=0) const;
				__host__ __device__ inline index_t getIndexWarped(index_t i, index_t j, index_t k=0) const;
					 __device__ inline index_t getIndexFFTShift(void) const;
					 __device__ inline index_t getIndexFFTInverseShift(void) const;
				__host__ __device__ inline bool isInside(index_t i, index_t j, index_t k=0) const;
					 __device__ inline bool isInside(void) const;
				__host__ __device__ inline bool validRowIndex(index_t i) const;
				__host__ __device__ inline bool validColumnIndex(index_t j) const;
				__host__ __device__ inline bool validSliceIndex(index_t k) const;
				__host__ __device__ inline void unpackIndex(index_t index, index_t& i, index_t& j, index_t& k) const;

			// Other Tools : 
				__host__ 	    inline dim3 getBlockSize(void) const;
				__host__ 	    inline dim3 getNumBlock(void) const;
				__host__	    inline Layout getVectorLayout(void) const;
				__host__	    inline Layout getSliceLayout(void) const;
				__host__	    inline Layout getMonolithicLayout(void) const;

				template<class Op, typename T>
				__host__ void hostScan(T* ptr, const Op& op) const;

				__host__ static inline Layout readFromFile(std::fstream& file, int* typeIndex=NULL);
				__host__ static inline Layout readFromFile(const std::string& filename, int* typeIndex=NULL);
				__host__ inline void writeToFile(std::fstream& file, int typeIndex);
				__host__ inline void writeToFile(const std::string& filename, int typeIndex);
				template<typename T>
				__host__ inline void writeToFile(std::fstream& file);
				template<typename T>
				__host__ inline void writeToFile(const std::string& file);
				__host__ friend inline std::ostream& operator<<(std::ostream& os, const Layout& layout);
	};

	// Set the constant (modify <void> to change this behavior, e.g. Layout::StaticContainer<void>::numThreads = 1024;)
	template<typename T>
	index_t Layout::StaticContainer<T>::numThreads = 512;

	template<typename T>
	const char Layout::StaticContainer<T>::fileHeader[] = "KARTET01";

	// To compute on a specific layout : 
	#define COMPUTE_LAYOUT(x) <<<(x).getNumBlock(), (x).getBlockSize()>>>
 	
	template<typename T>
	class Accessor : public Layout
	{
		protected :
			T* devicePtr; // Does not include the offset.
			
				__host__ __device__ Accessor(index_t r, index_t c=1, index_t s=1, index_t lc=0, index_t ls=0, index_t o=0);
				__host__ __device__ Accessor(const Layout& layout);
				__host__ __device__ Accessor(T* ptr, index_t r, index_t c=1, index_t s=1, index_t lc=0, index_t ls=0, index_t o=0); // offset will not change the given ptr, and only is informative.
				__host__ __device__ Accessor(T* ptr, const Layout& layout);

		public :
				__host__            Accessor(const Array<T>& a);
				__host__ __device__ Accessor(const Accessor<T>& a);
			
			// Data Tools :
				__host__ __device__        T* getPtr(void) const;
				__host__ __device__        size_t getSize(void) const;
					 __device__ inline T& data(index_t i, index_t j, index_t k=0) const;
					 __device__ inline T& data(index_t p) const;
					 __device__ inline T& data(void) const;
					 __device__ inline T& dataInSlice(int k) const;
					 __device__ inline T& dataFFTShift(void) const;
					 __device__ inline T& dataFFTInverseShift(void) const;
				__host__                   T* getData(void) const;
				__host__                   void getData(T* ptr) const;
				__host__                   void setData(const T* ptr) const;
	
			// Layout tools :
				__host__	           const Layout& getLayout(void) const;
				__host__	           Accessor<T> element(index_t i, index_t j=0, index_t k=0) const;
				__host__	           Accessor<T> elements(index_t p, index_t numElements) const;
				__host__ 	           Accessor<T> vector(index_t j) const;
				__host__ 	           Accessor<T> endVector(void) const;
				__host__ 	           Accessor<T> vectors(index_t jBegin, index_t numVectors, index_t jStep=1) const;
				__host__ 	           Accessor<T> slice(index_t k=0) const;
				__host__ 	           Accessor<T> endSlice(void) const;
				__host__ 	           Accessor<T> slices(index_t kBegin, index_t numSlices, index_t kStep=1) const;
				__host__ 	           Accessor<T> subArray(index_t iBegin, index_t jBegin, index_t numRows, index_t numColumns) const;
				__host__ 	           std::vector< Accessor<T> > splitPages(index_t jBegin, index_t numVectors) const;

			// Assignment :
				template<typename TExpr>
				Accessor<T>& assign(const TExpr& expr, cudaStream_t stream=NULL);
				Accessor<T>& assign(const Accessor<T>& a, cudaStream_t stream=NULL);

			// Operator =
				template<typename TExpr>
				Accessor<T>& operator=(const TExpr& expr);
				Accessor<T>& operator=(const Accessor<T>& a);	 

			// Masked assignment : 
				template<typename TExprMask, typename TExpr>
				Accessor<T>& maskedAssignment(const TExprMask& exprMask, const TExpr& expr);

				template<class Op>
				__host__ void hostScan(const Op& op) const;

			// Other tools :
				template<typename TBis>
				__host__ friend std::ostream& operator<<(std::ostream& os, const Accessor<TBis>& A); // For debug, not for performance.
	};

	template<typename T>
	class Array : public Accessor<T>
	{
		public :
			__host__ Array(index_t r, index_t c=1, index_t s=1);
			__host__ Array(const Layout& layout);
			__host__ Array(const T* ptr, index_t r, index_t c=1, index_t s=1);
			__host__ Array(const T* ptr, const Layout& layout);
			__host__ Array(const Array<T>& a);
			template<typename TIn>
			__host__ Array(const Accessor<TIn>& a);
			__host__ Array(const std::string& filename, bool convert=true, size_t maxBufferSize=104857600); // 100 MB
			__host__ ~Array(void);

			// From Accessor<T>::Layout
			using Accessor<T>::Layout::getNumElements;
			using Accessor<T>::Layout::getNumRows;
			using Accessor<T>::Layout::getNumColumns;
			using Accessor<T>::Layout::getNumSlices;
			using Accessor<T>::Layout::getWidth;
			using Accessor<T>::Layout::getHeight;
			using Accessor<T>::Layout::getDepth;
			using Accessor<T>::Layout::getLeadingColumns;
			using Accessor<T>::Layout::getLeadingSlices;
			using Accessor<T>::Layout::getOffset;
			using Accessor<T>::Layout::setOffset;
			using Accessor<T>::Layout::getDimensions;
			using Accessor<T>::Layout::isMonolithic;
			using Accessor<T>::Layout::isSliceMonolithic;
			using Accessor<T>::Layout::reinterpretLayout;
			using Accessor<T>::Layout::flatten;
			using Accessor<T>::Layout::vectorize;
			using Accessor<T>::Layout::splitLayoutPages;
			using Accessor<T>::Layout::sameLayoutAs;
			using Accessor<T>::Layout::sameSliceLayoutAs;
			using Accessor<T>::Layout::getI;
			using Accessor<T>::Layout::getJ;
			using Accessor<T>::Layout::getK;
			using Accessor<T>::Layout::getIndex;
			using Accessor<T>::Layout::isInside;
			using Accessor<T>::Layout::validRowIndex;
			using Accessor<T>::Layout::validColumnIndex;
			using Accessor<T>::Layout::validSliceIndex;
			using Accessor<T>::Layout::getBlockSize;
			using Accessor<T>::Layout::getNumBlock;		

			// From Accessor<T>
			using Accessor<T>::getPtr;
			using Accessor<T>::getSize;
			using Accessor<T>::getData;
			using Accessor<T>::setData;
			using Accessor<T>::element;
			using Accessor<T>::elements;
			using Accessor<T>::vector;
			using Accessor<T>::endVector;
			using Accessor<T>::vectors;
			using Accessor<T>::slice;
			using Accessor<T>::endSlice;
			using Accessor<T>::slices;
			using Accessor<T>::subArray;
			using Accessor<T>::splitPages;
			using Accessor<T>::assign;
			using Accessor<T>::operator=;
			using Accessor<T>::maskedAssignment;
			using Accessor<T>::hostScan;
			
			Accessor<T>& accessor(void);
			void readFromFile(std::fstream& file, bool convert=true, size_t maxBufferSize=104857600); // 100 MB
			void readFromFile(const std::string& filename, bool convert=true, size_t maxBufferSize=104857600); // 100 MB
			void writeToFile(std::fstream& file, size_t maxBufferSize=104857600); // 100 MB
			void writeToFile(const std::string& filename, size_t maxBufferSize=104857600); // 100 MB
	};

} // namespace Kartet

// Include sub-definitions : 
	#include "Core/TypeTools.hpp"
	#include "Core/ArrayTools.hpp"
	#include "Core/ArrayExpressions.hpp"
	#include "Core/ArrayOperators.hpp"	

#endif

