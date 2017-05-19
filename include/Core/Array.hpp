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
	\file    Array.hpp
	\brief   Array classes definitions.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_MAIN_ARRAY__
#define __KARTET_MAIN_ARRAY__

// Includes :
	#include <iostream>
	#include <iomanip>
	#include <vector>
	#include <fstream>
	#include <string>	
	#include <cstring>
	#include "Core/LibTools.hpp"
	#include "Core/Exceptions.hpp"
	#include "Core/Meta.hpp"
	#include "Core/TemplateSharedMemory.hpp"

/**
\def KARTET_DEFAULT_LOCATION
\brief Default location when unspecified for an array (see Kartet::Location).
**/
#ifndef KARTET_DEFAULT_LOCATION
	#ifdef __CUDACC__
		#define KARTET_DEFAULT_LOCATION (Kartet::DeviceSide)
	#else
		#define KARTET_DEFAULT_LOCATION (Kartet::HostSide)
	#endif
#endif

/**
\def KARTET_DEFAULT_COMPUTE_LAYOUT
\brief Default computation layout.

See also Kartet::ComputeLayout.
**/
#ifndef KARTET_DEFAULT_COMPUTE_LAYOUT
	#define KARTET_DEFAULT_COMPUTE_LAYOUT (Kartet::StripesLayout)
#endif

/**
\def KARTET_DEFAULT_BUFFER_SIZE
\brief Default buffer size for transfer operations.
**/
#ifndef KARTET_DEFAULT_BUFFER_SIZE
	// 100 MB :
	#define KARTET_DEFAULT_BUFFER_SIZE (104857600)
#endif

/// Kartet main namespace.
namespace Kartet
{
	// Flags :
		/// Describe the location of an object.
		enum Location
		{
			/// Host side (typically CPU/RAM).
			HostSide,
			/// Device side (typically GPU/VRAM).
			DeviceSide,
			/// Either Host or Device side (analytical expression not tied to hardware, cannot be used to declare an accessor or an array).
			AnySide
		};

		/// Data transfer directions.
		enum Direction
		{
			/// Host to host.
			HostToHost,
			/// Host to device.
			HostToDevice,
			/// Device to device.
			DeviceToDevice,
			/// Device to host.
			DeviceToHost
		};

		inline Direction getDirection(const Location& from, const Location& to);
		#ifdef __CUDACC__
			inline cudaMemcpyKind getCudaDirection(const Direction& direction);
		#endif

		/// Computation Layout.
		enum ComputeLayout
		{
			/// Stripes layout.
			StripesLayout,
			/// Blocks layout.
			BlocksLayout
		};

	// Prototypes : 
		template<typename, Location>
		class Array;

		template<typename T>
		struct ExpressionContainer;

		template<class Op>
		struct NullaryExpression;

		template<typename T, template<typename> class Op>
		struct UnaryExpression;

		template<typename T, class Op>
		struct TransformExpression;

		template<typename T, class Op>
		struct LayoutReinterpretationExpression;

		template<typename T1, typename T2, template<typename,typename> class Op>
		struct BinaryExpression;

		template<typename T1, typename T2, template<typename> class Op>
		struct ShuffleExpression;

		template<typename T1, typename T2, typename T3, template<typename,typename,typename> class Op>
		struct TernaryExpression;

	/// 64 bits indexing type.
	typedef signed long long index64_t;
	/// 32 bits indexing type.
	typedef signed int index32_t;
	/**
	\typedef TYPE index_t
	\brief Default indexing type.
	**/
	#ifdef KARTET_USE_64BITS_INDEXING
		typedef signed long long index_t;
	#else
		typedef signed int index_t;
	#endif

	/**
	\brief Array layout description.

	Contains all the information about the dimensions of an array.
	**/
	class Layout
	{
		private :
			index_t	nRows,			// Number of rows, also Height
				nColumns,		// Number of columns, also width
				nSlices,		// Number of slices, also depth
				sColumns,		// Columns stride, the number of elements between the start of each column.
				sSlices,		// Slices stride, the number of elements between the start of each slice.
				ofs;			// Offset, starting point from the original position (informative parameter, not decisive).
							//  It will not be taken into account in most of the transformations.

		public :
			template<typename T>
			struct StaticContainer
			{
				// Functions using this data cannot be device'd ...
				STATIC_ASSERT_VERBOSE((IsSame<void,T>::value), STATIC_CONTAINER_MUST_HAVE_VOID_PARAMETER)
				static index_t 	numThreads,
						warpSize,
						maxXThreads,
						maxYThreads,
						maxZThreads,
						maxBlockRepetition;
				static const char streamHeader[];
				static size_t streamHeaderLength();
			};

			// Constructors :
				__host__ __device__ inline Layout(index_t r, index_t c, index_t s, index_t cs, index_t ss, index_t o=0);
				__host__ __device__ inline Layout(index_t r, index_t c=1, index_t s=1);
				__host__ __device__ inline Layout(const Layout& l);

			// Dimensions :
				__host__ __device__ inline bool isValid(void) const;
				__host__ __device__ inline index_t numElements(void) const;
				__host__ __device__ inline index_t numElementsPerSlice(void) const;
				__host__ __device__ inline index_t numElementsPerFragment(void) const; // per monolithic fragment
				__host__ __device__ inline index_t numRows(void) const;
				__host__ __device__ inline index_t numColumns(void) const;
				__host__ __device__ inline index_t numSlices(void) const;
				__host__ __device__ inline index_t numFragments(void) const;
				__host__ __device__ inline int numDims(void) const;
				__host__ __device__ inline int numDimsPacked(void) const;
				__host__ __device__ inline index_t width(void) const;		// For convenience ?
				__host__ __device__ inline index_t height(void) const;		// For convenience ?
				__host__ __device__ inline index_t depth(void) const;		// For convenience ?
				__host__ __device__ inline index_t columnsStride(void) const;
				__host__ __device__ inline index_t slicesStride(void) const;
				__host__ __device__ inline index_t offset(void) const;
				__host__ __device__ inline index_t setOffset(index_t newOffset);
				__host__ __device__ inline dim3 dimensions(void) const;
				__host__ __device__ inline dim3 strides(void) const;
				__host__ __device__ inline bool isMonolithic(void) const;
				__host__ __device__ inline bool isSliceMonolithic(void) const;
				__host__            inline void reinterpretLayout(index_t r, index_t c=1, index_t s=1);
				__host__            inline void reinterpretLayout(const Layout& other);
				__host__            inline void flatten(void);
				__host__	    inline void stretch(void);
				__host__            inline void vectorize(void);
				__host__	    inline void squeeze(void);
				__host__ 	    inline std::vector<Layout> splitLayoutColumns(index_t jBegin, index_t nColumns) const;
				__host__ 	    inline std::vector<Layout> splitLayoutSlices(index_t kBegin, index_t nSlices) const;
				__host__ 	    inline std::vector<Layout> splitLayoutSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, index_t nRows=0, index_t nColumns=0, index_t nSlices=0) const;
				__host__ 	    inline std::vector<Layout> splitLayoutSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, const Layout& layout) const;
				__host__ __device__ inline bool sameLayoutAs(const Layout& other) const;
				__host__ __device__ inline bool sameMonolithicLayoutAs(const Layout& other) const;
				__host__ __device__ inline bool sameSliceLayoutAs(const Layout& other) const;
				__host__ __device__ inline bool sameMonolithicSliceLayoutAs(const Layout& other) const;

			// Position tools :
			#ifdef __CUDACC__
			static		 __device__ inline index_t getI(void);
			static 		 __device__ inline index_t getJ(void);
			static		 __device__ inline index_t getK(void);
			static		 __device__ inline index_t getI(dim3 blockRepetition);
			static 		 __device__ inline index_t getJ(dim3 blockRepetition);
			static		 __device__ inline index_t getK(dim3 blockRepetition);
			#endif
					template<typename TOut>
				__host__ __device__ inline TOut getINorm(index_t i) const; // exclusive, from 0 to 1, (1 NOT INCLUDED)
					template<typename TOut>
				__host__ __device__ inline TOut getJNorm(index_t j) const; // exclusive
					template<typename TOut>
				__host__ __device__ inline TOut getKNorm(index_t k) const; // exclusive
			#ifdef __CUDACC__
					template<typename TOut>
					 __device__ inline TOut getINorm(void) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getJNorm(void) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getKNorm(void) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getINorm(dim3 blockRepetition) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getJNorm(dim3 blockRepetition) const; // exclusive
					template<typename TOut>
					 __device__ inline TOut getKNorm(dim3 blockRepetitio) const; // exclusive
			#endif
					 template<typename TOut>
				__host__ __device__ inline TOut getINormIncl(index_t i) const; // inclusive
					template<typename TOut>
				__host__ __device__ inline TOut getJNormIncl(index_t j) const; // inclusive
					template<typename TOut>
				__host__ __device__ inline TOut getKNormIncl(index_t k) const; // inclusive
			#ifdef __CUDACC__
					template<typename TOut>
					 __device__ inline TOut getINormIncl(void) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getJNormIncl(void) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getKNormIncl(void) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getINormIncl(dim3 blockRepetition) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getJNormIncl(dim3 blockRepetition) const; // inclusive
					template<typename TOut>
					 __device__ inline TOut getKNormIncl(dim3 blockRepetition) const; // inclusive
			#endif
				__host__ __device__ inline index_t getIClamped(index_t i) const;
				__host__ __device__ inline index_t getJClamped(index_t j) const;
				__host__ __device__ inline index_t getKClamped(index_t k) const;
				__host__ __device__ inline index_t getIWrapped(index_t i) const;
				__host__ __device__ inline index_t getJWrapped(index_t j) const;
				__host__ __device__ inline index_t getKWrapped(index_t k) const;
				__host__ __device__ inline index_t getIndex(index_t i, index_t j=0, index_t k=0) const;
				__host__ __device__ inline index_t getPosition(index_t i, index_t j=0, index_t k=0) const;
			#ifdef __CUDACC__
					 __device__ inline index_t getIndex(void) const;
					 __device__ inline index_t getPosition(void) const;
					 __device__ inline index_t getIndex(dim3 repetition) const;
					 __device__ inline index_t getPosition(dim3 repetition) const;
			#endif
				__host__ __device__ inline index_t getPositionFFTShift(index_t& i, index_t& j, index_t& k) const;
				__host__ __device__ inline index_t getPositionFFTInverseShift(index_t& i, index_t& j, index_t& k) const;
				__host__ __device__ inline index_t getPositionClampedToEdge(index_t& i, index_t& j, index_t& k) const;
				__host__ __device__ inline index_t getPositionWarped(index_t& i, index_t& j, index_t& k) const;
			#ifdef __CUDACC__
					 __device__ inline index_t getPositionFFTShift(void) const;
					 __device__ inline index_t getPositionFFTInverseShift(void) const;
					 __device__ inline index_t getPositionFFTShift(dim3 blockRepetition) const;
					 __device__ inline index_t getPositionFFTInverseShift(dim3 blockRepetition) const;
			#endif
				__host__ __device__ inline bool isInside(index_t i, index_t j, index_t k=0) const;	
			#ifdef __CUDACC__
					 __device__ inline bool isInside(void) const;
					 __device__ inline bool isInside(dim3 blockRepetition) const;
			#endif
				__host__ __device__ inline bool isRowValid(index_t i) const;
				__host__ __device__ inline bool isColumnValid(index_t j) const;
				__host__ __device__ inline bool isSliceValid(index_t k) const;
				__host__ __device__ inline void unpackIndex(index_t index, index_t& i, index_t& j, index_t& k) const;
				__host__ __device__ inline void unpackPosition(index_t index, index_t& i, index_t& j, index_t& k) const;
				__host__ __device__ inline void moveToNext(index_t& i, index_t& j, index_t& k) const;
				__host__ __device__ inline void moveToNext(index_t& j, index_t& k) const;

			// Other Tools :
				template<ComputeLayout cl>
				__host__	    inline dim3 blockSize(void) const;
				__host__ 	    inline dim3 blockSize(void) const;
				template<ComputeLayout cl>
				__host__	    inline dim3 numBlocks(void) const;
				__host__ 	    inline dim3 numBlocks(void) const;
				template<ComputeLayout cl>
				__host__	    inline void computeLayout(dim3& blockSize, dim3& numBlocks, dim3& repeatBlock) const;
				__host__	    inline void computeLayout(dim3& blockSize, dim3& numBlocks, dim3& repeatBlock) const;
				__host__	    inline Layout columnLayout(const bool& includeSlices=false) const;
				__host__	    inline Layout sliceLayout(void) const;
				__host__	    inline Layout monolithicLayout(void) const;

				template<class Op, typename T>
				__host__ void singleScan(T* ptr, const Op& op) const;

				template<class Op, typename T>
				__host__ static void dualScan(const Layout& layoutA, T* ptrA, const Layout& layoutB, T* ptrB, const Op& op);

				__host__ static inline Layout readFromStream(std::istream& stream, int* typeIndex=NULL, bool* isComplex=NULL);
				__host__ static inline Layout readFromFile(const std::string& filename, int* typeIndex=NULL, bool* isComplex=NULL);
				__host__ inline void writeToStream(std::ostream& stream, const int typeIndex, const bool isComplex);
				__host__ inline void writeToFile(const std::string& filename, const int typeIndex, const bool isComplex);
				template<typename T>
				__host__ inline void writeToStream(std::ostream& stream);
				template<typename T>
				__host__ inline void writeToFile(const std::string& file);
				__host__ friend inline std::ostream& operator<<(std::ostream& os, const Layout& layout);
	};

	// Set the constant (modify <void> to change this behavior, e.g. Layout::StaticContainer<void>::numThreads = 1024;)
	/**
	\var Layout::StaticContainer<T>::numThreads
	\related Kartet::Layout
	\brief Number of CUDA threads per block (global setting).

	Number of threads per block can be changed as follow : 
	\code
	Kartet::Layout::StaticContainer<void>::numThreads = 512;
	\endcode

	See also Kartet::initialize().
	**/
	template<typename T>
	index_t Layout::StaticContainer<T>::numThreads = 512;
	
	/**
	\var Layout::StaticContainer<T>::warpSize
	\related Kartet::Layout
	\brief Size of threads warps.

	Use the initialize method to set automatically.

	See also Kartet::initialize().
	**/
	template<typename T>
	index_t Layout::StaticContainer<T>::warpSize = 32;

	/**
	\var Layout::StaticContainer<void>::maxXThreads
	\related Kartet::Layout
	\brief Maximum number of CUDA threads in the X direction.

	Use the initialize method to set automatically.

	See also Kartet::initialize().
	**/
	template<typename T>
	index_t Layout::StaticContainer<T>::maxXThreads = 1024;

	/**
	\var Layout::StaticContainer<void>::maxYThreads
	\related Kartet::Layout
	\brief Maximum number of CUDA threads in the Y direction.

	Use the initialize method to set automatically.

	See also Kartet::initialize().
	**/
	template<typename T>
	index_t Layout::StaticContainer<T>::maxYThreads = 1024;


	/**
	\var Layout::StaticContainer<void>::maxZThreads
	\related Kartet::Layout
	\brief Maximum number of CUDA threads in the Z direction.

	See also Kartet::initialize().
	**/
	template<typename T>
	index_t Layout::StaticContainer<T>::maxZThreads = 64; // For some reason, CUDA allows less threads in the Z direction, operations on native arrays with only X-Z or Y-Z dimensions will be somewhat slower

	/**
	\var Layout::StaticContainer<T>::maxBlockRepetition
	\related Kartet::Layout
	\brief Maximum number of block repetitions.
	**/
	template<typename T>
	index_t Layout::StaticContainer<T>::maxBlockRepetition = 64;

	/**
	\var Layout::StaticContainer<void>::streamHeader
	\related Kartet::Layout
	\brief Stream header written by Kartet.

	Read : 
	\code
	const char* streamHeader = Kartet::Layout::StaticContainer<void>::streamHeader;
	\endcode
	**/
	template<typename T>
	const char Layout::StaticContainer<T>::streamHeader[] = "KARTET02";

	// To compute on a specific layout :
	#ifdef __CUDACC__
		/**
		\def COMPUTE_LAYOUT_STRIPES
		\brief Compute layout descriptor tool (CUDA), stripes version.

		Call a global function with :
\code
functionName COMPUTE_LAYOUT_STRIPES(myLayout) (...)
\endcode
		**/
		#define COMPUTE_LAYOUT_STRIPES(x) <<<(x).numBlocks<StripesLayout>(), (x).blockSize<StripesLayout>()>>>

		/**
		\def COMPUTE_LAYOUT_STREAM_STRIPES
		\brief Compute layout descriptor tool (CUDA), stripes version.

		Call a global function with :
\code
functionName COMPUTE_LAYOUT_STREAM_STRIPES(myLayout, stream) (...)
\endcode
		**/
		#define COMPUTE_LAYOUT_STREAM_STRIPES(x, s) <<<(x).numBlocks<StripesLayout>(), (x).blockSize<StripesLayout>(), 0, (s)>>>

		/**
		\def COMPUTE_LAYOUT_BLOCKS
		\brief Compute layout descriptor tool (CUDA), blocks version.

		Call a global function with :
\code
functionName COMPUTE_LAYOUT_BLOCKS(myLayout) (...)
\endcode
		**/
		#define COMPUTE_LAYOUT_BLOCKS(x) <<<(x).numBlocks<BlocksLayout>(), (x).blockSize<BlocksLayout>()>>>

/**
		\def COMPUTE_LAYOUT_STREAM_BLOCKS
		\brief Compute layout descriptor tool (CUDA), blocks version.

		Call a global function with :
\code
functionName COMPUTE_LAYOUT_STREAM_BLOCKS(myLayout, stream) (...)
\endcode
		**/
		#define COMPUTE_LAYOUT_STREAM_BLOCKS(x, s) <<<(x).numBlocks<BlocksLayout>(), (x).blockSize<BlocksLayout>(), 0, (s)>>>

		/**
		\def COMPUTE_LAYOUT
		\brief Compute layout descriptor tool (CUDA).

		Call a global function with :
\code
functionName COMPUTE_LAYOUT(myLayout) (...)
\endcode
		**/
		#define COMPUTE_LAYOUT(x) <<<(x).numBlocks(), (x).blockSize()>>>

/**
		\def COMPUTE_LAYOUT_STREAM
		\brief Compute layout descriptor tool (CUDA).

		Call a global function with :
\code
functionName COMPUTE_LAYOUT_STREAM(myLayout, stream) (...)
\endcode
		**/
		#define COMPUTE_LAYOUT_STREAM(x, s) <<<(x).numBlocks(), (x).blockSize(), 0, (s)>>>
	#endif

	/**
	\brief Accessor to array data.
	\tparam T Type of the array (either a primitive type or a complex type).
	\tparam l Location of the data (see Kartet::Location).

	Defines a virtual array tied to some portion memory. Can be used to modify temporarily the layout of an array or access parts of it (R/W).
	Can also be used as a proxy to memory space not managed by the library (no release will be performed).

	See also \ref OperatorsGroup and \ref FunctionsGroup for available functions.
	**/
	template<typename T, Location l=KARTET_DEFAULT_LOCATION>
	class Accessor : public Layout
	{
		protected :
			T* ptr; // Already includes the offset.
			
				__host__ __device__ Accessor(index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o=0);
				__host__ __device__ Accessor(index_t r, index_t c=1, index_t s=1);
				__host__ __device__ Accessor(const Layout& layout);

		public :
			// Constructors :
				__host__ __device__ Accessor(T* ptr, index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o=0); // offset will not change the given ptr, and only is informative.
				__host__ __device__ Accessor(T* ptr, index_t r, index_t c=1, index_t s=1);
				__host__ __device__ Accessor(T* ptr, const Layout& layout);
				__host__	    Accessor(const Array<T,l>& a);
				__host__ __device__ Accessor(const Accessor<T,l>& a);	
					
			// Data Tools :
				__host__ __device__ 	   Location location(void) const;
				__host__ __device__        T* dataPtr(void) const;
				__host__ __device__	   bool isNull(void) const;
				__host__ __device__        size_t size(void) const;
				__host__ __device__ inline T& data(index_t i, index_t j, index_t k=0) const;
				__host__ __device__ inline T& data(index_t p) const;
				#ifdef __CUDACC__
					 __device__ inline T& data(void) const;
					 __device__ inline T& dataInSlice(int k) const;
					 __device__ inline T& dataFFTShift(void) const;
					 __device__ inline T& dataFFTInverseShift(void) const;
					 __device__ inline T& data(dim3 blockRepetition) const;
					 __device__ inline T& dataInSlice(int k, dim3 blockRepetition) const;
					 __device__ inline T& dataFFTShift(dim3 blockRepetition) const;
					 __device__ inline T& dataFFTInverseShift(dim3 blockRepetition) const;

				#endif
				__host__                   T* getData(void) const;
				__host__                   void getData(T* ptr, const Location lout=HostSide) const;
				__host__                   void setData(const T* ptr, const Location lin=HostSide) const;
				__host__                   void readFromStream(std::istream& stream, bool convert=true, const size_t maxBufferSize=KARTET_DEFAULT_BUFFER_SIZE, const bool skipHeader=false, int sourceTypeIndex=GetTypeIndex<T>::index, bool sourceIsComplex=Traits<T>::isComplex);
				__host__                   void readFromFile(const std::string& filename, bool convert=true, const size_t maxBufferSize=KARTET_DEFAULT_BUFFER_SIZE);
				__host__                   void writeToStream(std::ostream& stream, const size_t maxBufferSize=KARTET_DEFAULT_BUFFER_SIZE);
				__host__                   void writeToFile(const std::string& filename, const size_t maxBufferSize=KARTET_DEFAULT_BUFFER_SIZE);
	
			// Layout tools :
				__host__	           const Layout& layout(void) const;
				__host__	           Accessor<T,l> element(index_t i, index_t j=0, index_t k=0) const;
				__host__	           Accessor<T,l> elements(index_t p, index_t numElements) const;
				__host__	           Accessor<T,l> elements(void) const;
				__host__	           Accessor<T,l> row(index_t i) const; 
				__host__	           Accessor<T,l> endRow(index_t i) const;
				__host__	           Accessor<T,l> rows(index_t iBegin, index_t r) const;
				__host__ 	           Accessor<T,l> column(index_t j) const;
				__host__ 	           Accessor<T,l> endColumn(void) const;
				__host__ 	           Accessor<T,l> columns(index_t jBegin, index_t c, index_t jStep=1) const;
				__host__ 	           Accessor<T,l> slice(index_t k=0) const;
				__host__ 	           Accessor<T,l> endSlice(void) const;
				__host__ 	           Accessor<T,l> slices(index_t kBegin, index_t s, index_t kStep=1) const;
				__host__ 	           Accessor<T,l> subArray(index_t iBegin, index_t jBegin, index_t r, index_t c) const;
				__host__ 	           Accessor<T,l> subArray(index_t iBegin, index_t jBegin, index_t kBegin, index_t r, index_t c, index_t s) const;
				__host__		   Accessor<T,l> topLeftCorner(index_t r, index_t c) const;
				__host__		   Accessor<T,l> bottomLeftCorner(index_t r, index_t c) const;
				__host__		   Accessor<T,l> topRightCorner(index_t r, index_t c) const;
				__host__		   Accessor<T,l> bottomRightCorner(index_t r, index_t c) const;
				__host__		   Accessor<T,l> diagonal(index_t o=0) const;
				__host__		   Accessor<T,l> secondaryDiagonal(index_t o=0) const;
				__host__		   Accessor<T,l> reinterpretedLayout(index_t r, index_t c=1, index_t s=1) const;
				__host__		   Accessor<T,l> reinterpretedLayout(const Layout& other) const;
				__host__  	           Accessor<T,l> flattened(void) const;
				__host__  	           Accessor<T,l> stretched(void) const;
				__host__  	           Accessor<T,l> vectorized(void) const;
				__host__		   Accessor<T,l> squeezed(void) const;
				__host__ 	           std::vector< Accessor<T,l> > splitColumns(index_t jBegin, index_t c) const;
				__host__ 	           std::vector< Accessor<T,l> > splitSlices(index_t kBegin, index_t s) const;
				__host__ 	           std::vector< Accessor<T,l> > splitSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, index_t r=0, index_t c=0, index_t s=0) const;
				__host__ 	           std::vector< Accessor<T,l> > splitSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, const Layout& layout) const;

			// Assignment :
				template<typename TExpr>
				Accessor<T,l>& assign(const TExpr& expr, cudaStream_t stream=NULL);
				Accessor<T,l>& assign(const Accessor<T,l>& a, cudaStream_t stream=NULL);
				template<Location l2>
				Accessor<T,l>& assign(const Accessor<T,l2>& a, cudaStream_t stream=NULL);
				Accessor<T,l>& assign(const Array<T,l>& a, cudaStream_t stream=NULL);
				template<Location l2>
				Accessor<T,l>& assign(const Array<T,l2>& a, cudaStream_t stream=NULL);

			// Operator =
				template<typename TExpr>
				Accessor<T,l>& operator=(const TExpr& expr);
				Accessor<T,l>& operator=(const Accessor<T,l>& a);
				template<Location l2>
				Accessor<T,l>& operator=(const Accessor<T,l2>& a);
				Accessor<T,l>& operator=(const Array<T,l>& a);
				template<Location l2>
				Accessor<T,l>& operator=(const Array<T,l2>& a);

			// Compound assignment operator :
				#define ACCESSOR_COMPOUND_ASSIGNMENT( operatorName ) \
					template<typename TExpr> \
					Accessor<T,l>& operatorName (const TExpr& expr); \
					Accessor<T,l>& operatorName (const Accessor<T,l>& a); \
					template<Location l2> \
					Accessor<T,l>& operatorName (const Accessor<T,l2>& a); \
					Accessor<T,l>& operatorName (const Array<T,l>& a); \
					template<Location l2> \
					Accessor<T,l>& operatorName (const Array<T,l2>& a);

				ACCESSOR_COMPOUND_ASSIGNMENT( operator+= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator-= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator*= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator/= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator%= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator&= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator|= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator^= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator<<= )
				ACCESSOR_COMPOUND_ASSIGNMENT( operator>>= )
				#undef ACCESSOR_COMPOUND_ASSIGNMENT

			// Masked assignment : 
				template<typename TExprMask, typename TExpr>
				Accessor<T,l>& maskedAssignment(const TExprMask& exprMask, const TExpr& expr, cudaStream_t stream=NULL);

				template<class Op>
				__host__ void singleScan(const Op& op) const;

			// Other tools :
				template<typename TBis, Location lBis>
				__host__ friend std::ostream& operator<<(std::ostream& os, const Accessor<TBis, lBis>& A); // For debug, not for performance.
	};

	/**
	\brief Array class.
	\tparam T Type of the array (either a primitive type or a complex type).
	\tparam l Location of the data (see Kartet::Location).

	Main array class. See also \ref OperatorsGroup and \ref FunctionsGroup for available functions.
	**/
	template<typename T, Location l=KARTET_DEFAULT_LOCATION>
	class Array : public Accessor<T, l>
	{
		private :
			__host__ void allocateMemory(void);

		public :
			__host__ Array(index_t r, index_t c=1, index_t s=1);
			__host__ Array(const Layout& layout);
			__host__ Array(const T* ptr, index_t r, index_t c=1, index_t s=1);
			__host__ Array(const T* ptr, const Layout& layout);
			__host__ Array(const Array<T,l>& a);
			template<typename TIn, Location lin>
			__host__ Array(const Accessor<TIn,lin>& a);
			__host__ Array(std::istream& stream, const bool convert=true, const size_t maxBufferSize=KARTET_DEFAULT_BUFFER_SIZE, int sourceTypeIndex=0);
			__host__ Array(const std::string& filename, const bool convert=true, const size_t maxBufferSize=KARTET_DEFAULT_BUFFER_SIZE, int sourceTypeIndex=0);
			__host__ ~Array(void);

			// From Accessor<T,l>::Layout
				using Accessor<T,l>::Layout::numElements;
				using Accessor<T,l>::Layout::numElementsPerSlice;
				using Accessor<T,l>::Layout::numElementsPerFragment;
				using Accessor<T,l>::Layout::numRows;
				using Accessor<T,l>::Layout::numColumns;
				using Accessor<T,l>::Layout::numSlices;
				using Accessor<T,l>::Layout::numFragments;
				using Accessor<T,l>::Layout::width;
				using Accessor<T,l>::Layout::height;
				using Accessor<T,l>::Layout::depth;
				using Accessor<T,l>::Layout::columnsStride;
				using Accessor<T,l>::Layout::slicesStride;
				using Accessor<T,l>::Layout::offset;
				using Accessor<T,l>::Layout::setOffset;
				#ifdef __CUDACC__
				using Accessor<T,l>::Layout::dimensions;
				using Accessor<T,l>::strides;
				#endif
				using Accessor<T,l>::Layout::isMonolithic;
				using Accessor<T,l>::Layout::isSliceMonolithic;
				using Accessor<T,l>::Layout::reinterpretLayout;
				using Accessor<T,l>::Layout::flatten;
				using Accessor<T,l>::Layout::stretch;
				using Accessor<T,l>::Layout::vectorize;
				using Accessor<T,l>::Layout::splitLayoutColumns;
				using Accessor<T,l>::Layout::splitLayoutSlices;
				using Accessor<T,l>::Layout::splitLayoutSubArrays;
				using Accessor<T,l>::Layout::sameLayoutAs;
				using Accessor<T,l>::Layout::sameSliceLayoutAs;
				#ifdef __CUDACC__
				using Accessor<T,l>::Layout::getI;
				using Accessor<T,l>::Layout::getJ;
				using Accessor<T,l>::Layout::getK;
				#endif
				using Accessor<T,l>::Layout::getINorm;
				using Accessor<T,l>::Layout::getJNorm;
				using Accessor<T,l>::Layout::getKNorm;
				using Accessor<T,l>::Layout::getINormIncl;
				using Accessor<T,l>::Layout::getJNormIncl;
				using Accessor<T,l>::Layout::getKNormIncl;
				using Accessor<T,l>::Layout::getIClamped;
				using Accessor<T,l>::Layout::getJClamped;
				using Accessor<T,l>::Layout::getKClamped;
				using Accessor<T,l>::Layout::getIWrapped;
				using Accessor<T,l>::Layout::getJWrapped;
				using Accessor<T,l>::Layout::getKWrapped;
				using Accessor<T,l>::Layout::getIndex;
				using Accessor<T,l>::Layout::getPosition;
				using Accessor<T,l>::Layout::getPositionFFTShift;
				using Accessor<T,l>::Layout::getPositionFFTInverseShift;
				using Accessor<T,l>::Layout::getPositionClampedToEdge;
				using Accessor<T,l>::Layout::getPositionWarped;
				using Accessor<T,l>::Layout::isInside;
				using Accessor<T,l>::Layout::isRowValid;
				using Accessor<T,l>::Layout::isColumnValid;
				using Accessor<T,l>::Layout::isSliceValid;
				using Accessor<T,l>::Layout::unpackIndex;
				using Accessor<T,l>::Layout::unpackPosition;
				using Accessor<T,l>::Layout::moveToNext;
				#ifdef __CUDACC__
				using Accessor<T,l>::Layout::blockSize;
				using Accessor<T,l>::Layout::numBlocks;
				#endif
				using Accessor<T,l>::Layout::singleScan;
				using Accessor<T,l>::Layout::dualScan;
				using Accessor<T,l>::Layout::columnLayout;
				using Accessor<T,l>::Layout::sliceLayout;
				using Accessor<T,l>::Layout::monolithicLayout;

			// From Accessor<T,l>
				using Accessor<T,l>::location;
				using Accessor<T,l>::dataPtr;
				using Accessor<T,l>::size;
				#ifdef __CUDACC__
				using Accessor<T,l>::data;
				using Accessor<T,l>::dataInSlice;
				using Accessor<T,l>::dataFFTShift;
				using Accessor<T,l>::dataFFTInverseShift;
				#endif
				using Accessor<T,l>::getData;
				using Accessor<T,l>::setData;
				using Accessor<T,l>::readFromStream;
				using Accessor<T,l>::readFromFile;
				using Accessor<T,l>::writeToStream;
				using Accessor<T,l>::writeToFile;
				using Accessor<T,l>::layout;
				using Accessor<T,l>::element;
				using Accessor<T,l>::elements;
				using Accessor<T,l>::column;
				using Accessor<T,l>::endColumn;
				using Accessor<T,l>::columns;
				using Accessor<T,l>::slice;
				using Accessor<T,l>::endSlice;
				using Accessor<T,l>::slices;
				using Accessor<T,l>::subArray;
				using Accessor<T,l>::flattened;
				using Accessor<T,l>::stretched;
				using Accessor<T,l>::vectorized;
				using Accessor<T,l>::splitColumns;
				using Accessor<T,l>::splitSlices;
				using Accessor<T,l>::splitSubArrays;
				using Accessor<T,l>::assign;
				using Accessor<T,l>::maskedAssignment;

			// Specifics :
				Accessor<T,l>& accessor(void);
				const Accessor<T,l>& accessor(void) const;

				template<typename TExpr>
				Array<T,l>& operator=(const TExpr& expr);
				Array<T,l>& operator=(const Accessor<T,l>& a);
				template<Location l2>
				Array<T,l>& operator=(const Accessor<T,l2>& a);
				Array<T,l>& operator=(const Array<T,l>& a);
				template<Location l2>
				Array<T,l>& operator=(const Array<T,l2>& a);

			// Compound assignment operator :
				#define ARRAY_COMPOUND_ASSIGNMENT( operatorName ) \
					template<typename TExpr> \
					Array<T,l>& operatorName (const TExpr& expr); \
					Array<T,l>& operatorName (const Accessor<T,l>& a); \
					template<Location l2> \
					Array<T,l>& operatorName (const Accessor<T,l2>& a); \
					Array<T,l>& operatorName (const Array<T,l>& a); \
					template<Location l2> \
					Array<T,l>& operatorName (const Array<T,l2>& a);

				ARRAY_COMPOUND_ASSIGNMENT( operator+= )
				ARRAY_COMPOUND_ASSIGNMENT( operator-= )
				ARRAY_COMPOUND_ASSIGNMENT( operator*= )
				ARRAY_COMPOUND_ASSIGNMENT( operator/= )
				ARRAY_COMPOUND_ASSIGNMENT( operator%= )
				ARRAY_COMPOUND_ASSIGNMENT( operator&= )
				ARRAY_COMPOUND_ASSIGNMENT( operator|= )
				ARRAY_COMPOUND_ASSIGNMENT( operator^= )
				ARRAY_COMPOUND_ASSIGNMENT( operator<<= )
				ARRAY_COMPOUND_ASSIGNMENT( operator>>= )

				#undef ARRAY_COMPOUND_ASSIGNMENT

			__host__ static Array<T,l>* buildFromStream(std::istream& stream, const bool convert=true, const size_t maxBufferSize=KARTET_DEFAULT_BUFFER_SIZE);
			__host__ static Array<T,l>* buildFromFile(const std::string& filename, const bool convert=true, const size_t maxBufferSize=KARTET_DEFAULT_BUFFER_SIZE);
	};
	
} // namespace Kartet

// Include sub-definitions :
	#include "Core/ArrayTools.hpp"
	#include "Core/ArrayExpressions.hpp"
	#include "Core/ArrayOperators.hpp"

#endif

