/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ALLOCATORA_H_
#define _FA_ALLOCATORA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// General interface for allocators. _DEBUG_MEMORY should be defined when 
/// memory usage debugging is needed.
///

class FAAllocatorA {

public:

#ifndef _DEBUG_MEMORY

  /// allocates size of characters
  virtual void * Alloc (const int size) = 0;
  /// if the data does not fit the memory it allocates a new block and
  /// makes a byte copy
  virtual void * ReAlloc (void * ptr, const int size) = 0;
  /// frees the allocation
  virtual void Free (void * ptr) = 0;

#else

  /// allocates size of characters
  virtual void * Alloc (
            const int size, 
            const char * pFile, 
            const int Line
        ) = 0;
  /// if the data does not fit the memory it allocates a new block and
  /// makes a byte copy
  virtual void * ReAlloc (
            void * ptr, 
            const int size, 
            const char * pFile, 
            const int Line
        ) = 0;
  /// frees the allocation
  virtual void Free (
            void * ptr, 
            const char * pFile, 
            const int Line
        ) = 0;

#endif // _DEBUG_MEMORY

};

}

#ifndef _DEBUG_MEMORY

#define FAAlloc(A, S) (A)->Alloc(S)
#define FAReAlloc(A, P, S) (A)->ReAlloc(P, S)
#define FAFree(A, P) (A)->Free(P)

#else

#define FAAlloc(A, S) (A)->Alloc(S, __FILE__, __LINE__)
#define FAReAlloc(A, P, S) (A)->ReAlloc(P, S, __FILE__, __LINE__)
#define FAFree(A, P) (A)->Free(P, __FILE__, __LINE__)

#endif // _DEBUG_MEMORY

#endif
