/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ALLOCATOR_H_
#define _FA_ALLOCATOR_H_

#include "FAConfig.h"
#include "FAAllocatorA.h"

#ifdef _DEBUG_MEMORY
#include <map>
#endif

namespace BlingFire
{

///
/// FAFAAllocatorA implementation see FAAllocatorA.h for details.
///
/// Note: Only a fixed number of bytes is checked behind the boundary of 
/// the allocated buffer so not all possible overruns can be detected.
///

class FAAllocator : public FAAllocatorA {

public:

  FAAllocator ();
  virtual ~FAAllocator ();

public:

#ifndef _DEBUG_MEMORY

  void * Alloc (const int size);
  void * ReAlloc (void * ptr, const int size);
  void Free (void * ptr);

#else

  void * Alloc (
            const int size, 
            const char * pFile, 
            const int Line
        );
  void * ReAlloc (
            void * ptr, 
            const int size, 
            const char * pFile, 
            const int Line
        );
  void Free (
            void * ptr, 
            const char * pFile, 
            const int Line
        );
  void PrintLeaks (
            std::ostream & os
        ) const;

private:

    class _TRec {
    public:
        _TRec (const int Size, const char * pFile, const int Line);
        _TRec (const _TRec& C);

    public:
        const int m_Size;
        const char * const m_pFile;
        const int m_Line;
    };

    std::map < void *, _TRec > m_ptr2rec;

#endif // _DEBUG_MEMORY

};

}

#ifndef _DEBUG_MEMORY
#define FAPrintLeaks(A, S)
#else
#define FAPrintLeaks(A, S) (A)->PrintLeaks (S)
#endif // _DEBUG_MEMORY

#endif
