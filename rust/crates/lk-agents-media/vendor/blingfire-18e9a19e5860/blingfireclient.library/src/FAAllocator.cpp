/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAAllocator.h"
#include "FAException.h"

#ifdef _DEBUG_MEMORY
#include <iostream>
#endif

namespace BlingFire
{

FAAllocator::FAAllocator ()
{}

FAAllocator::~FAAllocator ()
{}


#ifndef _DEBUG_MEMORY


void * FAAllocator::Alloc (const int size)
{
  DebugLogAssert (0 < size);

  void * ptr = malloc (size);
  FAAssert (ptr, FAMsg::OutOfMemory);

  return ptr;
}

void * FAAllocator::ReAlloc (void * ptr, const int size)
{
  DebugLogAssert (0 < size);

  void * new_ptr = realloc (ptr, size);
  FAAssert (new_ptr, FAMsg::OutOfMemory);

  return new_ptr;
}

void FAAllocator::Free (void* ptr)
{
  if (NULL != ptr) {
    free (ptr);
  }
}


#else


FAAllocator::_TRec::
    _TRec (const int Size, const char * pFile, const int Line) :
        m_Size (Size),
        m_pFile (pFile),
        m_Line (Line)
{}

FAAllocator::_TRec::
    _TRec (const _TRec& C) :
        m_Size (C.m_Size),
        m_pFile (C.m_pFile),
        m_Line (C.m_Line)
{}


void * FAAllocator::
    Alloc (
        const int size, 
        const char * pFile, 
        const int Line
    )
{
  DebugLogAssert (0 < size);

  void* ptr = malloc (size + sizeof (int));
  DebugLogAssert (ptr);

  *(int*)(((char*)ptr) + size) = 0xAAAAAAAA;
  m_ptr2rec.insert (std::make_pair (ptr, _TRec (size, pFile, Line)));

  return ptr;
}

void * FAAllocator::
    ReAlloc (
        void* ptr, 
        const int size, 
        const char * pFile,
        const int Line
    )
{
  DebugLogAssert (0 < size);

  if (NULL != ptr) {

    std::map < void *, _TRec >::iterator I = m_ptr2rec.find (ptr);
    DebugLogAssert (I != m_ptr2rec.end ());

    const int OldSize = I->second.m_Size;
    DebugLogAssert (0 < OldSize);

    DebugLogAssert (0xAAAAAAAA == *(int*)(((char*)ptr) + OldSize));
    m_ptr2rec.erase (ptr);
  }

  void* new_ptr = realloc (ptr, size + sizeof (int));
  DebugLogAssert (new_ptr);

  *(int*)(((char*)new_ptr) + size) = 0xAAAAAAAA;
  m_ptr2rec.insert (std::make_pair (new_ptr, _TRec (size, pFile, Line)));

  return new_ptr;
}

void FAAllocator::
    Free (
        void* ptr,
        const char * /*pFile*/,
        const int /*Line*/
    )
{
  if (NULL != ptr) {

    std::map < void *, _TRec >::iterator I = m_ptr2rec.find (ptr);
    DebugLogAssert (I != m_ptr2rec.end ());

    const int OldSize = I->second.m_Size;
    DebugLogAssert (0 < OldSize);

    DebugLogAssert (0xAAAAAAAA == *(int*)(((char*)ptr) + OldSize));
    m_ptr2rec.erase (ptr);

    free (ptr);
  }
}

void FAAllocator::PrintLeaks (std::ostream & os) const
{
    std::map < void *, _TRec >::const_iterator I = \
        m_ptr2rec.begin ();

    for (; m_ptr2rec.end () != I; ++I) {

        const _TRec & r = I->second;

        os << "ERROR: Memory Leak : " << r.m_Size << " bytes " \
           << " in " << r.m_pFile << " at line " << r.m_Line << '\n';
    }
}

#endif // _DEBUG_MEMORY

}
