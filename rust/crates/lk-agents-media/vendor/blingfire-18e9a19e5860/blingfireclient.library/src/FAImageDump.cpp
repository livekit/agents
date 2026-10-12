/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAImageDump.h"

namespace BlingFire
{

FAImageDump::FAImageDump () :
    m_pImageDump (NULL),
    m_MustDelete (false),
    m_hFileMapping (0),
    m_MustUnmap (false)
{}


FAImageDump::~FAImageDump ()
{
    FAImageDump::FAFreeHeap ();
    FAImageDump::FAFreeMm ();
}


void FAImageDump::FAFreeHeap ()
{
    if (m_MustDelete) {
        LogAssert (m_pImageDump);
        delete [] m_pImageDump;
        m_pImageDump = NULL;
        m_MustDelete = false;
    }
}


void FAImageDump::FAFreeMm ()
{
#ifndef BLING_FIRE_NOWINDOWS

    if(m_MustUnmap) {
        BOOL fRes = ::UnmapViewOfFile ((const void*) m_pImageDump);
        LogAssert (0 != fRes, "Cannot unmap the view of a file, GetLastError()=%lu", GetLastError());
        m_pImageDump = NULL;
        m_MustUnmap = false;
    }
    if(m_hFileMapping) {
        BOOL fRes = ::CloseHandle (m_hFileMapping);
        LogAssert (0 != fRes, "Cannot close handle, GetLastError()=%lu", GetLastError());
        m_hFileMapping = 0;
    }

#else

#endif
}


void FAImageDump::Load (const char * pFileName, const bool fUseMemMapping)
{
    LogAssert (pFileName);

    // free the memory and resources if there was anything loaded
    FAImageDump::FAFreeHeap ();
    FAImageDump::FAFreeMm ();

#ifndef BLING_FIRE_NOWINDOWS

    if (false == fUseMemMapping) {

        // load the file using fopen_s into the heap
        FALoadHeap (pFileName);

    } else {

        // load the file using memory mapping
        FALoadMm (pFileName);
    }

#else

    // load the file using fopen_s into the heap
    FALoadHeap (pFileName);

#endif
}


void FAImageDump::FALoadHeap (const char * pFileName)
{
    LogAssert (pFileName);

    FILE * file = NULL;
    int res = fopen_s (&file, pFileName, "rb");
    LogAssert (0 == res && NULL != file, "Failed to successfully open file %s", pFileName);

    res = fseek (file, 0, SEEK_END);
    LogAssert (0 == res);

    const unsigned int Size = ftell (file);
    LogAssert (0 < Size);

    res = fseek (file, 0, SEEK_SET);
    LogAssert (0 == res);

    m_pImageDump = NEW unsigned char [Size];
    LogAssert (m_pImageDump);

    const size_t ActSize = fread (m_pImageDump, sizeof (char), Size, file);
    LogAssert (ActSize == Size);

    fclose (file);

    m_MustDelete = true;
}


void FAImageDump::FALoadMm (const char * pFileName)
{

#ifndef BLING_FIRE_NOWINDOWS

    LogAssert (pFileName);

    HANDLE hFile = ::CreateFileA (pFileName, GENERIC_READ, FILE_SHARE_READ,
        NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    LogAssert (0 != hFile, "Failed to open a file %s for memory mapping, GetLastError()=%lu", 
        pFileName, GetLastError());

    m_hFileMapping = ::CreateFileMapping (hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    LogAssert (0 != m_hFileMapping, "Failed to create a memory mapping for file %s, GetLastError()=%lu", 
        pFileName, GetLastError());

    m_pImageDump = (unsigned char *) ::MapViewOfFile(m_hFileMapping, FILE_MAP_READ, 0, 0, 0);
    LogAssert (NULL != m_pImageDump, "Failed to get a pointer from the memory mapped file %s, GetLastError()=%lu", 
        pFileName, GetLastError());

    BOOL fRes = ::CloseHandle (hFile);
    LogAssert (0 != fRes, "Cannot close handle, GetLastError()=%lu", GetLastError());

    m_MustUnmap = true;

#else

#endif
}


void FAImageDump::SetImageDump (const unsigned char * pImageDump)
{
    FAImageDump::FAFreeHeap ();
    FAImageDump::FAFreeMm ();

    m_pImageDump = (unsigned char *) pImageDump;
}


const unsigned char * FAImageDump::GetImageDump () const
{
    return m_pImageDump;
}

}
