/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_IMAGEDUMP_H_
#define _FA_IMAGEDUMP_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// Keeps thread sharable data resources
///

class FAImageDump {

public:
    FAImageDump ();
    ~FAImageDump ();

public:
    // loads image dump from file, entire file is used as single image
    void Load (const char * pFileName, const bool fUseMemMapping = false);
    // sets up image dump from the external pointer
    void SetImageDump (const unsigned char * pImageDump);
    // returns pointer to the image dump
    const unsigned char * GetImageDump () const;

private:
    // load file into the heap
    void FALoadHeap (const char * pFileName);
    // frees heap memory
    void FAFreeHeap ();
    // load file via memory mapped files
    void FALoadMm (const char * pFileName);
    // returns all memory map related resources back
    void FAFreeMm ();

private:
    /// pointer to the data
    unsigned char * m_pImageDump;
    /// true if the memory should be returned to heap
    bool m_MustDelete;
    /// not 0, if data are loaded thru the memory mapped file
    HANDLE m_hFileMapping;
    /// true if the memory should be unmapped
    bool m_MustUnmap;
};

}

#endif
