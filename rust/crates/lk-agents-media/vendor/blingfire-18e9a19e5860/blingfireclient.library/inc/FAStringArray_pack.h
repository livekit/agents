/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STRING_ARRAY_PACK_H_
#define _FA_STRING_ARRAY_PACK_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class interprets the dump created by FAStringArrayPack.
///

class FAStringArray_pack : public FASetImageA {

public:
    FAStringArray_pack ();

public:
    /// sets up image dump
    void SetImage (const unsigned char * pImage);

    /// given an index, returns the length of the string and 
    ///  a pointer to the string
    const int GetAt (const int Idx, const unsigned char ** ppStr) const;

    /// given an index, returns the length of the string and 
    ///  copies the string into the destination memory upto MaxBuffSize bytes
    const int GetAt (
            const int Idx, 
            __out_ecount (MaxBuffSize) unsigned char * pBuff, 
            const int MaxBuffSize
        ) const;

    /// returns number of elementes in the array
    const int GetCount () const;

private:
    int m_Count;                   // number of elements in the array
    const unsigned int * m_pOffsetsAndEnd;  // int array of length m_Count + 1
    const unsigned char * m_pData; // a pointer to all strings
};

}

#endif
