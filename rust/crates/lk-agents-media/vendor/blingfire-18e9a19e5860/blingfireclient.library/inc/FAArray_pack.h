/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ARRAY_PACK_H_
#define _FA_ARRAY_PACK_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAArrayCA.h"

namespace BlingFire
{

///
/// This class interprets the dump created by FAArrayPack, see FAArrayPack
/// for the details of representation.
///

class FAArray_pack : public FASetImageA,
                     public FAArrayCA {

public:
    FAArray_pack ();

public:
    /// sets up image dump
    void SetImage (const unsigned char * pImage);
    /// returns value by the index, 0..Count-1
    const int GetAt (const int Idx) const;
    /// returns number of elementes in the array
    const int GetCount () const;

private:
    /// header
    int m_M;
    int m_SizeOfIndex;
    int m_SizeOfValue;
    int m_Count;
    int m_SizeOfChain;
    /// index data, may be empty
    const unsigned char * m_pIndex;
    /// data themselves
    const unsigned char * m_pData;
};

}

#endif
