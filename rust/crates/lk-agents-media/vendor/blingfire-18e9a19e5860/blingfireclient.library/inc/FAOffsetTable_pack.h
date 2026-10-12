/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_OFFSETTABLE_PACK_H_
#define _FA_OFFSETTABLE_PACK_H_

#include "FAConfig.h"
#include "FASetImageA.h"

namespace BlingFire
{

///
/// This class interprets image dump created by FAOffsetTablePack class
///

class FAOffsetTable_pack : public FASetImageA {

public:
    FAOffsetTable_pack ();

public:
    /// sets up image dump
    void SetImage (const unsigned char * pImage);
    /// returns offset by its index
    const unsigned int GetOffset (const int Idx) const;

private:
    // array of bases
    const unsigned char * m_pBase;
    // array of deltas
    const unsigned char * m_pDelta;
    // base size in bytes
    int m_BaseSize;
    // shift value
    int m_ShiftValue;
    // number of indices for validation
    int m_OffsetCount;
};

}

#endif
