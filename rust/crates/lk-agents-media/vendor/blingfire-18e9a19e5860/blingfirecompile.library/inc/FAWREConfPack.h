/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WRECONFPACK_H_
#define _FA_WRECONFPACK_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAArrayPack.h"
#include "FAMultiMapPack.h"
#include "FADfaPack_triv.h"

namespace BlingFire
{

class FAAllocatorA;
class FAWREConfA;

///
/// This class builds a memory-dump representation for the compiled WRE
///

class FAWREConfPack {

public:
    FAWREConfPack (FAAllocatorA * pAlloc);

public:
    /// sets up WRE
    void SetWre (const FAWREConfA * pWre);
    /// builds dump
    void Process ();
    /// returns output dump representation
    const int GetDump (const unsigned char ** ppDump) const;

private:
    /// makes object ready
    void Prepare ();
    /// returns the resulting dump size
    const int BuildDumps ();

private:
    /// input WRE
    const FAWREConfA * m_pWre;
    /// resulting memory dump
    FAArray_cont_t < unsigned char > m_dump;
    /// configuration vector
    FAArray_cont_t < int > m_conf;
    /// a trivial-pack family
    FADfaPack_triv m_pack_txt_dig;
    FAArrayPack m_pack_dct_dig;
    FADfaPack_triv m_pack_fsm1;
    FADfaPack_triv m_pack_fsm2;
    FAMultiMapPack m_pack_trbr;
};

}

#endif
