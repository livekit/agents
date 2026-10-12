/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SPLITSTATES_H_
#define _FA_SPLITSTATES_H_

#include "FAConfig.h"
#include "FASplitSets.h"
#include "FAArray_cont_t.h"
#include "FAEncoder_pref.h"
#include "FAChain2Num_hash.h"

namespace BlingFire
{

class FARSNfaA;


class FASplitStates {

public:
    FASplitStates (FAAllocatorA * pAlloc);

public:
    /// sets up input automaton
    void SetNfa (const FARSNfaA * pNfa);
    /// sets up initial state -> eq class mapping
    void SetS2C (const int * pS2C, const int Size);
    /// makes processing
    void Process ();
    /// sets up output state -> eq class mapping
    const int GetS2C (const int ** ppS2C);
    /// returns object into the initial state
    void Clear ();

private:
    inline const int GetDstId (const int * pDsts, const int Dsts);

private:
    const FARSNfaA * m_pNfa;
    const int * m_pS2C;
    int m_Size;

    FAArray_cont_t < int > m_alphabet;
    FAArray_cont_t < int > m_info;
    FAArray_cont_t < int > m_tmp;
    FASplitSets m_split_sets;
    FAEncoder_pref m_enc;
    FAChain2Num_hash m_dsts2num;
};

}

#endif
