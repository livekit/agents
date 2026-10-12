/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFARENUM_REMOVE_GAPS_H_
#define _FA_RSDFARENUM_REMOVE_GAPS_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FABitArray.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaA;

///
/// This class renumerates so that there are no gaps between subsequent states.
///

class FARSDfaRenum_remove_gaps {

public:
    FARSDfaRenum_remove_gaps (FAAllocatorA * pAlloc);

public:
    /// sets up input DFA
    void SetDfa (const FARSDfaA * pDfa);
    /// makes processing
    void Process ();
    /// returns mapping from OldStates to NewStates
    /// the array is MaxState + 1 long, can be called only after Process
    const int * GetOld2NewMap () const;
    /// returns object into the initial state
    void Clear ();

private:
    inline void Prepare ();
    inline const bool IsUsed (const int State) const;

private:
    // input automaton
    const FARSDfaA * m_pDfa;
    // is final implementation for extra safety
    FABitArray m_is_final;
    // mapping from old state nums to new one
    int m_MaxNewState;
    FAArray_cont_t < int > m_old2new;
};

}

#endif
