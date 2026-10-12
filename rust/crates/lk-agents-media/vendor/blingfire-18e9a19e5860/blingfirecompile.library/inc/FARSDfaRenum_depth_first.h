/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFARENUM_DEPTH_FIRST_H_
#define _FA_RSDFARENUM_DEPTH_FIRST_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAArray_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaA;

///
/// This class renumerates states in the depth first order.
///

class FARSDfaRenum_depth_first {

public:
    FARSDfaRenum_depth_first (FAAllocatorA * pAlloc);

public:
    // sets up input DFA
    void SetDfa (const FARSDfaA * pDfa);
    // makes processing
    void Process ();
    // returns mapping from OldStates to NewStates
    // the array is MaxState + 1 long, however unreachable won't be remapped
    const int * GetOld2NewMap () const;

private:
    // makes internal structures ready
    void Prepare ();
    // builds destination states set, ordered by probability to get there
    inline void build_dsts (const int State);

private:
    const FARSDfaA * m_pDfa;
    int m_MaxState;
    int m_IwsCount;
    const int * m_pIws;

    int m_LastState;

    FAArray_cont_t < int > m_old2new;
    FAArray_cont_t < int > m_tmp_dsts;
    FAArray_t < int > m_stack;

    FAArray_cont_t < int > m_iw2f;
    FAArray_cont_t < int > m_idx2f;
    FAArray_cont_t < int > m_tr;

};

}

#endif
