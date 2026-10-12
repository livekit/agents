/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_FSM_RENUM_H_
#define _FA_FSM_RENUM_H_

#include "FAConfig.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"
#include "FAMap_judy.h"
#include "FAState2Ow.h"
#include "FAState2Ows.h"
#include "FAMealyNfa.h"
#include "FAStringTokenizer.h"

#include <iostream>

namespace BlingFire
{

class FAAllocatorA;

///
/// Reads fsm (Rs, Moore, etc.) and renumerates its states in the way that:
/// final states have largest values, initial - lowest. No gaps in states.
///

class FAFsmRenum {

public:
    FAFsmRenum (FAAllocatorA * pAlloc);

public:
    // sets up automaton type to be read, TYPE_RS_NFA is used by default
    void SetFsmType (const int fsm_type);
    // makes convertion
    void Process (std::ostream * pOs, std::istream * pIs);

private:

    void ReadFsm (std::istream * pIs);
    void ReadOwMap (std::istream * pIs);
    void ReadOwsMap (std::istream * pIs);

    void PrintFsm (std::ostream * pOs) const;
    void PrintOwMap (std::ostream * pOs) const;
    void PrintOwsMap (std::ostream * pOs) const;

    void Remap ();
    void Sort ();
    void Clear ();

    inline void AddTranition (const int * pTr, const int TrSize);
    inline const int Old2New (const int State) const;

private:

    // keeps transitions, [[Src, Dst, Iw, [Ow]]_1, [Src, Dst, Iw, [Ow]]_2, ...]
    FAArray_t < int > m_trs_storage;
    // keeps indexes in m_trs_storage, where transitions starts
    FAArray_cont_t < int > m_trs;

    int m_DeadState;
    int m_OldMaxState;
    int m_NewMaxState;
    int m_MaxIw;
    int m_MaxOw;
    bool m_has_ows;

    int m_ICount;
    FAMap_judy m_I;

    int m_FCount;
    FAMap_judy m_F;

    int m_QCount;
    FAMap_judy m_Q;

    int m_fsm_type;

    FAStringTokenizer m_tokenizer;
    FAArray_cont_t < int > m_ows;

    FAState2Ow m_state2ow;
    FAState2Ows m_state2ows;
    FAMealyNfa m_in_sigma;
    FAMealyNfa m_out_sigma;

private:

    class TrCmp {
    public:
        TrCmp (const FAArray_t < int > * pTrs);

    public:
        const bool operator() (const int TrIdx1, const int TrIdx2) const;

    private:
        const FAArray_t < int > * m_pTrs;
    };
};

}

#endif
