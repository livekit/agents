/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFA2PERFHASH_H_
#define _FA_RSDFA2PERFHASH_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FATopoSort_t.h"
#include "FADfaTopoGraph.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaA;
class FAState2OwsA;

///
/// This class calculates perfect-hash output weights (Ows), one per 
/// transition in such a way that summ along the string matching path gives 
/// a string id.
///
/// Note: The output stored in FAState2OwsA map, where i-th Ow corresponds
/// to the i-th transition ordered by the values of corresponding Iw.
///

class FARSDfa2PerfHash {

public:
    FARSDfa2PerfHash (FAAllocatorA * pAlloc);

public:
    /// sets up input automaton
    void SetRsDfa (const FARSDfaA * pDfa);
    /// sets up output Ows container
    void SetState2Ows (FAState2OwsA * pState2Ows);
    /// calculates perfect-hash's Ows
    void Process ();

private:
    void Prepare ();
    void TopoSort ();
    inline bool IsFinal (const int State) const;
    void CalcCds ();

private:
    // input 
    const FARSDfaA * m_pDfa;
    // output
    FAState2OwsA * m_pState2Ows;
    // finals, helper
    int m_FinalCount;
    const int * m_pFinals;
    // in Dfa graph
    FADfaTopoGraph m_Graph;
    // topological sorter
    FATopoSort_t < FADfaTopoGraph > m_sorter;
    // state cardinality map
    FAArray_cont_t < int > m_state2count;
    // temporary Ows storage
    FAArray_cont_t < int > m_ows;

};

}

#endif
