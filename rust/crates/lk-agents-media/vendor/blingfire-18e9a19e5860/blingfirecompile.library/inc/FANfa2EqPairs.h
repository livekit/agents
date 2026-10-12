/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_NFA2EQPAIRS_H_
#define _FA_NFA2EQPAIRS_H_

#include "FAConfig.h"
#include "FANfa2Dfa_t.h"
#include "FAChain2Num_hash.h"
#include "FARSNfaA.h"
#include "FAState2OwsA.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This processor takes FSM and calculates all pairs of states < q1 q2 > 
/// reachable by word w, such as q1 != q2 .
///
/// Note: Unlike E.Roche's this algorithm does not calculate A^2, this
/// algorithm is based on NFA -> DFA but while DFA states are constructed
/// it collects all uniq pairs of NFA states from there. On practice that is
/// faster than calculate A^2.
///

class FANfa2EqPairs : private FAState2OwsA {

public:
    FANfa2EqPairs (FAAllocatorA * pAlloc);
    virtual ~FANfa2EqPairs ();

public:
    /// sets up input NFA
    void SetNfa (const FARSNfaA * pNfa);
    /// gets all pairs
    void Process ();
    /// returns pairs Count
    const int GetPairCount () const;
    /// returns specified pair by index
    void GetPair (const int Num, int * pQ1, int * pQ2) const;
    /// returns object into the initial state
    void Clear ();

private:
    // collects pairs via this method
    void SetOws (const int State, const int * pOldStates, const int Count);

private:
    const int GetOws (const int, int *, const int) const;
    const int GetOws (const int, const int **) const;
    const int GetMaxOwsCount () const;

    // stub, as we are not interested in DFA output
    class TDfaStub {
    public:
        void SetTransition (const int, const int *, const int *, const int);
        void SetTransition (const int, const int, const int);
        void SetInitial (const int);
        void SetFinals (const int * , const int);
        void Prepare ();
    };

private:
    const FARSNfaA * m_pNfa;
    TDfaStub m_dfa;
    FANfa2Dfa_t < FARSNfaA, TDfaStub > m_nfa2dfa;
    FAChain2Num_hash m_pairs;
};

}

#endif
