/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSNFA_AR_JUDY_H_
#define _FA_RSNFA_AR_JUDY_H_

#include "FAConfig.h"
#include "FARSNfaA.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAMultiMap_judy;
class FAAllocatorA;

//
// FARSNfaA implementation, see FARSNfaA.h for the methods description
//

class FARSNfa_ar_judy : public FARSNfaA {

public:

    FARSNfa_ar_judy (FAAllocatorA * pAlloc);
    virtual ~FARSNfa_ar_judy ();

public:

    const int GetMaxState () const;
    const int GetMaxIw () const;
    const int GetInitials (const int ** ppStates) const;
    const int GetFinals (const int ** ppStates) const;
    const bool IsFinal (const int State) const;
    const bool IsFinal (const int * pStates, const int Size) const;
    const int GetIWs (
            const int State,
            const int ** ppIws
        ) const;
    const int GetDest (
            const int State, 
            const int Iw,
            const int ** ppIwDstStates
        ) const;
    const int GetDest (
            const int State,
            const int Iw,
            int * pDstStates,
            const int MaxCount
        ) const;

public:

    // call this method before Create
    void SetMaxState (const int MaxState);
    // call this method before Create
    void SetMaxIw (const int MaxIw);
    // must be called before any transitions have been added
    void Create ();
    // call this method once for each pair <FromState, Iw>
    void SetTransition (const int FromState,
                        const int Iw,
                        const int * pDstStates,
                        const int DstStatesCount);
    // adds a transition
    void SetTransition (const int FromState, const int Iw, const int DstState);
    void SetInitials (const int * pStates, const int StateCount);
    // expects finals to have the largest numbers amoung all the states, e.g.
    // f > q \forall f \in F and q \in Q/F
    void SetFinals (const int * pStates, const int StateCount);

    // must be called after all transitions have been added
    void Prepare ();
    /// return Nfa into the just after constructor call (frees memory)
    void Clear ();

public:
    /// additional functionality
    void PrepareState (const int State);
    /// adds more states to the automaton,
    /// new size is OldSize + StatesCount states
    void AddStateCount (const int StatesCount);
    /// actually, just increments m_IwCount
    /// the new IwCount is old IwCount plus IwCount
    void AddIwCount (const int IwCount);

private:
    /// creates all necessary structures for the State
    inline void create_state (const int State);

private:

    unsigned int m_IwCount;
    unsigned int m_StateCount;
    FAArray_cont_t < int > m_initials;
    FAArray_cont_t < int > m_finals;
    int m_min_final;
    FAArray_cont_t < FAArray_cont_t < int > * > m_state2iws;
    FAArray_cont_t < FAMultiMap_judy * > m_state_iw2dsts;

    FAAllocatorA * m_pAlloc;
};

}

#endif
