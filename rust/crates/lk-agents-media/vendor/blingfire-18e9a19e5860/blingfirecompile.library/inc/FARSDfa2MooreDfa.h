/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFA2MOOREDFA_H_
#define _FA_RSDFA2MOOREDFA_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaA;
class FAState2OwA;
class FAState2OwsA;

///
/// This class converts (min) Rabin-Scott Dfa into (min) Moore Dfa.
///
/// Note: This class works only for the languages where words end with
///   Ows, for example a dictionary. In other words states reachable from
///   the states with Ows should not have any outgoing arcs. The general
///   case of convertion will require a use of NFA.
///

class FARSDfa2MooreDfa {

public:
    FARSDfa2MooreDfa (FAAllocatorA * pAlloc);

public:
    // input RS automaton
    void SetRSDfa (const FARSDfaA * pInDfa);
    // output Moore automaton
    void SetMooreDfa (FARSDfaA * pOutDfa);
    // State -> Ow reaction map, m_pState2Ows must be NULL
    void SetState2Ow (FAState2OwA * pState2Ow);
    // State -> Ows reaction map, m_pState2Ow must be NULL
    void SetState2Ows (FAState2OwsA * pState2Ows);
    // sets up range for Iws which will be converted to Ows
    void SetOwsRange (const int BaseOw, const int MaxOw);
    // if true then BaseOw won't be subtracted from Iw during convertion
    void SetKeepOws (const bool KeepOws);
    // makes convertion
    void Process ();
    // returns object into the initial state
    void Clear ();

private:
    inline void ProcessState (const int State);
    inline const int GetNewMaxIw () const;

private:
    // input RS-automaton
    const FARSDfaA * m_pInDfa;
    // output Moore-automaton
    FARSDfaA * m_pOutDfa;
    FAState2OwA * m_pState2Ow;
    FAState2OwsA * m_pState2Ows;
    // specifies the Ows range, m_BaseOw will be substructed from the Iw
    int m_BaseOw;
    int m_MaxOw;
    //
    bool m_KeepOws;
    // temporary Ows storage
    FAArray_cont_t < int > m_ows;
	// temporary storage for final states
    FAArray_cont_t < int > m_finals;
    // alphabet
    const int * m_pAlphabet;
    int m_AlphabetSize;
};

}

#endif
