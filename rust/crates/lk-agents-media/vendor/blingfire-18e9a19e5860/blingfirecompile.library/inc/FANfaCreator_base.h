/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_NFA_CREATOR_BASE_H_
#define _FA_NFA_CREATOR_BASE_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FARSNfa_wo_ro.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Base class for different types of regular expressions processing.
///

class FANfaCreator_base {

public:
    FANfaCreator_base (FAAllocatorA * pAlloc);
    virtual ~FANfaCreator_base ();

public:
    /// returns created Nfa
    const FARSNfaA * GetNfa () const;
    /// returns object into the initial state
    void Clear ();
    // must be called after all transitions have been added
    void Prepare ();

    /// sets any symbol iw
    void SetAnyIw (const int AnyIw);
    /// sets any symbol iw
    void SetAnchorLIw (const int AnchorLIw);
    /// sets any symbol iw
    void SetAnchorRIw (const int AnchorRIw);
    /// sets up new regular expression
    void SetRegexp (const char * pRegexp, const int ReLength);
    // sets up whether to treat triangular brackets as symbols or not
    void SetTrBr2Iw (const bool TrBr2Iw);
    /// sets up the base for the output nfa's Iws
    void SetTrBrBaseIw (const int TrBrBaseIw);

public:
    /// sets up the initial state
    virtual void SetInitial (const int State);
    /// sets up the final state
    virtual void SetFinal (const int State);
    /// transitions with special symbols processing
    virtual void SetSpecTransition (
            const int FromState,
            const int ToState,
            const int Type
        );
    // converts label into an input weight
    virtual void SetTransition (
            const int FromState,
            const int ToState,
            const int LabelOffset,
            const int LabelLength
        );
    /// generates error message
    virtual void SetError (const int ErrorOffset);

protected:
    /// input regular expression
    const char * m_pRegexp;
    int m_ReLength;
    /// maps FARegexpTree types into specified Iws
    FAArray_cont_t < int > m_type2spec;
    /// output nfa
    FARSNfa_wo_ro m_tmp_nfa;
    /// tmp storage for the initial states
    FAArray_cont_t < int > m_initials;
    /// tmp storage for the final states
    FAArray_cont_t < int > m_finals;
    /// specifies whether to convert triangular bracket-symbols into iws
    bool m_TrBr2Iw;
    int m_TrBrBaseIw;

};

}

#endif
