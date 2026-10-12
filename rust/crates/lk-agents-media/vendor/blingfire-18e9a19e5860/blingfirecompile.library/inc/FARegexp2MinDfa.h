/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_REGEXP2MINDFA_H_
#define _FA_REGEXP2MINDFA_H_

#include "FAConfig.h"
#include "FARSNfaA.h"
#include "FARSDfaA.h"
#include "FARSNfa_wo_ro.h"
#include "FARSDfa_ro.h"
#include "FARSDfa_wo_ro.h"
#include "FARegexp2Nfa.h"
#include "FANfaCreator_char.h"
#include "FAAny2AnyOther_global_t.h"
#include "FANfa2Dfa_t.h"
#include "FADfa2MinDfa_hg_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This class builds Min DFA from the textual regular expression.
///

class FARegexp2MinDfa {

public:
    FARegexp2MinDfa (FAAllocatorA * pAlloc);

public:
    /// specifies input encoding name, e.g. CP1251, KOI8-R
    void SetEncodingName (const char * pEncStr);
    /// sets up input regexp text
    void SetRegexp (const char * pRegexp, const int Length);
    /// makes the construction
    void Process ();
    /// returns read-only interface to the corresponding Min DFA
    const FARSDfaA * GetRsDfa () const;

private:
    // Regexp -> NFA
    FARegexp2Nfa m_regexp2nfa;
    FANfaCreator_char m_nfa_char;
    // '.' - expansion
    FAAny2AnyOther_global_t < FARSNfaA, FARSNfa_wo_ro > m_dot_exp;
    FARSNfa_wo_ro m_nfa;
    // NFA -> DFA
    FANfa2Dfa_t < FARSNfa_wo_ro, FARSDfa_wo_ro > m_nfa2dfa;
    FARSDfa_wo_ro m_dfa;
    // DFA -> Min DFA
    FADfa2MinDfa_hg_t < FARSDfa_wo_ro, FARSDfa_ro > m_dfa2mindfa;
    FARSDfa_ro m_min_dfa;
};

}

#endif
