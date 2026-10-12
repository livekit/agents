/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_TOKENNFA2TUPLENFA_H_
#define _FA_TOKENNFA2TUPLENFA_H_

#include "FAConfig.h"
#include "FAAny2AnyOther_global_t.h"
#include "FATokenNfa2Nfa_t.h"
#include "FANfas2TupleNfa.h"
#include "FAIsDotNfa.h"
#include "FAArray_cont_t.h"
#include "FARSNfa_ro.h"
#include "FARSNfa_wo_ro.h"

namespace BlingFire
{

class FAAllocatorA;
class FAMultiMapA;
class FATagSet;

///
/// This class builds Tuple Ows NFA from the Token Nfa and Token -> Ows maps.
///

class FATokenNfa2TupleNfa {

public:
    FATokenNfa2TupleNfa (FAAllocatorA * pAlloc);
    ~FATokenNfa2TupleNfa ();

public:
    /// token NFA
    void SetTokenNfa (const FARSNfaA * pTokenNfa);
    /// output Tuple NFA
    void SetOutNfa (FARSNfaA * pOutNfa);

    /// tagset, optional
    void SetTagSet (const FATagSet * pTagSet);
    /// Dgt,Token --> TypeNum CNF
    void SetCNF (const FAMultiMapA * pToken2CNF);
    /// Dgt,TypeNum --> { Ows } map
    void SetType2Ows (const FAMultiMapA * pType2Ows);
    /// base token Iw
    void SetTnBaseIw (const int TnBaseIw);
    /// base of interval which should not be grouped into tuples, optional
    void SetIgnoreBase (const int IgnoreBase);
    /// max of interval which should not be grouped into tuples, optional
    void SetIgnoreMax (const int IgnoreMax);

    void Process ();

    const int GetTokenType () const;

private:
    void Prepare ();
    void Clear ();
    const bool IsDotNfa (const FARSNfaA * pNfa, const int Digitizer);

private:
    const FATagSet * m_pTagSet;
    int m_TnBaseIw;
    const FAMultiMapA * m_pToken2CNF;
    const FAMultiMapA * m_pType2Ows;
    int m_TokenType;
    FARSNfaA * m_pOutNfa;

    // '.' - expansion
    FAAny2AnyOther_global_t < FARSNfaA, FARSNfa_wo_ro > m_dot_exp;
    /// builds one NFA for each digitizer type
    FATokenNfa2Nfa_t < FARSNfaA > m_tnnfa2nfa;
    /// merges n-NFAs into a single one
    FANfas2TupleNfa m_nfas2nfa;
    /// identifies "."-NFAs
    FAIsDotNfa m_is_dot;
    /// temporary "."-alphabet storage
    FAArray_cont_t < int > m_ows;

    /// temporary "."-expanded token NFA
    FARSNfa_wo_ro m_ExpNfa;

    /// singleton NFAs
    FARSNfa_ro m_TxtNfa;
    FARSNfa_ro m_TagNfa;
    FARSNfa_ro m_DctNfa;
};

}

#endif
