/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_REGEXP2NFA_H_
#define _FA_REGEXP2NFA_H_

#include "FAConfig.h"
#include "FAToken.h"
#include "FAArray_cont_t.h"
#include "FARegexpTree.h"
#include "FARegexpParser_msyacc.h"
#include "FARegexpTree2Funcs.h"
#include "FARegexpTree2PosEqs.h"
#include "FATrBr2Symbol.h"
#include "FARegexpLexer_triv.h"
#include "FARegexpLexer_char.h"
#include "FARegexpLexer_wre.h"

namespace BlingFire
{

class FAAllocatorA;
class FANfaCreator_base;

///
/// This processor builds Nfa from Regexp
///
/// Note:
/// 1. Constructed Nfa's states has the meaning of positions in regular 
/// expression.
/// 2. Processes: "(RE)0" not "RE" directly
///

class FARegexp2Nfa {

public:
    FARegexp2Nfa (FAAllocatorA * pAlloc);

public:
    // sets up input regular expression
    void SetRegexp (const char * pRegexp, const int Length);
    // sets up the label type
    void SetLabelType (const int LabelType);
    // indicates whether, character-based regular expression is in UTF-8
    // no need to call for other types of regular expression, false by default
    void SetUseUtf8 (const bool UseUtf8);
    // sets up output NFA container
    void SetNfa (FANfaCreator_base * pNfa);
    // sets up whether to use position equivalence classes instead of 
    // positions themselves as output states, uses eq-classes by default
    void SetKeepPos (const bool KeepPos);
    // sets up whether to treat triangular brackets as symbols or not
    void SetTrBr2Iw (const bool TrBr2Iw);
    // makes construction
    void Process ();
    // returns regexp parsing tree
    const FARegexpTree * GetRegexpTree () const;
    // returns regexp functions, e.g. first, last, nullable, follow pos
    const FARegexpTree2Funcs * GetRegexpFuncs () const;
    // returns mapping from each position to the smallest equivalent position
    const FAMapA * GetPos2Class () const;
    // returns object into the initial state, called automatically by Process
    void Clear ();

private:
    void PutInitials ();
    void PutFinals ();
    void PutTransitions ();
    void PutError ();

    inline const int Pos2Class (const int Pos) const;

private:
    FANfaCreator_base * m_pNfa;

    bool m_TrBr2Iw;

    // additional containers
    FARegexpTree m_tree;
    FAArray_cont_t < FAToken > m_tokens;
    // special tokens
    FAToken m_left_br;
    FAToken m_right_br;
    FAToken m_terminal_symbol;

    FARegexpLexer_triv m_re2tokens_digit;
    FARegexpLexer_char m_re2tokens_char;
    FARegexpLexer_wre m_re2tokens_wre;
    FARegexpLexerA * m_pRe2Tokens;

    FARegexpParser_msyacc m_tokens2tree;
    FARegexpTree2Funcs m_tree2funcs;
    FARegexpTree2PosEqs m_pos_eqs;

    const FAMapA * m_pPos2Class;

    int m_MaxPos;

    FATrBr2Symbol m_trbr2iw;
};

}

#endif
