/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_REGEXPSIMPLIFY_H_
#define _FA_REGEXPSIMPLIFY_H_

#include "FAConfig.h"
#include "FAToken.h"
#include "FAArray_cont_t.h"
#include "FARegexpTree.h"
#include "FARegexpParser_msyacc.h"
#include "FARegexpTreeSort.h"
#include "FARegexpTreeSimplify_disj.h"
#include "FARegexpLexer_triv.h"
#include "FARegexpLexer_char.h"
#include "FARegexpLexer_wre.h"
#include "FARegexpTree2Str.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Simplifies regular expressions, e.g. reduces the amount of symbols.
///

class FARegexpSimplify {

public:
    FARegexpSimplify (FAAllocatorA * pAlloc);

public:
    // sets up input regular expression text
    void SetRegexp (const char * pRegexp, const int Length);
    // sets up the label type
    void SetLabelType (const int LabelType);
    // indicates whether, character-based regular expression is in UTF-8
    // no need to call for other types of regular expression, false by default
    void SetUseUtf8 (const bool UseUtf8);
    // sets up reverse sorting order (false by default)
    void SetReverse (const bool Reverse);
    // makes simplifications
    void Process ();
    // returns 0-terminated text of simplified regular expression
    const char * GetRegexp () const;
    // returns resulting tree
    const FARegexpTree * GetRegexpTree () const;
    // returns object into the initial state, called automatically
    void Clear ();

private:
    void PutError ();
    void ReverseTree ();

private:
    // tokens that always exist
    FAToken m_LeftBr;
    FAToken m_RightBr;
    FAToken m_TerminalSymbol;
    // original regular expression
    const char * m_pRegexp;
    int m_Length;
    // tokens
    FAArray_cont_t < FAToken > m_tokens;
    // syntax tree
    FARegexpTree m_tree;
    // lexer
    FARegexpLexer_triv m_re2tokens_digit;
    FARegexpLexer_char m_re2tokens_char;
    FARegexpLexer_wre m_re2tokens_wre;
    FARegexpLexerA * m_pRe2Tokens;
    // parser
    FARegexpParser_msyacc m_tokens2tree;
    // sorts disjunction arguments
    FARegexpTreeSort m_regexp_sort;
    // disjunctions simplificator
    FARegexpTreeSimplify_disj m_simpl;
    // builds regexp text from the tree
    FARegexpTree2Str m_tree2re;
    // pointer to simplified regexp
    const char * m_pSimpleRe;
};

}

#endif
