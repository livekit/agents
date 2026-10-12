/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_REGEXPREVERSE_H_
#define _FA_REGEXPREVERSE_H_

#include "FAConfig.h"
#include "FAToken.h"
#include "FAArray_cont_t.h"
#include "FARegexpTree.h"
#include "FARegexpParser_msyacc.h"
#include "FARegexpTree2Str.h"
#include "FARegexpLexer_triv.h"
#include "FARegexpLexer_char.h"
#include "FARegexpLexer_wre.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Builds a reversed regular expression.
///

class FARegexpReverse {

public:
    FARegexpReverse (FAAllocatorA * pAlloc);

public:
    // sets up input regular expression text
    void SetRegexp (const char * pRegexp, const int Length);
    // sets up the label type
    void SetLabelType (const int LabelType);
    // indicates whether, character-based regular expression is in UTF-8
    // no need to call for other types of regular expression, false by default
    void SetUseUtf8 (const bool UseUtf8);
    // makes reversal
    void Process ();
    // returns 0-terminated text of the reversed regular expression
    const char * GetRegexp () const;
    // returns resulting tree
    const FARegexpTree * GetRegexpTree () const;

private:
    void Clear ();
    void PutError ();

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
    // builds regexp text from the tree
    FARegexpTree2Str m_tree2re;
    // pointer to simplified regexp
    const char * m_pRevRe;
};

}

#endif
