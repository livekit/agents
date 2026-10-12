/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRBR2SYMBOL_H_
#define _FA_TRBR2SYMBOL_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAToken.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This class changes triangular brackets into artificial FARegexpTree::SYMBOL
/// tokens + ordinary brackets so the parser will treat them as an ordinary 
/// symbols. FANfaCreator_*_trbr(s) use these symbols to create special arcs.
///
/// 1. trbr-regexp:
///   <1 Re1 > <2 Re2 > 
/// 2. lexer's output:
///   LTRBR:'<1' Re1 RTRBR:'>' LTRBR:'<2' Re2 RTRBR:'>'
/// 3. tokens substitution:
///   LBR SYMBOL:<1,-1> LBR Re1 RBR SYMBOL::<1,-2> RBR
///   LBR SYMBOL:<2,-1> LBR Re2 RBR SYMBOL:<2,-2> RBR
/// 4. tokens sequence looks as if someone wrote:
///   ( '<1' ( Re1 ) '1>' ) ( '<2' ( Re2 ) '2>' )
///
/// SYMBOL:<1,-1> - token type symbol, offset == 1, length == -1
///

class FATrBr2Symbol {

public:
    FATrBr2Symbol (FAAllocatorA * pAlloc);

public:
    // sets up regular expression text
    void SetRegexp (const char * pRegexp, const int Length);
    // sets up input/output tokens contaner
    void SetTokens (FAArray_cont_t < FAToken > * pTokens);
    // makes processing
    void Process ();

private:
    // regexp text
    const char * m_pRegexp;
    int m_Length;
    // input/output container
    FAArray_cont_t < FAToken > * m_pTokens;

    FAArray_cont_t < int > m_stack;
    FAArray_cont_t < FAToken > m_tmp_tokens;

    static const FAToken m_left_br;
    static const FAToken m_right_br;
};

}

#endif
