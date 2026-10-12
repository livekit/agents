/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WRERULES2TOKENNFA_H_
#define _FA_WRERULES2TOKENNFA_H_

#include "FAConfig.h"
#include "FAChain2Num_hash.h"
#include "FANfaCreator_wre.h"
#include "FARegexpSimplify.h"
#include "FARegexp2Nfa.h"
#include "FANfas2CommonENfa.h"

namespace BlingFire
{

///
/// This class builds Token NFA from a list of WRE rules or a single rule.
///

class FAWRERules2TokenNfa {

public:
    FAWRERules2TokenNfa (FAAllocatorA * pAlloc);

public:
    /// sets up WRE_TYPE, e.g. one of FAFsmConst::WRE_TYPE_*
    void SetType (const int Type);
    /// full or empty TokenText <--> TokenNum map container
    void SetToken2NumMap (FAChain2NumA * pTokens);
    /// add one or more (in case of WRE_TYPE_MOORE) rules
    void AddRule (const char * pWRE, const int Length);
    /// makes the output ready
    void Process ();
    /// returns object into the initial state, frees memory
    void Clear ();
    /// returns constructed token NFA
    const FARSNfaA * GetTokenNfa () const;

private:
    /// WRE type
    int m_Type;
    /// simplifies the expression
    FARegexpSimplify m_simplify;
    /// builds NFA from the WRE rule
    FARegexp2Nfa m_rule2nfa;
    /// single rule NFA
    FANfaCreator_wre m_rule_nfa;
    /// NFA list --> ENFA
    FANfas2CommonENfa m_merge;
};

}

#endif
