/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSER_NEST_T_H_
#define _FA_PARSER_NEST_T_H_

#include "FAConfig.h"
#include "FAParser_base_t.h"
#include "FAResolveMatch_nest.h"
#include "FAState2OwsCA.h"

#include <algorithm>

namespace BlingFire
{

///
/// This parser allows nested results with different constituent names to be 
/// added into a tree at a single pass.
///

template < class Ty >
class FAParser_nest_t : public FAParser_base_t < Ty > {

public:
    FAParser_nest_t (FAAllocatorA * pAlloc);
    ~FAParser_nest_t ();

private:
    const int UpdateResults (
            const int FromPos,
            const int ToPos,
            const int State
        );

    const bool ApplyResults ();

private:
    enum {
        DefActionSize = 3,
    };
    typedef FAParser_base_t < Ty > BASE;
    // conflicts resolver
    FAResolveMatch_nest m_resolver;
};


template < class Ty >
FAParser_nest_t< Ty >::FAParser_nest_t (FAAllocatorA * pAlloc) :
    FAParser_base_t < Ty > (pAlloc),
    m_resolver (pAlloc)
{}


template < class Ty >
FAParser_nest_t< Ty >::~FAParser_nest_t ()
{}


template < class Ty >
const int FAParser_nest_t< Ty >::UpdateResults (
        const int FromPos, 
        const int ToPos, 
        const int State
    )
{
    DebugLogAssert (BASE::m_pState2Rules);

    const int Count = \
        BASE::m_pState2Rules->GetOws (State, BASE::m_pRules, BASE::m_MaxRulesSize);
    DebugLogAssert (0 < Count && Count <= BASE::m_MaxRulesSize);

    for (int i = 0; i < Count; ++i) {

        const int RuleNum = BASE::m_pRules [i];

        const int * pAct;
#ifndef NDEBUG
        const int ActSize = 
#endif
            BASE::m_pRule2Act->Get (RuleNum, &pAct);
        DebugLogAssert (DefActionSize == ActSize && pAct);

        // += Left Context Size
        int From2 = FromPos + pAct [0];
        if (0 > From2) {
            From2 = 0; // due to the included left anchor
        }
        // -= Right Context Size
        int To2 = ToPos - pAct [1];
        if (To2 >= BASE::m_UpperCount) {
            To2 = BASE::m_UpperCount - 1; // due to the included right anchor
        }
        // get action Tag
        const int Tag = pAct [2];

        DebugLogAssert (From2 <= To2 && 0 <= To2 && 0 <= From2);
        DebugLogAssert (FALimits::MinTag <= Tag && FALimits::MaxTag >= Tag);
        DebugLogAssert (0 <= (BASE::m_UpperCount - To2));

        m_resolver.AddResult (Tag, From2, To2);

    } // of for (int i = 0; ...

    return FromPos;
}


template < class Ty >
const bool FAParser_nest_t< Ty >::ApplyResults ()
{
    int From, To, Tag;

    m_resolver.Process ();

    const int Count = m_resolver.GetCount ();

    for (int i = 0; i < Count; ++i) {

        m_resolver.GetResult (i, &Tag, &From, &To);

        DebugLogAssert (FALimits::MinTag <= Tag && FALimits::MaxTag >= Tag);
        BASE::m_pTree->AddNode (-Tag, From, To);
    }

    m_resolver.Clear ();

    return 0 != Count;
}

}

#endif
