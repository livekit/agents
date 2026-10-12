/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RESOLVEMATCH_NEST_H_
#define _FA_RESOLVEMATCH_NEST_H_

#include "FAConfig.h"
#include "FAResolveMatchA.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// 1. Removes overlappings (prefers lefttest match)
/// 2. Removes inclusions of the same rule
///

class FAResolveMatch_nest : public FAResolveMatchA {

public:
    FAResolveMatch_nest (FAAllocatorA * pAlloc);
    virtual ~FAResolveMatch_nest ();

public:
    void AddResult (const int Rule, const int From, const int To);
    void Process ();
    const int GetCount () const;
    void GetResult (const int i, int * pRule, int * pFrom, int * pTo) const;
    void Clear ();

private:
    inline void SortResults ();
    inline void FilterResults_uniq ();
    inline void FilterResults_overlap_include ();

private:
    /// keeps packed results from a single parser pass
    FAArray_cont_t < unsigned int > m_results;
    /// needed for same rule inclusion removal
    FAArray_cont_t < int > m_rule2pos;
    /// inclusion stack
    FAArray_cont_t < unsigned int > m_stack;

    enum {
        DefMaxPos = 0xFF,    // local limitation
        DefMaxRule = 0xFFFF, // local limitation
    };
};

}

#endif
