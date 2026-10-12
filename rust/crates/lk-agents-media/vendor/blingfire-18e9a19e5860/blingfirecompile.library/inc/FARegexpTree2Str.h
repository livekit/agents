/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPTREE2STR_H_
#define _FA_REGEXPTREE2STR_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARegexpTree;

///
/// Builds regular expression text from the regular expression tree; 
/// can treat any node as a root.
///

class FARegexpTree2Str {

public:
    FARegexpTree2Str (FAAllocatorA * pAlloc);

public:

    // sets up original regular expression text
    // offsets from pTree refer to this text
    void SetRegexp (const char * pRegexp, const int Length);
    // regular expression tree
    void SetRegexpTree (const FARegexpTree * pTree);
    // returns 0-terminated text of regular expression
    const char * Process (const int NodeId);

private:
    inline void PutNode (const int NodeId);
    inline const bool NeedBrackets (const int NodeId) const;
    inline void PutLeftBracket (const int NodeId);
    inline void PutRightBracket (const int NodeId);
    void Process_rec (const int NodeId);

private:

    const char * m_pRegexp;
    int m_Length;
    const FARegexpTree * m_pTree;

    FAArray_cont_t < char > m_RegexpText;
};

}

#endif
