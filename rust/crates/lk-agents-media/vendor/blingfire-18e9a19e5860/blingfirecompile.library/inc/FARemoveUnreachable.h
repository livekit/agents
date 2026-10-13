/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REMOVE_UNREACHABLE_H_
#define _FA_REMOVE_UNREACHABLE_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAMultiMap_judy.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;


///
/// FARemoveUnreachable - removes all transitions from unreachable states
///

class FARemoveUnreachable {

public:
    FARemoveUnreachable (FAAllocatorA * pAlloc);

public:
    /// nfa for modification
    void SetNfa (FARSNfaA * pNfa);
    /// sets up removal type, true by default
    void SetRemoveIniUnreach (const bool RemoveIniUnreach);
    /// makes removal itself
    void Process ();

private:
    /// makes processor ready
    inline void Clear ();
    /// build arcs map
    inline void BuildArcs ();
    /// finds unreachable states
    inline void FindUnreach ();
    /// removes transitions from the unreachable states
    inline void RemoveTransitions ();

private:
    /// input NFA
    FARSNfaA * m_pNfa;
    /// 1 - reachable, 0 - unreachable
    FAArray_cont_t < unsigned char > m_state2info;
    /// traversal stack
    FAArray_cont_t < int > m_stack;
    /// Iws
    FAArray_cont_t < int > m_iws;
    /// arcs (or reversed arcs)
    FAMultiMap_judy m_arcs;
    /// final/initial-unreachable removal flag, true by default
    bool m_RemoveIniUnreach;
};

}

#endif
