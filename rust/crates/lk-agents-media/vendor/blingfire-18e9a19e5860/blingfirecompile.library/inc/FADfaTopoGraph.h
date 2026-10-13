/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DFATOPOGRAPH_H_
#define _FA_DFATOPOGRAPH_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaA;

///
/// FATopoSort_t-compatible graph for the given FARSDfaA interface.
///

class FADfaTopoGraph {

public:
    FADfaTopoGraph (FAAllocatorA * pAlloc);

public:
    void SetDfa (const FARSDfaA * pInDfa);
    const int GetNodeCount () const;
    const int GetDstNodes (
            const int Node,
            __out_ecount(DstNodesSize) int * pDstNodes,
            const int DstNodesSize
        ) const;

private:
    const FARSDfaA * m_pInDfa;
    FAAllocatorA * m_pAlloc;
};

}

#endif
