/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPTREETOPOGRAPH_H_
#define _FA_REGEXPTREETOPOGRAPH_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

class FARegexpTree;

///
/// This class is an interface between FARegexpTree and FATopoSort_t
///

class FARegexpTreeTopoGraph {

public:
    FARegexpTreeTopoGraph (const FARegexpTree * pTree);

public:
    const int GetNodeCount () const;
    const int GetDstNodes (const int NodeId,
                           __out_ecount(MaxDstNodes) int * pDstNodes,
                           const int MaxDstNodes) const;

private:
    const FARegexpTree * m_pTree;
};

}

#endif
