/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSETREE_H_
#define _FA_PARSETREE_H_

#include "FAConfig.h"
#include "FAParseTreeA.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Parsing tree container class.
///
/// Note:
///  The tree should be updated as follows:
///    0. Init
///    [ 1. GetUpper* ]    ; getting chain to parse
///    2. AddNode          ; setting up first results from parser
///      ...
///    2. AddNode
///    3. Update           ; making ready
///    4. GetUpper*        ; getting chain to parse
///    5. AddNode          ; setting up next results from parser
///      ...
///    5. AddNode          ; ...
///    6. Update
///    7. GetUpper*
///      ...
///

class FAParseTree : public FAParseTreeA {

public:
    FAParseTree (FAAllocatorA * pAlloc);
    virtual ~FAParseTree ();

public:
    const int GetUpperNodes (const int ** ppNodes) const;
    const int GetUpperLabels (const int ** ppLabels) const;
    const int GetNext (const int Node) const;
    const int GetChild (const int Node) const;
    const int GetLabel (const int Node) const;

public:
    void Init (const int Count);
    void AddNode (const int Label, const int FromPos, const int ToPos);
    void Update ();

private:
    struct FANodeData {
        int m_Next;   // right neighbour, -1 if no right neighbour
        int m_Child;  // left child, -1 if no left child
        int m_Label;
    };
    /// node idx -> data
    FAArray_cont_t < FANodeData > m_i2n;
    /// upper level position -> node idx
    FAArray_cont_t < int > m_p2i;
    /// upper level position -> node label (redundant but speeds things up)
    FAArray_cont_t < int > m_p2l;
};

}

#endif
