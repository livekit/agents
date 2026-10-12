/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSETREE_MEMLFP_H_
#define _FA_PARSETREE_MEMLFP_H_
#ifndef BLING_FIRE_NOAP

#include "FAConfig.h"
#include "FAParseTreeA.h"
#include "MemLfp.h"

namespace BlingFire
{

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

class FAParseTree_memlfp : public FAParseTreeA {

public:
    FAParseTree_memlfp (CMemLfpManager * pMemMgr);
    virtual ~FAParseTree_memlfp ();

public:
    /// returns node ids of nodes on top of the tree
    /// if the tree is A B(( C )) D(( E )) F then the output is ids of A B D F
    const int GetUpperNodes (const int ** ppNodes) const;
    /// returns node labels on top of the tree
    /// if the tree is A B(( C )) D(( E )) F then the output is A B D F
    const int GetUpperLabels (const int ** ppLabels) const;
    /// given the node return its right sibling or -1 if it does not exist
    const int GetNext (const int Node) const;
    /// given the node return its left most child or -1 if it does not exist
    const int GetChild (const int Node) const;
    /// given the node returns its label
    const int GetLabel (const int Node) const;

public:
    /// initializes the tree to contain Count leafs (with no parents)
    void Init (const int Count);
    /// adds a new node and makes all upper nodes between positions FromPos 
    /// and ToPos its children
    void AddNode (const int Label, const int FromPos, const int ToPos);
    /// makes sure the arrays of upper nodes and lables are up to date
    /// (this method has to be called after some modifications has been done)
    void Update ();

private:
    struct FANodeData {
        int m_Next;   // right neighbour, -1 if no right neighbour
        int m_Child;  // left child, -1 if no left child
        int m_Label;
    };
    /// node idx -> data
    CMemLfpGrowArrayT < FANodeData > m_i2n;
    /// upper level position -> node idx
    CMemLfpGrowArrayT < int > m_p2i;
    /// upper level position -> node label (redundant but speeds things up)
    CMemLfpGrowArrayT < int > m_p2l;
};

}

#endif
#endif
