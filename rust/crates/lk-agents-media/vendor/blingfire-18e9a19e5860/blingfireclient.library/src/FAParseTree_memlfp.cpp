/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"

#ifndef BLING_FIRE_NOAP
#include "FAConfig.h"
#include "FAParseTree_memlfp.h"

namespace BlingFire
{

FAParseTree_memlfp::FAParseTree_memlfp (CMemLfpManager * pMemMgr)
{
    m_i2n.Reset (pMemMgr);
    m_p2i.Reset (pMemMgr);
    m_p2l.Reset (pMemMgr);
}


FAParseTree_memlfp::~FAParseTree_memlfp ()
{}


void FAParseTree_memlfp::Init (const int Count)
{
    DebugLogAssert (0 <= Count);

    m_i2n.Clear ();
    m_p2i.Clear ();
    m_p2l.Clear ();

    if (0 < Count) {

        FANodeData * pI2N = m_i2n.AddElements(Count);
        int * pP2I = m_p2i.AddElements(Count);
        int * pP2L = m_p2l.AddElements(Count);
        LogAssert (pI2N && pP2I && pP2L);

        const int MaxPos = Count - 1;

        // i here has sense of both position and node idx
        for (int i = 0; i < MaxPos; ++i) {

            // get node data
            FANodeData & NodeData = pI2N [i];

            // setup node data
            NodeData.m_Label = i;
            NodeData.m_Next = i + 1;
            NodeData.m_Child = -1;

            // setup pos -> node idx map
            pP2I [i] = i;
            // setup pos -> label
            pP2L [i] = i;
        }

        // get node data
        FANodeData & NodeData = pI2N [MaxPos];

        // setup node data
        NodeData.m_Label = MaxPos;
        NodeData.m_Next = -1;
        NodeData.m_Child = -1;

        // setup pos -> node idx map
        pP2I [MaxPos] = MaxPos;
        // setup pos -> label
        pP2L [MaxPos] = MaxPos;

    } // of if (0 < Count) ...
}


void FAParseTree_memlfp::AddNode (const int Label, const int FromPos, const int ToPos)
{
    DebugLogAssert(0 <= FromPos && m_p2i.GetSize() > (unsigned) FromPos);
    DebugLogAssert(0 <= ToPos && m_p2i.GetSize() > (unsigned) ToPos);

    // allocate a new node
    const int NewNode = (int) m_i2n.GetSize ();
    FANodeData * pNewNodeData = m_i2n.AddElement();
    LogAssert (pNewNodeData);

    // get from-node and to-node
    const int ToNode = m_p2i [ToPos];
    const int FromNode = m_p2i [FromPos];

    // this is needed only if inclusion of consituents is allowed at single
    // stage of parsing (higher level constituents should be added first)
    m_p2i [FromPos] = NewNode;

    DebugLogAssert(0 <= FromNode && m_i2n.GetSize() > (unsigned) FromNode);
    DebugLogAssert(0 <= ToNode && m_i2n.GetSize() > (unsigned) ToNode);

    // get nodes' data
    FANodeData & NewNodeData = m_i2n [NewNode];
    FANodeData & FromNodeData = m_i2n [FromNode];
    FANodeData & ToNodeData = m_i2n [ToNode];

    // copy the content of a FromNode into a NewNode
    NewNodeData = FromNodeData;

    // setup a new label for an old FromNode
    FromNodeData.m_Label = Label;
    // setup a new next-node link for an old FromNode
    FromNodeData.m_Next = ToNodeData.m_Next;
    // setup a new child-node link for an old FromNode
    FromNodeData.m_Child = NewNode;

    // setup -1 as a Next node for ToNode
    if (FromPos != ToPos)
        ToNodeData.m_Next = -1;
    else
        NewNodeData.m_Next = -1;
}


void FAParseTree_memlfp::Update ()
{
    DebugLogAssert (0 < m_i2n.GetSize ());

    m_p2i.Clear ();
    m_p2l.Clear ();

    int TopCount = 0;
    int NodeIdx = 0;

    /// interate thru all the top nodes and count them
    while (-1 != NodeIdx) {
        DebugLogAssert(0 <= NodeIdx && m_i2n.GetSize() > (unsigned) NodeIdx);
        const FANodeData & NodeData = m_i2n [NodeIdx];
        TopCount++;
        NodeIdx = NodeData.m_Next;
    }

    /// allocate TopCount elements
    int * pP2I = m_p2i.AddElements (TopCount);
    int * pP2L = m_p2l.AddElements (TopCount);
    LogAssert (pP2I && pP2L);

    NodeIdx = 0;

    /// interate thru all the top nodes and fill in m_p2i and m_p2l
    while (-1 != NodeIdx) {
        DebugLogAssert(0 <= NodeIdx && m_i2n.GetSize() > (unsigned) NodeIdx);
        const FANodeData & NodeData = m_i2n [NodeIdx];
        *pP2I++ = NodeIdx;
        *pP2L++ = NodeData.m_Label;
        NodeIdx = NodeData.m_Next;
    }
}


const int FAParseTree_memlfp::GetUpperNodes (const int ** ppNodes) const
{
    DebugLogAssert (ppNodes);
    DebugLogAssert (m_p2i.GetSize () == m_p2l.GetSize ());

    *ppNodes = m_p2i.GetData ();
    const int Count = (int) m_p2i.GetSize ();

    return Count;
}


const int FAParseTree_memlfp::GetUpperLabels (const int ** ppLabels) const
{
    DebugLogAssert (ppLabels);
    DebugLogAssert (m_p2i.GetSize () == m_p2l.GetSize ());

    *ppLabels = m_p2l.GetData ();
    const int Count = (int) m_p2l.GetSize ();

    return Count;
}


const int FAParseTree_memlfp::GetNext (const int Node) const
{
    DebugLogAssert(0 <= Node && m_i2n.GetSize() > (unsigned) Node);

    const FANodeData & NodeData = m_i2n [Node];
    return NodeData.m_Next;
}


const int FAParseTree_memlfp::GetChild (const int Node) const
{
    DebugLogAssert(0 <= Node && m_i2n.GetSize() > (unsigned) Node);

    const FANodeData & NodeData = m_i2n [Node];
    return NodeData.m_Child;
}


const int FAParseTree_memlfp::GetLabel (const int Node) const
{
    DebugLogAssert(0 <= Node && m_i2n.GetSize() > (unsigned) Node);

    const FANodeData & NodeData = m_i2n [Node];
    return NodeData.m_Label;
}

}

#endif
