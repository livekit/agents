/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_COLOR_GRAPH_T_H_
#define _FA_COLOR_GRAPH_T_H_

#include "FAConfig.h"
#include "FAMultiMap_judy.h"
#include "FAArray_cont_t.h"
#include "FASetUtils.h"
#include "FAUtils.h"

namespace BlingFire
{

///
/// This algorithm tries to calculate minimal coloring such that no
/// two adjacent vertices have the same color.
///
/// _TGraph should have:
///   1. const int GetVertices (const int ** ppV) const;
///   2. const int GetArcCount () const;
///   3. void GetArc (const int Num, int * pFrom, int * pTo) const;
/// methods.
///

template < class _TGraph >
class FAColorGraph_t {

public:
    FAColorGraph_t (FAAllocatorA * pAlloc);

public:
    /// sets up intput graph
    void SetGraph (const _TGraph * pG);
    /// makes the coloring
    void Process ();
    /// returns mapping pV2C [V] == C, or -1 if V not \in G 
    /// return value is the size of pV2C array
    const int GetColorMap (const int ** ppV2C) const;
    /// returns object into the initial state
    void Clear ();

private:
    inline void Prepare ();
    inline void BuildAdjacent ();
    inline const int FindColor (const int V) const;
    inline void AddNewColor (const int V);
    inline void AddToColor (const int C, const int V);
    inline void Color ();

private:
    /// graph
    const _TGraph * m_pG;
    /// vertex -> { adjacent vertices }
    FAMultiMap_judy m_v2adj;
    /// class -> { adjacent vertices }
    FASetUtils m_c2adj;
    /// vertex -> color mapping
    FAArray_cont_t < int > m_v2c;
    /// max color
    int m_MaxColor;
};


template < class _TGraph >
FAColorGraph_t< _TGraph >::FAColorGraph_t (FAAllocatorA * pAlloc) :
    m_pG (NULL),
    m_c2adj (pAlloc),
    m_MaxColor (-1)
{
    m_v2adj.SetAllocator (pAlloc);

    m_v2c.SetAllocator (pAlloc);
    m_v2c.Create ();
}


template < class _TGraph >
void FAColorGraph_t< _TGraph >::Clear ()
{
    m_v2adj.Clear ();
    m_c2adj.Clear ();
    m_v2c.Clear ();
    m_v2c.Create ();
}


template < class _TGraph >
void FAColorGraph_t< _TGraph >::SetGraph (const _TGraph * pG)
{
    m_pG = pG;
}


template < class _TGraph >
void FAColorGraph_t< _TGraph >::Prepare ()
{
    DebugLogAssert (m_pG);

    m_v2adj.Clear ();
    m_MaxColor = -1;
    m_v2c.resize (0);

    const int * pVs;
    const int Count = m_pG->GetVertices (&pVs);
    DebugLogAssert (0 < Count && pVs);
    DebugLogAssert (FAIsSortUniqed (pVs, Count));
    DebugLogAssert (0 <= *pVs);

    m_c2adj.SetResCount (Count);

    const int MaxV = pVs [Count - 1];
    m_v2c.resize (MaxV + 1);
    memset (m_v2c.begin (), -1, sizeof (int) * (MaxV + 1));
}


template < class _TGraph >
void FAColorGraph_t< _TGraph >::BuildAdjacent ()
{
    DebugLogAssert (m_pG);

    int From;
    int To;

    const int ArcCount = m_pG->GetArcCount ();

    for (int i = 0; i < ArcCount; ++i) {

        m_pG->GetArc (i, &From, &To);
        DebugLogAssert (From != To);

        m_v2adj.Add (From, To);
        m_v2adj.Add (To, From);
    }

    m_v2adj.SortUniq ();
}


template < class _TGraph >
const int FAColorGraph_t< _TGraph >::FindColor (const int V) const
{
    for (int c = 0; c <= m_MaxColor; ++c) {

        const int * pAdj;
        const int Count = m_c2adj.GetRes (&pAdj, c);
        DebugLogAssert (FAIsSortUniqed (pAdj, Count));

        if (-1 == FAFind_log (pAdj, Count, V)) {
            return c;
        }
    }

    return -1;
}


template < class _TGraph >
void FAColorGraph_t< _TGraph >::AddNewColor (const int V)
{
    DebugLogAssert (0 <= V && m_v2c.size () > (unsigned int) V);

    m_v2c [V] = ++m_MaxColor;

    const int * pAdj;
    const int Count = m_v2adj.Get (V, &pAdj);

    if (0 < Count) {

        DebugLogAssert (FAIsSortUniqed (pAdj, Count));
        m_c2adj.SetRes (pAdj, Count, m_MaxColor);

    } else {

        m_c2adj.SetRes (NULL, 0, m_MaxColor);
    }
}


template < class _TGraph >
void FAColorGraph_t< _TGraph >::AddToColor (const int C, const int V)
{
    DebugLogAssert (0 <= C && m_MaxColor >= C);
    DebugLogAssert (0 <= V && m_v2c.size () > (unsigned int) V);

    m_v2c [V] = C;

    const int * pAdj;
    const int Count = m_v2adj.Get (V, &pAdj);

    if (0 < Count) {

        DebugLogAssert (FAIsSortUniqed (pAdj, Count));
        m_c2adj.SelfUnion (pAdj, Count, C);
    }
}


template < class _TGraph >
void FAColorGraph_t< _TGraph >::Color ()
{
    DebugLogAssert (m_pG);

    const int * pVs;
    const int Count = m_pG->GetVertices (&pVs);

    for (int i = 0; i < Count; ++i) {

        DebugLogAssert (pVs);
        const int V = pVs [i];

        const int C = FindColor (V);

        if (-1 == C) {
            AddNewColor (V);
        } else {
            AddToColor (C, V);
        }
    }
}


template < class _TGraph >
void FAColorGraph_t< _TGraph >::Process ()
{
    DebugLogAssert (m_pG);

    Prepare ();
    BuildAdjacent ();
    Color ();
}


template < class _TGraph >
const int FAColorGraph_t< _TGraph >::GetColorMap (const int ** ppV2C) const
{
    DebugLogAssert (0 < m_MaxColor);
    DebugLogAssert (ppV2C);

    *ppV2C = m_v2c.begin ();
    return m_v2c.size ();
}

}

#endif
