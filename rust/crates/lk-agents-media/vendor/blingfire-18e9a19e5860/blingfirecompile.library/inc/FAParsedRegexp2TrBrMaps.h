/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSEDREGEXP2TRBRMAPS_H_
#define _FA_PARSEDREGEXP2TRBRMAPS_H_

#include "FAConfig.h"
#include "FASetUtils.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARegexpTree;
class FARegexpTree2Funcs;
class FAMultiMapA;
class FAMapA;

///
/// This class builds maps necessary for sub-match extraction.
///

class FAParsedRegexp2TrBrMaps {

public:
    FAParsedRegexp2TrBrMaps (FAAllocatorA * pAlloc);

public:
    // input regexp tree
    void SetRegexpTree (const FARegexpTree * pRegexpTree);
    // input regexp functions container
    void SetRegexpFuncs (const FARegexpTree2Funcs * pRegexpFuncs);
    // sets up input position equivalence classes, 
    // if NULL all positions are unique
    void SetPos2Class (const FAMapA * pPos2Class);
    // output, maps regexp position to TrBr array which starts in this position
    // (including this position)
    void SetStartMap (FAMultiMapA * pStartTrBr);
    // output, maps regexp position to TrBr set which ends in this position
    // (not including this position)
    void SetEndMap (FAMultiMapA * pEndTrBr);
    // calculates output maps
    void Process ();

private:

    inline const int Pos2Class (const int Pos) const;

    void Prepare ();
    void BuildStartMap ();
    void BuildEndMap ();

private:

    const FARegexpTree * m_pRegexpTree;
    const FARegexpTree2Funcs * m_pRegexpFuncs;
    const FAMapA * m_pPos2Class;

    FAMultiMapA * m_pStartTrBr;
    FAMultiMapA * m_pEndTrBr;

    FASetUtils m_sets;
    FAArray_cont_t < int > m_pos_arr;
    FAArray_cont_t < int > m_node_arr;

    int m_MaxPos;

private:

    // compares node_ids by offsets
    class TTrBrOffsetCmp {
    public:
        TTrBrOffsetCmp (const FARegexpTree * pRegexpTree);

    public:
        const bool operator() (const int NodeId1, const int NodeId2) const;

    private:
        const FARegexpTree * m_pRegexpTree;
    };
};

}

#endif
