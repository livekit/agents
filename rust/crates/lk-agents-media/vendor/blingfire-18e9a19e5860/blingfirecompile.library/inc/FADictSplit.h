/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DICTSPLIT_H_
#define _FA_DICTSPLIT_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_hash.h"
#include "FAMultiMap_ar_uniq.h"

#include <iostream>

namespace BlingFire
{

class FAAllocatorA;

///
/// This class reads digitized dictionary stream (sorted or raw) of Key/Info
/// pairs and splits it into:
///   1. stream of Keys
///   2. KeyNum -> InfoId array
///   3. InfoId -> Info multi-map
/// The KeyNum is a unique Key index in the input stream.
///
/// Raw Mode:      KEY -> RAW DATA  ; no duplicate words
/// Tags Mode:     KEY -> TAGS      ; input sorted by KEY
/// Tag Prob Mode: KEY -> TAG, PROB ; input sorted by KEY
///

class FADictSplit {

public:
    FADictSplit (FAAllocatorA * pAlloc);

public:
    /// sets up key's output stream, can be NULL (no output in this case)
    void SetKeyOs (std::ostream * pOs);
    /// sets up output base (16 by default)
    void SetBase (const int Base);
    /// sets up number of characters per value (4 by default)
    void SetNumSize (const int NumSize);
    /// sets up input mode type: SORTED, RAW, or TP
    void SetMode (const int Mode);
    /// sets up whether InfoIds should be attached to Keys (false by default)
    void SetNoK2I (const bool NoK2I);
    /// base value for InfoId (0 by default)
    void SetInfoIdBase (const int InfoBase);

    /// adds chain of ints representing key -> info pair
    void AddChain (const int * pChain, const int Count);
    /// makes output structures ready
    void Process ();

    /// returns KeyNum -> InfoId array
    const int GetK2I (const int ** ppK2I) const;
    /// returns InfoId -> Info multi map
    const FAMultiMapA * GetI2Info () const;

    /// returns object into the initial state
    void Clear ();

private:
    inline void PrintKey (
            const int * pKey,
            const int KeySize,
            const int SetId    // used only if m_NoK2I is true
        );

    inline void ProcessEntry_raw ();

    inline void UpdateKeys_tags ();
    inline void ProcessEntry_tags ();

    inline void UpdateKeys_tag_prob ();
    inline void ProcessEntry_tag_prob ();

    inline void UpdateKeys_hyph ();
    inline void ProcessEntry_hyph ();

    inline void RevSetMap ();

private:
    // input chain
    const int * m_pChainBuffer;
    int m_ChainSize;
    int m_KeySize;
    int * m_pPrevKey;
    int m_PrevKeySize;
    int m_Base;
    int m_NumSize;
    // output stream for keys
    std::ostream * m_pOs;
    // helpers
    int m_Freq;
    FAArray_cont_t < int > m_Set;
    FAArray_cont_t < int > m_Tags;
    FAArray_cont_t < int > m_Probs;
    FAArray_cont_t < int > m_KeyNum2SetId;
    FAChain2Num_hash m_Sets;
    FAMultiMap_ar_uniq m_SetId2Set;
    FAArray_cont_t < int > m_PrevKey;

    int m_Mode;
    bool m_NoK2I;
    int m_InfoBase;

    enum {
        MaxChainSize = 4096,
    };
};

}

#endif
