/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_TAGGEDTEXTSTAT_H_
#define _FA_TAGGEDTEXTSTAT_H_

#include "FAConfig.h"
#include "FAChain2Num_hash.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;
class FATaggedTextCA;

///
/// This class collects different kinds of statistics on tagged text.
///
/// Usage:
///   1. Set*
///   2. foreach S in Corpus do Update*(S) done
///   3. GetStat
///   4. Clear 
///

class FATaggedTextStat {

public:
    FATaggedTextStat (FAAllocatorA * pAlloc);

public:
    /// sets up statistics mask
    void SetStatMask (const int StatMask);
    /// specifies whether case should be ignored
    void SetIgnoreCase (const bool IgnoreCase);
    /// sets up BOS tag, 0 is used by default
    void SetBosTag (const int BosTag);
    /// sets up EOS tag, 0 is used by default
    void SetEosTag (const int EosTag);
    /// sets up BOS word, an empty string is used by default
    void SetBosWord (const int * pBosWord, const unsigned int BosWordLen);
    /// sets up EOS word, an empty string is used by default
    void SetEosWord (const int * pEosWord, const unsigned int EosWordLen);
    /// returns object into the initial state
    void Clear ();

    /// updates all necessary statistics
    void UpdateStat (const FATaggedTextCA * pT);

    /// returns statistics, see FAFsmConst for the range of StatName values
    const FAChain2NumA * GetStat (const int StatName) const;

private:
    /// helper
    static inline void PutChain (
            FAChain2NumA * pMap, 
            const int * pChain, 
            const int Size
        );
    void UpdateTChains(
            FAChain2NumA * pMap,
            const FATaggedTextCA * pT,
            int cGram
        );

    /// updates W statistics from pT
    void UpdateW (const FATaggedTextCA * pT);
    /// updates WW statistics from pT
    void UpdateWW (const FATaggedTextCA * pT);
    /// updates WWW statistics from pT
    void UpdateWWW (const FATaggedTextCA * pT);
    /// updates WT statistics from pT
    void UpdateWT (const FATaggedTextCA * pT);
    /// updates WTT statistics from pT
    void UpdateWTT (const FATaggedTextCA * pT);
    /// updates TWT statistics from pT
    void UpdateTWT (const FATaggedTextCA * pT);
    /// updates WTWT statistics from pT
    void UpdateWTWT (const FATaggedTextCA * pT);
    /// updates T statistics from pT
    void UpdateT (const FATaggedTextCA * pT);
    /// updates TT statistics from pT
    void UpdateTT (const FATaggedTextCA * pT);
    /// updates TTT statistics from pT
    void UpdateTTT (const FATaggedTextCA * pT);
    /// updates TTTT statistics from pT
    void UpdateTTTT (const FATaggedTextCA * pT);
    /// updates W_T statistics from pT
    void UpdateW_T (const FATaggedTextCA * pT);
    /// updates TW statistics from pT
    void UpdateTW (const FATaggedTextCA * pT);

private:
    /// these keep Chain --> Freq mapping
    FAChain2Num_hash m_c2f_w;
    FAChain2Num_hash m_c2f_ww;
    FAChain2Num_hash m_c2f_www;
    FAChain2Num_hash m_c2f_wt;
    FAChain2Num_hash m_c2f_wtt;
    FAChain2Num_hash m_c2f_twt;
    FAChain2Num_hash m_c2f_wtwt;
    FAChain2Num_hash m_c2f_t;
    FAChain2Num_hash m_c2f_tt;
    FAChain2Num_hash m_c2f_ttt;
    FAChain2Num_hash m_c2f_tttt;
    FAChain2Num_hash m_c2f_w_t;
    FAChain2Num_hash m_c2f_tw;

    int m_StatMask;
    bool m_IgnoreCase;
    int m_BosTag;
    int m_EosTag;
    const int * m_pBosWord;
    unsigned int m_BosWordLen;
    const int * m_pEosWord;
    unsigned int m_EosWordLen;

};

}

#endif
