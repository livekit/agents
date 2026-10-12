/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SUMMTAGSCORES_H_
#define _FA_SUMMTAGSCORES_H_

#include "FAConfig.h"
#include "FAFsmConst.h"

namespace BlingFire
{

///
/// Calculates a combined score for each input tag of the 
/// 
///

class FASummTagScores {

public:
    FASummTagScores ();
    ~FASummTagScores ();

public:
    /// sets up score for the unfound/unspecified tags
    void SetUnkScore (const float Score);
    /// sets up maximum tag value, should be called before Process
    void SetMaxTag (const int MaxTag);

public:
    /// call this function for each new array of tags and scores
    void AddScores (const int * pTags, const float * pScores, const int Count);
    /// call this to make results ready
    void Process ();
    /// returns object into the initial state
    void Clear ();

    /// returns two arrays one is a mapping Tag --> Score and 
    /// the other (parallel) is a mapping Tag --> Count
    /// Note: the Process () should be called before this function can be used
    const int GetScores (const float ** ppScores, const int ** ppCounts) const;

    /// returns total count, the amount of found n-grams
    const int GetTotalCount () const;

private:

    enum {
        DefUnkScore = 2 * FAFsmConst::MIN_LOG_PROB,
    };

    int m_TotalCount;
    int * m_pCounts;
    float * m_pScores;
    int m_MaxTag;
    float m_UnkScore;
};

}

#endif
