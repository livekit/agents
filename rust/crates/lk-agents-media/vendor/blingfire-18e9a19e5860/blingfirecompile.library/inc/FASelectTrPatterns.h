/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SELECTTRPATTERNS_H_
#define _FA_SELECTTRPATTERNS_H_

#include "FAConfig.h"
#include "FAMultiMap_judy.h"
#include "FAArray_cont_t.h"
#include "FAMphInterpretTools_t.h"

namespace BlingFire
{

class FARSDfaCA;
class FAMealyDfaCA;
class FAMultiMapCA;
class FAOw2IwCA;
class FAArrayCA;
class FAAllocatorA;

///
/// This class makes patterns selection
///

class FASelectTrPatterns {

public:
    FASelectTrPatterns (FAAllocatorA * pAlloc);
    virtual ~FASelectTrPatterns ();

public:
    /// extracts patterns for hyphenating positions only, for all by default
    void SetNoEmpty (const bool NoEmpty);
    /// sets up patterns storage FSM
    void SetFsm (
            const FARSDfaCA * pDfa, 
            const FAMealyDfaCA * pMealy, 
            const FAOw2IwCA * pOw2Iw
        );
    /// sets up patterns storage K2I array
    void SetK2I (const FAArrayCA * pK2I);
    /// sets up patterns storage I2Info map
    void SetI2Info (const FAMultiMapCA * pI2Info);
    /// extracts "best" patterns for the given Iws/Ows input pair
    void AddIwsOws (const int * pIws, const int * pOws, const int Count);
    /// finishes processing
    void Process ();

public:
    /// this callback returns all selected patterns, on-by-one
    virtual void PutPattern (
            const int * pIws, 
            const int * pOws, 
            const int Count
        );
    /// callback which returns unsolved inputs, 
    ///   (unsolved Ows represented by their negative values)
    virtual void PutUnsolved (
            const int * pIws, 
            const int * pOws, 
            const int Count
        );
    /// callback which returns inputs causing conflicts
    /// Note: conflicts can occure only if not 100% patterns have been built
    virtual void PutConflict (
            const int * pIws, 
            const int Count,
            const int Pos
        );

private:
    /// validate patterns by the current input and identify unsolved
    void CalcCover (const int * pIws, const int * pOws, const int Count);
    /// select best
    void UpdateBest (const int * pOws, const int Count);
    /// returns the "don't-care" symbol count
    inline static const int GetDcCount (const int * pOws, const int Count);
    /// returns true if pattern Id1 is better than pattern Id2
    const bool Better (const int Id1, const int Id2) const;

private:
    /// input patterns storage
    const FARSDfaCA * m_pDfa;
    const FAMealyDfaCA * m_pMealy;
    const FAArrayCA * m_pK2I;
    const FAMultiMapCA * m_pI2Info;

    /// a set of patterns matched at the given position
    FAMultiMap_judy m_pos2ids_ends;
    /// a set of patterns covering given position
    FAMultiMap_judy m_pos2ids_cover;
    /// a set of selected patterns
    FAArray_cont_t < int > m_used_ids;
    /// tmp array for resulting Iws/Ows, etc.
    FAArray_cont_t < int > m_tmp;
    /// mph tools, PAT <--> ID
    FAMphInterpretTools_t < int > m_mph_tools;

    bool m_NoEmpty;
};

}

#endif
