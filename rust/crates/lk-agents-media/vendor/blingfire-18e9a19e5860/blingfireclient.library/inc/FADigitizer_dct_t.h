/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_DIGITIZER_DCT_T_H_
#define _FA_DIGITIZER_DCT_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAArrayCA.h"
#include "FADictInterpreter_t.h"

namespace BlingFire
{

///
/// Assigns a digitial value to the input chain acording the tag dictionary
/// and SetId --> Ow map.
/// 

template < class Ty >
class FADigitizer_dct_t {

public:
    FADigitizer_dct_t ();

public:
    /// sets up tag dictionary (needed for Process1 only)
    void SetTagDict (const FADictInterpreter_t <Ty> * pTagDict);
    /// sets up SetId --> Ow map
    void SetSet2Ow (const FAArrayCA * pSet2Ow);
    /// sets up AnyOw, FAFsmConst::IW_ANY by default
    void SetAnyOw (const int AnyOw);

    /// returns Ow for the given Word
    const int Process (const Ty * pWord, const int Size) const;
    /// returns Ow for SetId, if it has already been looked-up
    const int Process (const int SetId) const;

private:
    const FADictInterpreter_t <Ty> * m_pTagDict;
    const FAArrayCA * m_pSet2Ow;
    int m_IdCount;
    int m_AnyOw;
};


template < class Ty >
FADigitizer_dct_t< Ty >::
    FADigitizer_dct_t () :
        m_pTagDict (NULL),
        m_pSet2Ow (NULL),
        m_IdCount (0),
        m_AnyOw (FAFsmConst::IW_ANY)
{}


template < class Ty >
void FADigitizer_dct_t< Ty >::
    SetTagDict (const FADictInterpreter_t <Ty> * pTagDict)
{
    m_pTagDict = pTagDict;
}


template < class Ty >
void FADigitizer_dct_t< Ty >::
    SetSet2Ow (const FAArrayCA * pSet2Ow)
{
    m_pSet2Ow = pSet2Ow;

    if (pSet2Ow) {
        m_IdCount = pSet2Ow->GetCount ();
    } else {
        m_IdCount = 0;
    }
}


template < class Ty >
void FADigitizer_dct_t< Ty >::
    SetAnyOw (const int AnyOw)
{
    m_AnyOw = AnyOw;
}


template < class Ty >
const int FADigitizer_dct_t< Ty >::
    Process (const Ty * pWord, const int Size) const
{
    DebugLogAssert (m_pTagDict && m_pSet2Ow);
    DebugLogAssert (0 < m_IdCount);

    const int SetId = m_pTagDict->GetInfoId (pWord, Size);

    if (0 > SetId || SetId >= m_IdCount) {
        return m_AnyOw;
    } else {
        const int Ow = m_pSet2Ow->GetAt (SetId);
        return Ow;
    }
}


template < class Ty >
const int FADigitizer_dct_t< Ty >::Process (const int SetId) const
{
    DebugLogAssert (m_pSet2Ow);
    DebugLogAssert (0 < m_IdCount);

    if (0 > SetId || SetId >= m_IdCount) {
        return m_AnyOw;
    } else {
        const int Ow = m_pSet2Ow->GetAt (SetId);
        return Ow;
    }
}

}

#endif
