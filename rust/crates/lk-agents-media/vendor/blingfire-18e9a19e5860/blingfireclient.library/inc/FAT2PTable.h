/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_T2PTABLE_H_
#define _FA_T2PTABLE_H_

#include "FAConfig.h"

namespace BlingFire
{

class FATsConfKeeper;

///
/// returns P(T)
///

class FAT2PTable {

public:
    FAT2PTable ();
    ~FAT2PTable ();

public:
    // returns natural log of P(T)
    inline const float GetProb (const int T) const;

    // sets up the configuration
    void SetConf (const FATsConfKeeper * pConf);

private:
    const float * m_pArr;
    int m_Size;
};


inline const float FAT2PTable::GetProb (const int T) const
{
    DebugLogAssert (m_pArr && 0 < m_Size);

    DebugLogAssert (0 < T && T <= m_Size); // assumes all tag probs have been stored

    return m_pArr [T - 1];
}

}

#endif
