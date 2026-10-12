/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TSCONFKEEPER_H_
#define _FA_TSCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FAArrayCA;
class FAArray_pack;

///
/// Keeps data for P(T), P(T|T-1), and P(T|T-2,T-1) functions
///

class FATsConfKeeper {

public:
    FATsConfKeeper ();
    ~FATsConfKeeper ();

public:
    /// this LDB will be used to get the data from
    void SetLDB (const FALDB * pLDB);
    /// initialization vector
    void Init (const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    // returns packed array of integer values, if available
    const FAArrayCA * GetArr () const;
    // returns unpacked array of float values, if available
    const int GetArr (const float ** ppArr) const;
    // returns true if logarithmic scale was used
    const bool GetIsLog () const;
    // the numrical value corresponding to 1 of the prob, prob guesser only
    const int GetMaxProb () const;
    // returns maximum tag used
    const int GetMaxTag () const;    

private:
    /// pointer to the LDB
    const FALDB * m_pLDB;
    /// compressed/packed array object, not NULL if the data are packed
    FAArray_pack * m_pArr;
    /// indicates wether ln probs are stored
    bool m_LogScale;
    /// maximum integer probability value
    int m_MaxProb;
    /// maximum possible tag value
    int m_MaxTag;
    /// not compressed array of probability values
    const float * m_FloatArr;
    /// m_FloatArr array size
    int m_FloatArrSize;
};

}

#endif
