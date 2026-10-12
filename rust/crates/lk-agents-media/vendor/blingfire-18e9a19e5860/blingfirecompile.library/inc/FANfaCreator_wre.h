/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_NFA_CREATOR_WRE_H_
#define _FA_NFA_CREATOR_WRE_H_

#include "FAConfig.h"
#include "FANfaCreator_base.h"

namespace BlingFire
{

class FAAllocatorA;
class FAChain2NumA;


class FANfaCreator_wre : public FANfaCreator_base {

public:
    FANfaCreator_wre (FAAllocatorA * pAlloc);

public:
    /// sets up the base for the output nfa's Iws
    void SetBaseIw (const int BaseIw);
    /// sets up token -> num mapping
    void SetToken2NumMap (FAChain2NumA * pToken2Num);
    /// adds a rtansition
    void SetTransition (const int FromState,
                        const int ToState,
                        const int LabelOffset,
                        const int LabelLength);

private:
    /// assigns a unique identifer to the token
    inline const int Token2Num (const char * pStr, const int Len);

private:

    /// mapping from token string to its numerical identifer
    FAChain2NumA * m_pToken2Num;
    /// tmp chain storage
    FAArray_cont_t < int > m_arr;
    /// the base value for all token identifers
    int m_BaseIw;
};

}

#endif
