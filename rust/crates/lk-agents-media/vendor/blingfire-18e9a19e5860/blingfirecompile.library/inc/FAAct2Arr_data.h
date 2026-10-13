/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ACT2ARR_DATA_H_
#define _FA_ACT2ARR_DATA_H_

#include "FAConfig.h"
#include "FAAct2ArrA.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_hash.h"

namespace BlingFire
{

class FAAllocatorA;
class FATagSet;

///
/// Parses the action of arbitrary length data into array. The array is added 
/// into the m_data2tag which returns a tag (unique id of the action data).
/// Note: This format makes this action compatible with all the runtimes:
/// FALexTools_t.h and FAWreLexTools_t.h so there is no need to do any changes
/// there if a client application needs different data to be associated with 
/// each matched span.
///
/// Parses "data" actions of the following format:
///   Action ::= [ CxAdjust ] [ Data ] [ _call Tag1 [Tag2 ... ] ]
///   CxAdjust ::= [ LeftCxAdjust ] [ RightCxAdjust ]
///   LeftCxAdjust ::= <-N
///   LeftCxAdjust ::= <+N
///   RightCxAdjust ::= >-N
///   RightCxAdjust ::= >+N
///   Data ::= TagX Data
///   Data ::= Num Data
/// where:
///   TagX -- any known tag
///   Num -- any integer number
///   N -- non-negative integer number
///
/// Examples of valid actions:
///   DATE
///   DATE _call FnGetDateParts
///   DATE _call FnGetMonth FnGetDay FnGetYear
///   _call FnGetMonth FnGetDay FnGetYear
///   <+30 >+30 DATE_WITH_CONTEXT _call FnGetDate
///   11 2010
///   11 2010 _call FnGetDateParts
///   11 2010 _call FnGetMonth FnGetDay FnGetYear
///   <+30 >+30 11 2010
///

class FAAct2Arr_data : public FAAct2ArrA {

public:
    FAAct2Arr_data (FAAllocatorA * pAlloc);

public:
    /// (see FAAct2ArrA for details)
    void SetTagSet (const FATagSet * pTagSet);
    /// (see FAAct2ArrA for details)
    const int Process (const char * pStr, const int Len, const int ** ppArr, 
        int * pLeftCxAdjust, int * pRightCxAdjust);
    // returns the amount of saved data arrays during the parsing
    const int GetDataCount () const;
    // returns a particular data array
    const int GetData (const int Idx, const int ** ppData) const;

private:
    FAArray_cont_t < int > m_arr;
    FAArray_cont_t < int > m_data;
    FAChain2Num_hash m_data2tag;
    const FATagSet * m_pTagSet;
};

}

#endif
