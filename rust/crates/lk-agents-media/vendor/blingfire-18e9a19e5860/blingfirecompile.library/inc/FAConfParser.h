/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CONFPARSER_H_
#define _FA_CONFPARSER_H_

#include "FAConfig.h"
#include "FAStr2Int_hash.h"
#include "FAStringTokenizer.h"

#include <iostream>

namespace BlingFire
{

class FAMultiMapA;
class FAAllocatorA;

///
/// This class converts configuration file stream into a multi-map.
///
/// Input:
/// ...
/// # comment
/// [section1]
/// Param1
/// NumParam1 32
/// NumParam2 1
/// StrParam1 str
/// StrParam2 str1
/// StrParam2 str2
/// ...
///
/// Note:
/// The records:
///   StrParam str1
///   StrParam str2
/// are equivalent to:
///   StrParam str1,str2
///


class FAConfParser {

public:
    FAConfParser (FAAllocatorA * pAlloc);

public:
    /// adds section
    void AddSection (
            const char * pSecName,
            const int SecId
        );
    /// adds parameter which does not have value
    void AddParam (
            const char * pParamName, 
            const int ParamId
        );
    /// adds numerical parameter
    void AddNumParam (
            const char * pParamName, 
            const int ParamId
        );
    /// adds paramenter with string value
    void AddStrParam (
            const char * pParamName, 
            const int ParamId,
            const char * pValueStr, 
            const int ValueId
        );

    /// sets up configuration input stream
    void SetConfStream (std::istream * pConfIs);
    /// sets up output container
    void SetConfMap (FAMultiMapA * pConfMap);
    /// makes transformation
    void Process ();

private:
    const bool ProcessSection ();
    const bool ProcessParam ();
    const bool ProcessNumParam ();
    const bool ProcessStrParam ();

private:
    // input stream
    std::istream * m_pConfIs;
    // output map
    FAMultiMapA * m_pConfMap;
    // Keeps:
    // "[SecName]" -> SecId
    // "Param" -> ParamId
    // "Num.Param" -> ParamId
    // "Str.Param" -> ParamId
    // "Param Value" -> ValueId
    FAStr2Int_hash m_str2id;
    // helpers
    FAStringTokenizer m_tokenizer;
    FAStringTokenizer m_tokenizer2;
    int m_CurrSecId;
    const char * m_pFirstToken;
    int m_FirstTokenLen;

};

}

#endif
