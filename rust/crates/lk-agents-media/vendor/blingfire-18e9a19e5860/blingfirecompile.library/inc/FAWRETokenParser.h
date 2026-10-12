/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WRE_TOKENPARSER_H_
#define _FA_WRE_TOKENPARSER_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FAWREToken;
class FATagSet;

///
/// This class constructs FAWREToken from the input string.
///

class FAWRETokenParser {

public:
    FAWRETokenParser (FAAllocatorA * pAlloc);

public:
    /// input token text
    void SetTokenStr (const char * pTokenStr, const int Len);
    /// output parsed WRE token
    void SetToken (FAWREToken * pToken);
    /// makes the parsing
    void Process  ();

private:
    /// finds the end of what-ever list
    inline const char * FindListEnd (const char * pBegin, const char * pEnd) const;
    /// makes word list parsing
    inline void ParseWordList (const char * pBegin, const char * pEnd);
    /// makes dict list parsing
    inline void ParseRegexpList (const char * pBegin, const char * pEnd);
    /// makes regexp list parsing
    inline void ParseDictList (const char * pBegin, const char * pEnd);
    /// makes tag list parsing
    inline void ParseTagList (const char * pBegin, const char * pEnd);
    /// returns true if symbol at Pos is escaped
    inline static const bool IsEscaped (const char * pBegin, const int Pos);

private:
    /// tagset
    const FATagSet * m_pTagSet;
    /// token string
    const char * m_pTokenStr;
    int m_Len;
    /// token
    FAWREToken * m_pToken;
    // known characters
    enum {
        CharWordListBegin = '"',
        CharRegexpListBegin = '\'',
        CharDictListBegin = '@',
        CharNegative = '!',
        CharListDelimier = '/',
        CharDictDelimier = '@',
        CharDisjunction = '|',
        MaxCharSymbol = 255
    };
    // masks
    enum { 
        WordListBegin = 1,
        RegexpListBegin = 2,
        DictListBegin = 4,
        Negative = 8,
        ListDelimiter = 16,
        DictDelimiter = 32,
        Disjunction = 64
    };
    /// symbol -> info mapping
    FAArray_cont_t < int > m_char2info;
};

}

#endif
