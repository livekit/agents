/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_NFACREATOR_CHAR_H_
#define _FA_NFACREATOR_CHAR_H_

#include "FAConfig.h"
#include "FANfaCreator_base.h"
#include "FAStr2Utf16.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Converts symbolic label into byte-representation or UTF-16/UTF-32 code.
///

class FANfaCreator_char : public FANfaCreator_base {

public:
    FANfaCreator_char (FAAllocatorA * pAlloc);

public:
    /// specifies input encoding name, e.g. CP1251, KOI8-R
    /// NULL stands for taking byte representation of symbols
    void SetEncodingName (const char * pEncStr);
    /// converts label into an input weight
    void SetTransition (
            const int FromState,
            const int ToState,
            const int LabelOffset,
            const int LabelLength
        );

private:
    /// returns true if symbol at Offset is escaped
    inline const bool IsEscaped (const int Offset) const;
    /// parses range of characters
    void ParseRange (const char * pStr, const int Length);
    /// returns a value of hex string
    inline const int GetHex (const char * pStr, const int Len) const;
    /// returns true C canbe a part of hex string
    inline const bool IsHex (const char C) const;
    /// returns Iw value and length of the next symbol
    const int GetIw (const char * pStr, const int StrLen, int * pSymLen);
    /// adds named range Iws
    void AddNamedRange (const int RangeName);

private:
    const char * m_pEncName;
    bool m_IsUtf8;
    FAStr2Utf16 m_recode;
    bool m_IsNegative;
    FAArray_cont_t < int > m_iws;

    enum { 
        DefMaxRange = 64000, 
    };

    enum {
        IW_ALNUM = -1,
        IW_DIGIT = -2,
        IW_PUNCT = -3,
        IW_ALPHA = -4,
        IW_SPACE = -5,
        IW_BLANK = -6,
        IW_LOWER = -7,
        IW_UPPER = -8,
        IW_CNTRL = -9,
        IW_XDIGIT = -10,
        IW_GRAPH = -11,
        IW_PRINT = -12,
    };

};

}

#endif
