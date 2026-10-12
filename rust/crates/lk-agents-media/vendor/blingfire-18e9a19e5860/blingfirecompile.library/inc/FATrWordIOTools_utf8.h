/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_TRWORDIOTOOLS_UTF8_H_
#define _FA_TRWORDIOTOOLS_UTF8_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAMultiMapCA;

///
/// Input format:
///   a[XX]b[YY]...c[ZZ], where 
///     a,b,c - are non-empty input strings and 
///     [XX],[YY],[ZZ] - are optional output symbols
///
/// Only the following syntax is allowed in square brackets:
///
///   =D - normal hyphen
///   c=D - add letter before hyphen
///   Xc=D - change letter before hyphen
///   X=D - delete letter before hyphen
///   =DXc - change letter after hyphen
///   X=DXc - delete letter before and change letter after hyphen
///   ? - don't care (the annotation unknown)
///
///   D - non-negative integer value
///   c - a UTF-8 chracter (not 'X', '?', or '=')
///
/// For example:
///   correct:   ap[=0]pli[=0]ca[=0]tion
///   correct:   a[?]p[?]p[?]l[?]i[?]c[?]a[=]t[?]i[?]o[?]n
///   incorrect: ap[=0]pli[=0][=0]ca[=0]tion
///

class FATrWordIOTools_utf8 {

public:
    // makes word parsing
    static const int Str2IwOw (
            const char * pText,                  // word text
            const int Len,                       // word length
            __out_ecount(MaxOutSize) int * pIws, // Iws buffer, can be NULL
            __out_ecount(MaxOutSize) int * pOws, // Ows buffer, can be NULL
            const int MaxOutSize,                // buffer size
            const bool IgnoreCase = false,       // ignore-case flag
            const FAMultiMapCA * pCharMap = NULL // normalization charmap
        );

    // builds a parsable UTF-8 string from the Iws/Ows arrays
    static const int IwOw2Str (
            const int * pIws,
            const int * pOws,
            const int Size,
            __out_ecount (MaxOutSize) char * pOut,
            const int MaxOutSize
        );
};

}

#endif
