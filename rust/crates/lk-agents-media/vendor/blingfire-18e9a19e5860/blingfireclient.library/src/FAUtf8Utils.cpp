/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAUtf8Utils.h"

#ifndef SIZE_OPTIMIZATION
#include "FANormalizeDiacriticsMapPreserve.cxx"
#include "FANormalizeDiacriticsMapProd.cxx"
#include "FANormalizeDiacriticsMapRemove.cxx"
#endif

#define FAIsSurrogate(S) (0x0000D800 == (0xFFFFF800 & S))

namespace BlingFire
{

const int FAUtf8Size (const char * ptr)
{
    DebugLogAssert (ptr);

    int ch = *((const unsigned char *) ptr);

    if ((ch & 0x80) == 0x00) {
        return 1;
    } else if ((ch & 0xE0) == 0xC0) {
        return 2;
    } else if ((ch & 0xF0) == 0xE0) {
        return 3;
    } else if ((ch & 0xF8) == 0xF0) {
        return 4;
    } else {
        // not UTF-8 or out of Unicode 4.1.0 range: 0 - 10FFFF
        return 0;
    }
}

const int FAUtf8Size (const int Symbol)
{
    if (0x0000007Fu >= (unsigned int) Symbol) {
        return 1;
    } else if (0x000007FFu >= (unsigned int) Symbol) {
        return 2;
    } else if (0x0000FFFFu >= (unsigned int) Symbol) {
        return 3;
    } else if (0x0010FFFFu >= (unsigned int) Symbol) {
        return 4;
    } else {
        return 0;
    }
}


const char * FAUtf8ToInt (const char * ptr, int * result)
{
    int ret;
    int ch;
    int octet_count;
    int i;

    DebugLogAssert (ptr);
    DebugLogAssert (result);

    ch = (unsigned char) *ptr++;

    if ((ch & 0x80) == 0x00) {

        *result = ch;

    } else {

        if ((ch & 0xE0) == 0xC0){
            octet_count = 2;
            ch &= ~0xE0;
        }else if ((ch & 0xF0) == 0xE0){
            octet_count = 3;
            ch &= ~0xF0;
        }else if ((ch & 0xF8) == 0xF0){
            octet_count = 4;
            ch &= ~0xF8;
        }else{
            // not UTF-8 or out of Unicode 4.1.0 range: 0 - 10FFFF
            return NULL;
        }

        ret = ch;

        for (i = 1; i < octet_count; ++i) {

            ret <<= 6;
            ch = (unsigned char) *ptr++;
            if ((ch & 0xC0) != 0x80){
                // not UTF-8, broken sequence
                return NULL;
            }
            ret |= ch & 0x3f ;
        }

        if (octet_count != FAUtf8Size (ret)) {
            // input sequence must be the shortest one
            return NULL;
        }

        if(FAIsSurrogate(ret)) {
            // surrogate symbols cannot not appear in UTF-32/UTF-8
            return NULL;
        }

        *result = ret;
    }

    return ptr;
}


const char * FAUtf8ToInt (const char * pBegin, const char * pEnd, int * pResult)
{
    int ret;
    int ch;
    int octet_count;
    int i;

    DebugLogAssert (pBegin);
    DebugLogAssert (pResult);
    DebugLogAssert (pEnd);

    if (pEnd <= pBegin)
    {
        // empty buffers are not allowed
        return NULL;
    }

    // get the buffer size
    const size_t cbBuffSize = pEnd - pBegin;

    ch = (unsigned char) *pBegin++;

    if ((ch & 0x80) == 0x00) {

        *pResult = ch;

    } else {

        if ((ch & 0xE0) == 0xC0){
            octet_count = 2;
            ch &= ~0xE0;
        }else if ((ch & 0xF0) == 0xE0){
            octet_count = 3;
            ch &= ~0xF0;
        }else if ((ch & 0xF8) == 0xF0){
            octet_count = 4;
            ch &= ~0xF8;
        }else{
            // not UTF-8 or out of Unicode 4.1.0 range: 0 - 10FFFF
            return NULL;
        }

        // see if only a part of the symbol have been provided
        if (cbBuffSize < (unsigned) octet_count)
        {
            return NULL;
        }

        ret = ch;

        for (i = 1; i < octet_count; ++i) {

            ret <<= 6;
            ch = (unsigned char) *pBegin++;
            if ((ch & 0xC0) != 0x80){
                // not UTF-8, broken sequence
                return NULL;
            }
            ret |= ch & 0x3f ;
        }

        if (octet_count != FAUtf8Size (ret)) {
            // input sequence must be the shortest one
            return NULL;
        }

        if(FAIsSurrogate(ret)) {
            // surrogate symbols cannot not appear in UTF-32/UTF-8
            return NULL;
        }

        *pResult = ret;
    }

    return pBegin;
}


const int FAStrUtf8ToArray (
        const char * pStr,
        __out_ecount(MaxSize) int * pArray,
        const int MaxSize
    )
{
    DebugLogAssert (pStr);
    DebugLogAssert (pArray);

    // check for Byte-Order-Mark (Utf-8 encoded U+FEFF symbol)
    if (0xEFu == (unsigned char) pStr [0])
        if (0xBBu == (unsigned char) pStr [1])
            if (0xBFu == (unsigned char) pStr [2])
                pStr += 3;

    // process symbol sequence
    int i = 0;
    while (i < MaxSize && 0 != *pStr) {

        pStr = FAUtf8ToInt (pStr, pArray);

        if (NULL == pStr) {
            // invalid input sequence
            return -1;
        }

        pArray++;
        i++;
    }

    return i;
}


const int FAStrUtf8ToArray (
        const char * pStr,
        const int Len,
        __out_ecount(MaxSize) int * pArray,
        const int MaxSize
    )
{
    DebugLogAssert (0 == Len || pStr);
    DebugLogAssert (pArray);

    const char * pEnd = pStr + Len;
    const int * pArrayEnd = pArray + MaxSize;

    // check for Byte-Order-Mark (Utf-8 encoded U+FEFF symbol)
    if (3 <= Len) {
        if (0xEF == (unsigned char) pStr [0] && 
            0xBB == (unsigned char) pStr [1] && 
            0xBF == (unsigned char) pStr [2])
            pStr += 3;
    }

    // process symbol sequence
    int i = 0;
    while (pStr < pEnd && pArray < pArrayEnd) {

        pStr = FAUtf8ToInt (pStr, pEnd, pArray);

        if (NULL == pStr) {
            // invalid input sequence
            return -1;
        }

        pArray++;
        i++;
    }

    return i;
}


const int FAStrUtf8ToArray (
        const char * pStr,
        const int Len,
        __out_ecount(MaxSize) int * pArray,
        __out_ecount(MaxSize) int * pOffsets,
        const int MaxSize
    )
{
    DebugLogAssert (0 == Len || pStr);
    DebugLogAssert (pArray && pOffsets);

    const char * pBegin = pStr;
    const char * pEnd = pStr + Len;
    const int * pArrayEnd = pArray + MaxSize;

    // check for Byte-Order-Mark (Utf-8 encoded U+FEFF symbol)
    if (3 <= Len) {
        if (0xEF == (unsigned char) pStr [0] && 
            0xBB == (unsigned char) pStr [1] && 
            0xBF == (unsigned char) pStr [2])
            pStr += 3;
    }

    // process symbol sequence
    int i = 0;
    while (pStr < pEnd && pArray < pArrayEnd) {

        const int Offset = (int) (pStr - pBegin);
        pStr = FAUtf8ToInt (pStr, pEnd, pArray);

        if (NULL == pStr) {
            // invalid input sequence
            return -1;
        }

        pArray++;
        pOffsets [i++] = Offset;
    }

    return i;
}


const int FAStrUtf8AsBytesToArray (
        const char * pStr, 
        const int Len, 
        __out_ecount(MaxSize) int * pArray, 
        const int MaxSize
    )
{
    DebugLogAssert (0 == Len || pStr);
    DebugLogAssert (pArray);

    const char * pEnd = pStr + Len;
    const int * pArrayEnd = pArray + MaxSize;

    // check for Byte-Order-Mark (Utf-8 encoded U+FEFF symbol)
    if (3 <= Len) {
        if (0xEF == (unsigned char) pStr [0] && 
            0xBB == (unsigned char) pStr [1] && 
            0xBF == (unsigned char) pStr [2])
            pStr += 3;
    }

    // process symbol sequence
    int i = 0;
    while (pStr < pEnd && pArray < pArrayEnd) {
        *pArray++ = (unsigned char) *pStr++;
        i++;
    }

    return i;
}


const int FAStrUtf8AsBytesToArray (
        const char * pStr,
        const int Len,
        __out_ecount(MaxSize) int * pArray,
        __out_ecount(MaxSize) int * pOffsets,
        const int MaxSize
    )
{
    DebugLogAssert (0 == Len || pStr);
    DebugLogAssert (pArray && pOffsets);

    const char * pBegin = pStr;
    const char * pEnd = pStr + Len;
    const int * pArrayEnd = pArray + MaxSize;

    // check for Byte-Order-Mark (Utf-8 encoded U+FEFF symbol)
    if (3 <= Len) {
        if (0xEF == (unsigned char) pStr [0] && 
            0xBB == (unsigned char) pStr [1] && 
            0xBF == (unsigned char) pStr [2])
            pStr += 3;
    }

    // process symbol sequence
    int i = 0;
    while (pStr < pEnd && pArray < pArrayEnd) {

        const int Offset = (int) (pStr - pBegin);
        *pArray++ = (unsigned char) *pStr++;
        pOffsets [i++] = Offset;
    }

    return i;
}


wchar_t * FAIntToUtf16LE (
        int Symbol,
        __out_ecount(MaxSize) wchar_t * ptr,
        const int MaxSize
    )
{
    DebugLogAssert (ptr);

    if (0xFFFF >= (unsigned int) Symbol && 0 < MaxSize) {
        *ptr = (wchar_t) Symbol;
        ptr++;

        return ptr;

    } else if (1 < MaxSize) {

        Symbol -= 0x10000;

        // initialize first word with lower bracket
        *ptr = (wchar_t) (0xD800 | (Symbol >> 10));
        ptr++;

        // initialize second word with higher bracket
        *ptr = (wchar_t) (0xDC00 | (Symbol & 0x3FF));
        ptr++;

        return ptr;
    }

    return NULL;
}

const int FAStrUtf8ToUtf16LE (
        const char * pStr,
        const int Len,
        __out_ecount(MaxSize) wchar_t * pArray,
        __out_ecount(MaxSize) int * pOffsets,
        const int MaxSize
    )
{
    DebugLogAssert (0 == Len || pStr);
    DebugLogAssert (pArray && pOffsets);

    const char * pBegin = pStr;
    const char * pEnd = pStr + Len;
    const wchar_t * pArrayEnd = pArray + MaxSize;

    // check for Byte-Order-Mark (Utf-8 encoded U+FEFF symbol)
    if (3 <= Len) {
        if (0xEF == (unsigned char) pStr [0] && 
            0xBB == (unsigned char) pStr [1] && 
            0xBF == (unsigned char) pStr [2])
            pStr += 3;
    }

    // process symbol sequence
    int codepoint = 0;
    int i = 0;
    while (pStr < pEnd && pArray < pArrayEnd) {

        const int Offset = (int) (pStr - pBegin);
        pStr = FAUtf8ToInt (pStr, pEnd, &codepoint);

        if (NULL == pStr) {
            // invalid input sequence
            return -1;
        }

        wchar_t * out = FAIntToUtf16LE(codepoint, pArray, MaxSize - i);

        if (NULL == out) {
            //not enough memory
            break;
        }

        DebugLogAssert (out != pArray);

        for (std::ptrdiff_t j = 0; i < MaxSize && j < out - pArray; j++) {
            pOffsets[i++] = Offset;
        }

        pArray = out;
    }

    return i;
}

char * FAIntToUtf8 (
        const int Symbol, 
        __out_ecount(MaxSize) char * ptr, 
        const int MaxSize
    )
{
    DebugLogAssert (ptr);
    DebugLogAssert (0 <= Symbol && 0x10FFFF >= Symbol);

    if (0x0000007Fu >= (unsigned int) Symbol && 0 < MaxSize) {

        *ptr = (unsigned char) Symbol;
        ptr++;

        return ptr;

    } else if (0x000007FFu >= (unsigned int) Symbol && 1 < MaxSize) {

        *ptr = 0xC0u | (unsigned char) (Symbol >> 6);
        ptr++;
        *ptr = 0x80u | (unsigned char) (Symbol & 0x3Fu);
        ptr++;

        return ptr;

    } else if (0x0000FFFFu >= (unsigned int) Symbol && 2 < MaxSize) {

        if(FAIsSurrogate(Symbol)) {
            // surrogate symbols cannot not appear in UTF-32
            return NULL;
        }

        *ptr = 0xE0u | (unsigned char) (Symbol >> 12);
        ptr++;
        *ptr = 0x80u | (unsigned char) ((Symbol >> 6) & 0x3Fu);
        ptr++;
        *ptr = 0x80u | (unsigned char) (Symbol & 0x3Fu);
        ptr++;

        return ptr;

    } else if (0x0010FFFFu >= (unsigned int) Symbol && 3 < MaxSize) {

        *ptr = 0xF0u | (unsigned char) (Symbol >> 18);
        ptr++;
        *ptr = 0x80u | (unsigned char) ((Symbol >> 12) & 0x3Fu);
        ptr++;
        *ptr = 0x80u | (unsigned char) ((Symbol >> 6) & 0x3Fu);
        ptr++;
        *ptr = 0x80u | (unsigned char) (Symbol & 0x3Fu);
        ptr++;

        return ptr;
    }

    return NULL;
}


const int FAArrayToStrUtf8 (
        const int * pArray, 
        const int Size, 
        __out_ecount(MaxStrSize) char * pStr, 
        const int MaxStrSize
    )
{
    DebugLogAssert (pStr);
    DebugLogAssert (pArray && 0 < Size);

    char * ptr = pStr;

    for (int i = 0; i < Size; ++i) {

        const int Symbol = pArray [i];

        const int CurrSize = (const int) (ptr - pStr);
        ptr = FAIntToUtf8 (Symbol, ptr, MaxStrSize - CurrSize);

        if (NULL == ptr) {
            // invalid input sequence
            return -1;
        }
    }

    const int CurrSize = (const int) (ptr - pStr);
    return CurrSize;
}


const bool FAIsUtf8Enc (const char * pEncName)
{
    bool IsUtf8 = false;

    if (pEncName) {

        if (0 == strncmp ("UTF8", pEncName, 4) || 
            0 == strncmp ("UTF-8", pEncName, 5) ||
            0 == strncmp ("65001", pEncName, 5)) {
            IsUtf8 = true;
        }
    }

    return IsUtf8;
}


#ifndef SIZE_OPTIMIZATION
typedef const uint16_t (*_TPCharNormalizationMapDiacritics)[65536][2];

const int FAStrUtf8Normalize (
        const char * pStr,
        const int Len,
        __out_ecount(MaxOutSize) char * pOutStr,
        const int MaxOutSize,
        const int NormAlgo = FAFsmConst::NORMALIZE_DEFAULT
    )
{
    DebugLogAssert (0 == Len || pStr);
    DebugLogAssert (0 == MaxOutSize || pOutStr);

    // get the corresponding mapping array
    _TPCharNormalizationMapDiacritics pMap = &g_CharNormalizationMapDiacriticsProd;

    switch(NormAlgo)
    {
      case FAFsmConst::NORMALIZE_PRESERVE_DIACRITICS:
        pMap = &g_CharNormalizationMapDiacriticsPreserve;
        break;

      case FAFsmConst::NORMALIZE_REMOVE_DIACRITICS:
        pMap = &g_CharNormalizationMapDiacriticsRemove;
        break;
    };

    // Compute the end of the input buffer, the current output position 
    //  and the end of the output buffer.
    const char * const pEnd = pStr + Len;
    char * pOutStrCurr = pOutStr;
    char * const pOutStrEnd = pOutStr + MaxOutSize;

    // check for Byte-Order-Mark (Utf-8 encoded U+FEFF symbol) and ignore it
    if (3 <= Len) {
        if (0xEF == (unsigned char) pStr [0] && 
            0xBB == (unsigned char) pStr [1] && 
            0xBF == (unsigned char) pStr [2])
            pStr += 3;
    }

    // process the input symbol sequence
    while (pStr < pEnd) {

        // get the UTF-32LE value and get the pointer to the next UTF-8 character
        int utf32le = 0;
        const char * pNextStr = FAUtf8ToInt (pStr, pEnd, &utf32le);

        if (NULL == pNextStr) {
            // invalid input sequence
            return -1;
        }

        DebugLogAssert (pNextStr > pStr); // something weird, pointer should always move forward

        // get the mapping
        uint16_t c1 = 1;
        uint16_t c2 = 1;

        if(utf32le < 0xffff) {
            c1 = (*pMap) [utf32le][0];
            c2 = (*pMap) [utf32le][1];
        }

        // covert mapping characters to UTF8, -1 mean there is no entry in the map
        if(1 != c1) {

            if(0 < c1) {
                char * const pOut = FAIntToUtf8 (c1, pOutStrCurr, int(pOutStrEnd - pOutStrCurr));
                if(NULL == pOut) {
                    // non UTF-8 data in the map
                    return -1;
                }
                pOutStrCurr += FAUtf8Size (c1);
            }
            if(0 < c2) {
                char * const pOut = FAIntToUtf8 (c2, pOutStrCurr, int(pOutStrEnd - pOutStrCurr));
                if(NULL == pOut) {
                    // non UTF-8 data in the map
                    return -1;
                }
                pOutStrCurr += FAUtf8Size (c2);
            }

        // copy the character over as-is
        } else {

            if(pOutStrEnd - pOutStrCurr >= pNextStr - pStr) {
                memcpy(pOutStrCurr, pStr, pNextStr - pStr);
            }
            pOutStrCurr += (pNextStr - pStr);
        }

        // skip to the next charater start in the input
        pStr = pNextStr;
    }

    // Return the required size to store the output string.
    // Note: if MaxOutSize is equal or bigger than this value, 
    //  then the output string is stored in the pOutStr buffer.
    return int(pOutStrCurr - pOutStr);
}
#endif

}
