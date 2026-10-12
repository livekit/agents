/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STR2UTF16_H_
#define _FA_STR2UTF16_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

#ifdef HAVE_ICONV_LIB
#include <iconv.h>
#else
#include <windows.h>
#endif

#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;


///
/// Converts from specified 8-bit or UTF-8 encoding to UTF-16.
///
/// Note:
/// As input strings are byte encoded, they do not contain symbols outside
/// BMP and as a result UTF-16 output does not contain surrogate pairs so
/// the output can be safely used as UTF-32.
///

class FAStr2Utf16 {

public:
    FAStr2Utf16 (FAAllocatorA * pAlloc);
    ~FAStr2Utf16 ();

public:
    /// specifies input encoding name, e.g. CP1251, KOI8-R
    void SetEncodingName (const char * pEncStr);

    /// returns actual number of symbols in the output buffer
    /// return -1 if convertion is not possible
    const int Process (
        const char * pInStr,
        unsigned int InSize, 
        __out_ecount(OutSize) int * pOutStr,
        const unsigned int OutSize
    );

private:

    bool m_is_utf8;

#ifdef HAVE_ICONV_LIB
    iconv_t m_conv_type;
    FAArray_cont_t < unsigned short > m_tmp_buff;
#else
    unsigned int m_cp;
    FAArray_cont_t < wchar_t > m_tmp_buff;
#endif

    enum { DefTmpBuffSize = 1024 };

};

}

#endif
