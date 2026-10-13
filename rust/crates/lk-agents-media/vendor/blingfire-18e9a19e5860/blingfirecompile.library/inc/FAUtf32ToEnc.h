/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_UTF32TOENC_H_
#define _FA_UTF32TOENC_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Converts from the UTF-32LE into the specified encoding
///

class FAUtf32ToEnc {

public:
    FAUtf32ToEnc (FAAllocatorA * pAlloc);
    ~FAUtf32ToEnc ();

public:
    /// specifies output encoding or codepage name, e.g. 1251, CP1251, KOI8-R
    void SetEncodingName (const char * pEncStr);

    /// returns actual number of bytes (not symbols) in the output buffer
    /// returns -1 if convertion is not possible
    const int Process (
        const int * pInStr,
        unsigned int InSize,
        __out_ecount(MaxOutSize) char * pOutStr,
        const unsigned int MaxOutSize
    );

private:
    bool m_is_utf16;
    bool m_fBE;
    bool m_is_utf8;

    unsigned int m_cp;
    FAArray_cont_t < wchar_t > m_tmp_buff;

    enum { DefTmpBuffSize = 1024 };

};

}

#endif
