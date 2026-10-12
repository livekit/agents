/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_REGEXPLEXER_CHAR_H_
#define _FA_REGEXPLEXER_CHAR_H_

#include "FAConfig.h"
#include "FARegexpLexer_triv.h"

namespace BlingFire
{

///
/// Understands UTF-8 and any 8-bit encoded character regular expressions.
/// Uses 1-byte encoding by default.
///

class FARegexpLexer_char : public FARegexpLexer_triv {

public:
    FARegexpLexer_char ();

public:
    void SetUseUtf8 (const bool UseUtf8);

private:
    const int FindToken (const int Pos, int * pLength) const;
    /// returns true if any of '\\', '\t', '\n', '\r' is encountered
    inline const bool IsSpace (const char C) const;

private:
    bool m_is_space [AsciiSize];
    bool m_UseUtf8;
};

}

#endif
