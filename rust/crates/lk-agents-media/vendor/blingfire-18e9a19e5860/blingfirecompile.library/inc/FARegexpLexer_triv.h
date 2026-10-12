/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPLEXER_TRIV_H_
#define _FA_REGEXPLEXER_TRIV_H_

#include "FAConfig.h"
#include "FARegexpLexerA.h"

namespace BlingFire
{

///
/// The lexer assumes that the input is the combination of the following:
///
/// 1. space or '\n' separated non-negative numbers: /[0-9]+/
/// 2. and following special symbols: /[.*+?|)(}{^$]\]\[/
///
/// The following expression is correct:
/// (10|23|234)? 234 234*
///

class FARegexpLexer_triv : public FARegexpLexerA {

public:
  FARegexpLexer_triv ();
  virtual ~FARegexpLexer_triv ();

public:
  void SetRegexp (const char * pRegexp, const int Length);
  void SetTokens (FAArray_cont_t < FAToken > * pTokens);
  void Process ();

protected:

  /// returns token's type and length at position Pos 
  /// or -1 if not identified
  virtual const int FindToken (const int Pos, int * pLength) const;
  /// returns true if symbol at position Pos is escaped
  const bool IsEscaped (const int Pos) const;
  /// returns true, if C is a special symbol
  const bool IsSpecial (const char C) const;
  /// returns true if C can be a part of hex number
  const bool IsHex (const char C) const;
  /// returns FARegexpTree::TYPE_ for the given symbols
  const int GetType (const char C) const;

protected:

  const char * m_pRegexp;
  int m_Length;
  FAArray_cont_t < FAToken > * m_pTokens;

  enum { AsciiSize = 128 };
  int m_char2type [AsciiSize];

private:

  /// adds token to the tokens container
  void PushToken (const int Type, const int Offset, const int Length);

};

}

#endif
