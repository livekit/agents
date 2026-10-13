/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPLEXER_WRE_H_
#define _FA_REGEXPLEXER_WRE_H_

#include "FAConfig.h"
#include "FARegexpLexer_triv.h"

namespace BlingFire
{

///
/// The lexer assumes that the input is the combination of the following:
///
/// <token> ::= <word-list>[/<regexp-list>][/<dict-list>][/<pos-list>]
/// <token> ::= [<word-list>/]<regexp-list>[/<dict-list>][/<pos-list>]
/// <token> ::= [<word-list>/][<regexp-list>/]<dict-list>[/<pos-list>]
/// <token> ::= [<word-list>/][<regexp-list>/][<dict-list>/]<pos-list>
/// 
/// <word-list> ::= <word-or-list>
/// <word-list> ::= <word-not-list>
/// <word-or-list> ::= "text"
/// <word-or-list> ::= "text"|<word-or-list>
/// <word-not-list> ::= !"text"
/// <word-not-list> ::= !"text"<word-not-list>
/// 
/// <tag-list> ::= <tag-or-list>
/// <tag-list> ::= <tag-not-list>
/// <tag-or-list> ::= TAG
/// <tag-or-list> ::= TAG|<tag-or-list>
/// <tag-not-list> ::= !TAG
/// <tag-not-list> ::= !TAG<tag-not-list>
/// 
/// TAG ::= /[A-Za-z][A-Za-z0-9_]*/
/// 
/// <regexp-list> ::= <regexp-or-list>
/// <regexp-list> ::= <regexp-and-list>
/// <regexp-or-list> ::= [!]'regexp'
/// <regexp-or-list> ::= [!]'regexp'|<regexp-or-list>
/// <regexp-and-list> ::= [!]'regexp'
/// <regexp-and-list> ::= [!]'regexp'<regexp-and-list>
/// 
/// <dict-list> ::= <dict-or-list>
/// <dict-list> ::= <dict-and-list>
/// <dict-or-list> ::= @[!]<dictname>|<dict-or-list>
/// <dict-and-list> ::= @[!]<dictname><dict-and-list>
/// 
/// <dictname> ::= /[A-Za-z][A-Za-z0-9_]*/
/// 
/// Notes:
/// 1. No empty tokens allowed
/// 2. The spaces in tokens are only allowed withing quotations
///

class FARegexpLexer_wre : public FARegexpLexer_triv {

public:

  FARegexpLexer_wre ();

private:

  /// overriden 
  const int FindToken (const int Pos, int * pLength) const;
  /// returns true is any of ' ', '\t', '\n', '\r' is encountered at Pos
  inline const bool IsSpace (const int Pos) const;
  /// returns true if symbol at Pos can be a part of WRE token
  inline const bool IsToken (const int Pos) const;
  /// returns true if symbol at Pos can be a start of WRE token
  inline const bool IsTokenStart (const int Pos) const;

private:

  enum {
      InfoMaskSpace = 1,
      InfoMaskToken = 2,
      InfoMaskTokenStart = 4,
  };

  /// mapping from character into
  unsigned int m_char2info [AsciiSize];

};

}

#endif
