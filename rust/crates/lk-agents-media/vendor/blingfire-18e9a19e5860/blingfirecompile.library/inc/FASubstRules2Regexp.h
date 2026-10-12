/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SUBSTRULES2REGEXP_H_
#define _FA_SUBSTRULES2REGEXP_H_

#include "FAConfig.h"
#include "FAMultiMap_ar.h"
#include "FAStr2Utf16.h"
#include "FARegexpLexer_char.h"
#include "FAArray_cont_t.h"
#include "FAToken.h"

#include <iostream>
#include <string>
#include <map>

namespace BlingFire
{

class FATagSet;

///
/// This class converts substituter rules into a digital regular expression
/// and extracts rules' right parts into a separate map. If tagset is specified
/// then it converts tag names into corresponding tag values.
///
/// 1.1. Rules syntax (without tags)
///   ...
///   regexp_i --> act_i
///   regexp_{i+1} --> act_{i+1}
///   ...
///
/// 1.2. Rules syntax (with tag or two tags)
///   ...
///   regexp_i --> [FromTag][ToTag]act_i
///   regexp_{i+1} --> [FromTag][ToTag]act_{i+1}
///   ...
///   or
///   ...
///   regexp_i --> [Tag]act_i
///   regexp_{i+1} --> [Tag]act_{i+1}
///   ...
///
///   regexp_i ::= regular expression with triangular brackets
///
///   act_i ::= $N
///   act_i ::= Text
///   act_i ::= act_i$N
///   act_i ::= act_iText
///
/// 2. Action map is multimap of integers:
///   ...
///   i -> v_1 v_2 ... v_M
///   ...
///
///   if v_i <= 0 then v_i is a triangular bracket identifier
///   else it is a unicode symbol
///

class FASubstRules2Regexp {

public:

    FASubstRules2Regexp (FAAllocatorA * pAlloc);

public:
    // input text encoding
    void SetEncodingName (const char * pInputEnc);
    // tag set, if there are tags in rules
    void SetTagSet (const FATagSet * pTagSet);
    // specifies whether regular expressions will be compiled in reverse order
    void SetReverse (const bool Reverse);
    // input stream of rules
    void SetRulesIn (std::istream * pIs);
    // output stream of regular expression
    void SetRegexpOut (std::ostream * pOut);
    // makes convertion
    void Process ();
    // returns stored action map
    const FAMultiMapA * GetActions () const;

private:

    // returns into initial state
    void Clear ();
    // finds right part of the rule
    // returns NULL if there is no right part
    inline static const char * FindRight (const char * pStr, const int Len);
    // adds new entry to m_right2left or
    // updates existing one if there is already someting
    inline void AddEntry (const char * pLeft, const char * pRight);
    // builds m_right2left map
    void BuildRight2Left ();
    void ReadHeader ();
    void ReadFooter ();
    // prints left parts as a single regular expression
    void WriteLeftParts ();
    // fills in m_tokens
    void Tokenize (const char * pRegexp, const int RegexpLen);
    // modifies all '<N ' tokens with '<M ', where M = N | (Num << 16)
    void WriteRegexp (
            const char * pRegexp, 
            const int RegexpLen, 
            const int Num
        );
    // fills in m_acts map
    void BuildRightParts ();

private:

    std::istream * m_pIs;
    std::ostream * m_pOut;
    const FATagSet * m_pTagSet;
    bool m_Reverse;

    // glues together different left parts with the same right part
    std::map < std::string, std::string > m_right2left;
    // mapping from action number into action
    int m_MaxActNum;
    FAMultiMap_ar m_acts;
    // encodes actions' text
    FAStr2Utf16 m_recode;
    // indicates whether input is in UTF-8 
    bool m_use_utf8;
    // lexers
    FARegexpLexer_char m_lexer;
    // tokens
    FAArray_cont_t < FAToken > m_tokens;

};

}

#endif
