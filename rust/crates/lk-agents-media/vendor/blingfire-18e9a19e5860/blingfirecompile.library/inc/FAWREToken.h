/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WRE_TOKEN_H_
#define _FA_WRE_TOKEN_H_

#include "FAConfig.h"
#include "FAArray_t.h"
#include "FAToken.h"

namespace BlingFire
{

class FAAllocatorA;


class FAWREToken {

public:
    /// Token Types
    enum {
        TT_POSITIVE_WORD = 0,
        TT_NEGATIVE_WORD = 1,
        TT_POSITIVE_REGEXP = 2,
        TT_NEGATIVE_REGEXP = 3,
        TT_POSITIVE_DICT = 4,
        TT_NEGATIVE_DICT = 5,
        TT_POSITIVE_TAG = 6,
        TT_NEGATIVE_TAG = 7,
    };

public:
    /// an empty token constructor
    FAWREToken (FAAllocatorA * pAlloc);

public:
    /// sets up and returns the pointer to the unparsed token
    void SetStr (const char * pStr);
    const char * GetStr () const;

    /// returns the number of tokens in each group
    const int GetWordCount () const;
    const int GetRegexpCount () const;
    const int GetDictCount () const;
    const int GetTagCount () const;

    /// returns token by index, 0-based
    const FAToken * GetWordToken (const int Idx) const;
    const FAToken * GetRegexpToken (const int Idx) const;
    const FAToken * GetDictToken (const int Idx) const;
    const FAToken * GetTagToken (const int Idx) const;

    /// adds a new token
    void AddWordToken (const int Type, const int Offset, const int Length);
    void AddRegexpToken (const int Type, const int Offset, const int Length);
    void AddDictToken (const int Type, const int Offset, const int Length);
    void AddTagToken (const int Type, const int Offset, const int Length);

    /// returns true if the corresponding list is disjunctive
    const bool GetWordsDisj () const;
    const bool GetRegexpsDisj () const;
    const bool GetDictsDisj () const;
    const bool GetTagsDisj () const;

    /// returns true if the corresponding list is disjunctive
    void SetWordsDisj (const bool WordsDisj);
    void SetRegexpsDisj (const bool RegexpsDisj);
    void SetDictsDisj (const bool DictsDisj);
    void SetTagsDisj (const bool TagsDisj);

    /// returns into the initial state
    void Clear ();

private:
    /// a list of word tokens
    FAArray_t < FAToken > m_word_list;
    /// true if m_word_list is disjunctive
    bool m_disj_words;
    /// a list of regexp tokens
    FAArray_t < FAToken > m_regexp_list;
    /// true if m_regexp_list is disjunctive
    bool m_disj_regexps;
    /// a list of dictionary tokens
    FAArray_t < FAToken > m_dict_list;
    /// true if m_dict_list is disjunctive
    bool m_disj_dicts;
    /// a list of tag tokens
    FAArray_t < FAToken > m_tag_list;
    /// true if m_tag_list is disjunctive
    bool m_disj_tags;
    /// does not actualy store the string, just keeps the pointer
    const char * m_pStr;
};

}

#endif
