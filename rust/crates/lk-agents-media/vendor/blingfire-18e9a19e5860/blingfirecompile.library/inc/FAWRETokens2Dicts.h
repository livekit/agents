/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WRETOKENS2DICTS_H_
#define _FA_WRETOKENS2DICTS_H_

#include "FAConfig.h"
#include "FAChain2Num_hash.h"
#include "FAWRETokenParser.h"
#include "FAWREToken.h"
#include "FAArray_cont_t.h"

#include <iostream>

namespace BlingFire
{

class FAAllocatorA;
class FAChain2NumA;
class FAMultiMapA;
class FATagSet;

///
/// Reads WRE tokens and creates Digitizer\TokenNum -> CNF mapping and
/// a list of commands for text-digitizer
///
/// 1. Stream of commands for building text-digitizer
/// 2. FAChain2NumA Map from Digitizer\TokenNum into CNF of TypeNums
/// 3. TypeNum depend on digitizer type
///
/// Sample text-digitizer commands:
///
///   dict\n
///   <dictname>\n
///   \n                 ; reads dictionary from the outer resources
///   regexp\n
///   <regexp>\n
///   \n                 ; compiles dynamic dictionary from the regexp
///   wordlist\n
///   word_1
///   word_2
///   ...
///   word_n
///   \n                 ; compiles dynamic dictionary from the words
///   ...
///
/// Sample map from Digitizer\TokenNum into CNF of elementary types:
///
///   0x00000008 -> 3 7 -8 -3     ; Txt\Token8 = Ty7 | !Ty8 | !Ty3
///   0x00000006 -> 5 7 0 -8 0 -3 ; Txt\Token6 = Ty7 & !Ty8 & !Ty3
///   0x00000005 -> 1 6           ; Txt\Token5 = Ty6
///   0x01000005 -> 1 6           ; Tag\Token5 = Ty6
///   0x02000005 -> 1 6           ; Dct\Token5 = Ty6
///   ...
///

class FAWRETokens2Dicts {

public:
    FAWRETokens2Dicts (FAAllocatorA * pAlloc);
    virtual ~FAWRETokens2Dicts ();

public:
    /// sets up POS tagger's tagset (not used by default)
    void SetTagSet (const FATagSet * pTagSet);
    /// sets up bits lexicon's tagset (not used by default)
    void SetTagSet2 (const FATagSet * pTagSet2);
    /// sets up ordered wre tokens list
    void SetTokens (const FAChain2NumA * pToken2Num);
    /// Stream for commands for building text digitizer
    void SetDictOs (std::ostream * pDictOs);
    /// storage for token num 2 cnf map
    /// Token0 = (Ty1 | T2 | !T3) & (Ty4 | T5 | !T6) & T8 & !T9
    /// 0 -> 1 2 -3 0 4 5 -6 0 8 0 -9
    void SetToken2CNF (FAMultiMapA * pTokenNum2CNF);
    /// Makes processing
    void Process ();
    /// returns object into initial state (called automatically)
    virtual void Clear ();

private:
    /// makes consequent parsing for tokens from m_pToken2Num map
    void ParseTokens ();
    /// makes processing for newly reinitialized m_curr_token
    void ProcessCurrToken (const int TokenNum);
    void ProcessCurrWords (const int TokenNum);
    void ProcessCurrRegexps (const int TokenNum);
    void ProcessCurrDicts (const int TokenNum);
    void ProcessCurrTags (const int TokenNum);

    /// returns all the types to build text-digitzer
    void PutTxtTypes ();

    /// puts all words from m_curr_token into m_arr
    inline void words2arr ();
    /// puts regexp_i into the m_arr
    /// Note: regexp '.' and word "." will have the different representation
    inline void regexp2arr (const int i);
    /// puts dict_i into the m_arr
    inline void dict2arr (const int i);
    /// returns type num by words
    inline const int arr2typenum ();
    /// starts a new disjunction associated with the TokenNum
    inline void StartDisjunct (const int TokenNum);
    /// adds to the last disjunct associated with the TokenNum
    inline void AddToDisjunct (const int TokenNum, 
                               const int TypeNum, 
                               const bool IsNegative);

protected:

    // returns unescaped string
    const int UnEscape (
            const char * pStr, 
            const int StrLen, 
            const char ** ppOutStr
        );

    /// returns next wordlist for text-digitizer
    virtual void PutTxtWords (const char * pBegin, const int Length);
    /// returns next regexp for text-digitizer
    virtual void PutTxtRegexp (const char * pBegin, const int Length);
    /// returns next dict name for text-digitizer
    virtual void PutTxtDictName (const char * pBegin, const int Length);
    /// returns a set of tags for tag digitizer
    virtual void PutPosTags (const int * pTags, const int Count);
    /// returns a set of tags for dict digitizer
    virtual void PutDictTags (const int * pTags, const int Count);
    /// should be called when all tokens have been processed
    virtual void PutDone ();

private:
    /// input : WRE tokens
    const FAChain2NumA * m_pToken2Num;
    /// output stream : token -> type ops
    FAMultiMapA * m_pTokenNum2CNF;
    /// output stream : dictionaries, required by the rules
    std::ostream * m_pDictOs;

    /// WRE token parser
    FAWRETokenParser m_token_parser;
    /// currently processing WRE token
    FAWREToken m_curr_token;
    /// allocator 
    FAAllocatorA * m_pAlloc;
    /// tmp array
    FAArray_cont_t < int > m_arr;
    /// first unused value of TypeNum, 1-based
    int m_min_type;

    // keeps unescaped text, if needed
    FAArray_cont_t < char > m_UnEsc;

    // needed for text digitizer

    /// maps already seen words, dictionary names and regexps into type num
    FAChain2Num_hash m_token2type;
    /// key types in m_token2type
    enum {
        KEY_WORDLIST = 0,
        KEY_REGEXP = 1,
        KEY_DICT_NAME = 2
    };

    // needed for tag test or dict digitizers
    FAArray_cont_t < int > m_dict_tags;
    // keeps POS tagger tags
    FAArray_cont_t < int > m_pos_tags;

protected:
    /// tagsets
    const FATagSet * m_pTagSet;
    const FATagSet * m_pTagSet2;
};

}

#endif
