/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SUFFIXRULES2CHAINS_H_
#define _FA_SUFFIXRULES2CHAINS_H_

#include "FAConfig.h"
#include "FAMultiMap_ar.h"
#include "FAChain2Num_hash.h"
#include "FAArray_cont_t.h"
#include "FAStr2Utf16.h"

#include <iostream>

namespace BlingFire
{

class FATagSet;

///
/// This class converts suffix rules into a digital chains and extracts rules' 
/// right parts into a separate map. If tagset is specified then it converts 
/// tag names into corresponding tag values.
///
/// 1.1. Rules syntax (without tags)
///   ...
///   suffix_i --> act_i
///   suffix_{i+1} --> act_{i+1}
///   ...
///
/// 1.2. Rules syntax (with tag or two tags)
///   ...
///   suffix_i --> [FromTag][ToTag]act_i
///   suffix_{i+1} --> [FromTag][ToTag]act_{i+1}
///   ...
///   or
///   ...
///   suffix_i --> [Tag]act_i
///   suffix_{i+1} --> [Tag]act_{i+1}
///   ...
///
///   suffix_i ::= ^<non-empty string>
///   suffix_i ::= <non-empty string>
///
///   act_i ::= -N_i+str_i
///   act_i ::= -N_i
///   N_i ::= <number of letters to trip from the end>
///   str_i ::= <non-empty string>
///
///   The following are valid suffix rules:
///
///     ioned --> [VBD][VB]-1
///     ion --> [VB][VBD]-0+ed
///     cut --> [VB][VBD]-0
///     cut --> [VBD][VB]-0
///
/// 2. Action map is multimap of integers:
///   ...
///   i -> N c_1 ... c_M
///   ...
///
///   c_i, i \in 1..M are unicode (UTF-32) symbols
///

class FASuffixRules2Chains {

public:
    FASuffixRules2Chains (FAAllocatorA * pAlloc);

public:
    // input text encoding
    void SetEncodingName (const char * pInputEnc);
    // tag set, if there are tags in rules
    void SetTagSet (const FATagSet * pTagSet);
    // sets up base value for Keys identifying actions, 0 is used by default
    void SetKeyBase (const int KeyBase);
    // specifies the number of digitis per value to use in the output
    void SetNumSize (const int  NumSize);
    // if true then it will always insert a 0-delimiter
    // otherwise only if '^' is explicitly specified withing a rule
    void SetImplicitDelim (const bool ImplicitDelim);
    // input stream of rules
    void SetRulesIn (std::istream * pIs);
    // output stream of regular expression
    void SetChainsOut (std::ostream * pOs);
    // makes convertion
    void Process ();
    // returns stored action map
    const FAMultiMapA * GetActions () const;
    // returns stored Ow -> Freq array
    const int GetOw2Freq (const int ** pOw2Freq) const;

private:
    // returns object into initial state
    void Clear ();
    // finds " --> " delimiter
    inline const int GetDelimPos () const;
    // right part processing
    void ParseRight (const int RightPos, const int RightLen);
    // left part processing
    void ParseLeft (const int LeftPos, const int LeftLen);
    // helper method, builds an array of UTF-32 characters from the string
    inline void Str2Arr (
            const char * pStr, 
            const int StrLen, 
            FAArray_cont_t < int > * pArr
        );
    // helper, parses -N+str chunks
    inline void ParseAct (
            const char * pRight,
            const int RightLen,
            int * pCut,
            FAArray_cont_t < int > * pArr
        );
    // maps chain into a unique key
    inline const int GetKey (const int * pChain, const int Size);
    // make left and right chains ready for output
    void PrepareChains ();
    // print left part
    void PrintLeft () const;
    // builds multi-map of actions and Ow --> Freq array
    void BuildMap ();

private:
    // indicates whether input is in UTF-8 
    bool m_use_utf8;
    // key base
    int m_KeyBase;
    // number of digits to print
    int m_NumSize;
    // indicates whether delimiter should always be insterted
    bool m_ImplicitDelim;
    // input stream
    std::istream * m_pIs;
    // output stream
    std::ostream * m_pOs;
    // tagset pointer, can be NULL
    const FATagSet * m_pTagSet;
    // encodes actions' text
    FAStr2Utf16 m_recode;
    // actions in the above format
    FAMultiMap_ar m_acts;
    // keeps set of actions
    FAChain2Num_hash m_act2num;
    // input string
    const char * m_pStr;
    int m_StrLen;
    // first tag
    int m_TagFrom;
    // second tag
    int m_TagTo;
    // cut values
    int m_SuffixCut;
    int m_PrefixCut;
    // chains
    FAArray_cont_t < int > m_left_chain;
    FAArray_cont_t < int > m_suffix_chain;
    FAArray_cont_t < int > m_prefix_chain;
    FAArray_cont_t < int > m_right_chain;
    // Ow -> Freq array
    FAArray_cont_t < int > m_Ow2Freq;

    enum { DefNumSize = 5 };

};

}

#endif
