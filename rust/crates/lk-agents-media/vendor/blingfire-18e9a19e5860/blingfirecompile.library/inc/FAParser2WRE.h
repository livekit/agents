/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSER2WRE_H_
#define _FA_PARSER2WRE_H_

#include "FAConfig.h"
#include "FANfaCreator_base.h"
#include "FARegexp2Nfa.h"
#include "FAMultiMap_judy.h"
#include "FARegexpTree2Str.h"
#include "FATagSet.h"

#include <iostream>
#include <string>
#include <set>
#include <map>

namespace BlingFire
{

class FAAllocatorA ;
class FAAct2ArrA ;


///
/// This class converts parser rules into a list of [W]RE rules and calculates
/// extended tagset.
///
/// Syntax of parser rules:
///
///   Rule ::= Left --> Right
///   Right ::= Tag
///   Left ::= Lc < B > Rc
///   B ::= [W]RE
///   Lc ::= [W]REa|<empty>
///   Rc ::= [W]REa|<empty>
///   [W]REa ::= acyclic [W]RE
///
/// Note: 
///
/// 1. Left and Right parts are read as two separate streams.
/// 2. if Tag does not belong to the input tagset it will be added into output
///    tagset, resulting [W]REs should be compiled with that output tagset, but
///    it is not required at execution time.
/// 3. Number of output rules differes from the number of input rules as rules
///    with equal action and equal context length are grouped together.
///

class FAParser2WRE {

public:
    FAParser2WRE (FAAllocatorA * pAlloc);

public:
    // sets up the label type, FAFsmConst::LABEL_WRE is used by default
    void SetLabelType (const int LabelType);
    // indicates whether, character-based regular expression is in UTF-8
    // no need to call for other types of regular expression, false by default
    void SetUseUtf8 (const bool UseUtf8);
    // input stream of rules
    void SetLeftIn (std::istream * pLeftIs);
    // input stream of actions
    void SetRightIn (std::istream * pRightIs);
    // input tagset, can be NULL
    void SetInTagSet (const FATagSet * pInTagSet);
    // output stream of [W]REs
    void SetLeftOut (std::ostream * pLeftOs);
    // this number is substructed from the left context size, 0 by default
    void SetLcCut (const int LcCut);
    // makes processing
    void Process ();
    // returns output RuleNum --> Action map
    const FAMultiMapA * GetActMap () const;
    // returns output tagset
    const FATagSet * GetOutTagSet () const;
    // sets up custom action side parser
    void SetActionParser (FAAct2ArrA * pAct2Arr);

private:
    void Prepare ();
    // reads in right parts of the input rules
    void ProcessRight ();

    // returns tree's root id
    inline const int GetRootNode () const;
    // finds node which contain the main body of the rule
    inline const int GetTrBrNode () const;
    // adds size -> [Pos] entries
    inline void UpdateMap (
            FAMultiMapA * pMap, 
            const int Key, 
            const int * pVals, 
            const int Count
        );
    // checks for cycles in context 
    inline void Validate (
            const int Pos, 
            const int * pFollow, 
            const int Count
        ) const;
    // calculates context information
    void CalcContext_rec (
            const int * pFrom, 
            const int FromSize,
            const int * pTo, 
            const int ToSize,
            FAMultiMapA * pMap
        );
    // calculates left and right contexts
    void CalcContexts ();
    // reads and processes left parts of the input rules
    void ProcessLeft ();

    // builds m_B text
    inline void BuildBody ();
    // builds new context text
    inline void BuildContext (
            const int * pPos, 
            const int Count, 
            std::string * pContext
        );
    // adds output right -> left mapping
    inline void AddRule ();
    // adds one by one fixed context rules
    void AddRules ();

    // writes down output streams of left and right parts
    void WriteOutput ();
    // returns object into the initial state
    void Clear ();

private:
    // input streams
    std::istream * m_pLeftIs;
    std::istream * m_pRightIs;
    const FATagSet * m_pInTagSet;

    // output streams
    std::ostream * m_pLeftOs;
    FATagSet m_OutTagSet;
    int m_ExtTagBase;
    int m_ExtTagCount;
    FAMultiMap_judy m_out_i2a;

    // label type
    int m_LabelType;

    // current input rule text
    const char * m_pRuleStr;
    // current input rule text length
    int m_RuleStrLen;
    // current input rule number
    int m_RuleNum;

    // keeps 0-terminaetd buffers of right parts of input rules
    FAArray_cont_t < char > m_acts;
    // mapping from rule num into its right part
    FAArray_cont_t < int > m_i2a;

    // makes full parsing of each rule to calculate left/right context
    FANfaCreator_base m_tmp_nfa;
    FARegexp2Nfa m_regexp2nfa;
    const FARegexpTree2Funcs * m_pFuncs;

    // size -> [left context disjuction], -1 used as disjunct delimiter
    FAMultiMap_judy m_size2pos_lc;
    // size -> [right context disjuction], -1 used as disjunct delimiter
    FAMultiMap_judy m_size2pos_rc;
    // temporary array, keeps position next sequence
    FAArray_cont_t < int > m_tmp;
    FAArray_cont_t < int > m_tmp2;

    FARegexpTree2Str m_tree2str;
    std::set < std::string > m_disj_set;

    // left context size
    int m_lcs;
    // right context size
    int m_rcs;
    // left context text
    std::string m_Lc;
    // body text
    std::string m_B;
    // right context text
    std::string m_Rc;
    // optional function name
    std::string m_Fn;

    // mapping from the resulting right part into the resulting left part
    std::map < std::string, std::string > m_right2left;

    // it allows to have a "feature" prefix before extracting rules
    int m_LcCut;

    // a custom action side parser, NULL by default
    FAAct2ArrA * m_pAct2Arr;

    // constants
    enum {
        DefMaxContext = 100,
        DefMaxActSize = 100,
    };
};

}

#endif
