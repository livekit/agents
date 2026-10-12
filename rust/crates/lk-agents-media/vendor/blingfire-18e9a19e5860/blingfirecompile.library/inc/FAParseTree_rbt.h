/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSETREE_RBT_H_
#define _FA_PARSETREE_RBT_H_

#include "FAConfig.h"
#include "FAParseTreeA.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FATaggedTextCA;
class FATaggedTextA;

///
/// This class takes shallow parser constituents and produces tagged words 
/// or MWEs. This allows to use shallow parser syntax for Rule-Base 
/// Tagging (RBT) task. This tree does not allow to have constituents other
/// than consiting of words only.
/// 

class FAParseTree_rbt : public FAParseTreeA {

public:
    FAParseTree_rbt (FAAllocatorA * pAlloc);
    virtual ~FAParseTree_rbt ();

public:
    void SetInText (const FATaggedTextCA * pIn);
    void SetOutText (FATaggedTextA * pOut);

// implementes a variant of a FAParseTreeA used by a parser
public:
    /// returns the array of indices of words: [0, 1, ..., N-1]
    const int GetUpperLabels (const int ** ppLabels) const;
    /// adjusts the m_From2To
    void Init (const int Count);
    /// makes words from From to To (index) be a single one with the tag Tag
    void AddNode (const int Tag, const int From, const int To);
    /// this will not allow parser to create more than one level of 
    /// constituents, this is not necessary if parser is set up for one pass
    void Update ();

// not implemented methods
private:
    const int GetUpperNodes (const int ** ppNodes) const;
    const int GetNext (const int Node) const;
    const int GetChild (const int Node) const;
    const int GetLabel (const int Node) const;

private:
    /// input tagged text
    const FATaggedTextCA * m_pIn;
    /// output tagged text
    FATaggedTextA * m_pOut;
    /// cannot be modified
    bool m_fCanUpdate;
    /// indicates that text has not been touched
    bool m_fChanged;
    /// maps From -> To, by default into itself
    FAArray_cont_t < int > m_From2To;
    /// maps From -> Tag, by default into m_pIn->GetTag (From)
    FAArray_cont_t < int > m_From2Tag;
};

}

#endif
