/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTMORPH_H_
#define _FA_TESTMORPH_H_

#include "FAConfig.h"
#include "FAPrmInterpreter_t.h"
#include "FAArray_cont_t.h"
#include "FASetUtils.h"

#include <iostream>

namespace BlingFire
{

class FAAllocatorA;

///
/// Base class for FAMorphLDB_t < int > tests
///

class FATestMorph {

public:
    FATestMorph (FAAllocatorA * pAlloc);
    virtual ~FATestMorph ();

public:
    /// sets up output stream
    void SetOutStream (std::ostream * pOs);
    /// returns output stream pointer
    std::ostream * GetOutStream () const;
    /// sets up FAPrmInterpreter_t < int > class to test
    void SetPRM (FAPrmInterpreter_t < int > * pPRM);
    /// sets up ignore case flag
    void SetIgnoreCase (const bool IgnoreCase);
    /// performs single for the pLineStr data
    virtual void Test (const char * pLineStr, const int LineLen) = 0;
    /// returns object into the initial state
    virtual void Clear ();
    /// prints out the test report
    virtual void PrintReport (std::ostream * pOs);

protected:

    static const int FindDelim (
            const int * pChain,
            const int ChainSize
        );

    static const int WordCount (
            const int * pWordList,
            const int Size
        );

    /// returns one of the error types
    const int UpdateCounts (
            const int * pA, 
            const int ASize,
            const int * pB, 
            const int BSize
        );

private:
    std::ostream * m_pOs;

protected:
    // PRM
    FAPrmInterpreter_t < int > * m_pPRM;
    bool m_IgnoreCase;

    // test count
    int m_Count;
    // passed test count
    int m_CorrectCount;
    // number of matched X (tags, segments, base-forms, hyph-points etc.)
    int m_MatchedEventCount;
    // total number generated X
    int m_EventCount;
    // total number of correct X in corpus
    int m_ExpectedEventCount;
    // number of times nothing was generated
    int m_NoEventCount;
    // maximum event count per 1 input
    int m_MaxEventCount;
    // expected maximum event count per 1 input
    int m_ExpectedMaxEventCount;
    // minimum event count per 1 input
    int m_MinEventCount;
    // expected minimum event count per 1 input
    int m_ExpectedMinEventCount;

    FAArray_cont_t < int > m_in_chain;
    FAArray_cont_t < int > m_out_chain;
    FAArray_cont_t < int > m_out2_chain;

    int * m_ChainBuffer;
    int * m_OutChainBuffer;
    int * m_OutChain2Buffer;

    // is used to compare input and generated sets of events,
    // e.g. A - B and B - A,
    FASetUtils m_set_utils;

    enum { 
        MaxChainSize = 409600,
    };

    enum {
        ERR_NO_ERROR = 0,
        ERR_EMPTY,
        ERR_BIGGER,
        ERR_SMALLER,
        ERR_DIFFERENT,
    };
};

}

#endif
