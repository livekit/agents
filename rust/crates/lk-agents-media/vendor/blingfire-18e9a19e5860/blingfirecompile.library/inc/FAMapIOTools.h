/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_MAP_IO_TOOLS_H_
#define _FA_MAP_IO_TOOLS_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

#include <string>
#include <iostream>

namespace BlingFire
{

class FAAllocatorA;
class FAMapA;
class FAMultiMapA;
class FAChain2NumA;
class FAStr2IntA;

///
/// A set of methods for printing out, reading in different kinds of maps
/// in ascii text representation.
///

class FAMapIOTools {

public:
    FAMapIOTools (FAAllocatorA * pAlloc);

/// tools for FAMapA
public:
    /// reads FAMapA from text
    /// stops reading on empty line
    void Read (std::istream& is, FAMapA * pMap);
    /// prints out FAMapA in ascii format
    void Print (std::ostream& os, const FAMapA * pMap);

/// tools for FAMultiMapA
public:
    /// reads FAMultiMapA from text
    /// stops reading on empty line
    void Read (std::istream& is, FAMultiMapA * pMMap);
    /// prints out FAMultiMapA in ascii format
    void Print (std::ostream& os, const FAMultiMapA * pMMap);

/// tools for FAChain2NumA
public:
    /// reads FAChain2NumA from text
    /// stops reading on empty line
    void Read (std::istream& is, FAChain2NumA * pMap);
    /// prints out FAChain2NumA in ascii format
    void Print (std::ostream& os, const FAChain2NumA * pMap);

/// tools for FAStr2IntA
public:
    /// reads FAStr2IntA from text
    /// stops reading on empty line
    void Read (std::istream& is, FAStr2IntA * pMap);
    /// prints out FAStr2IntA in ascii format
    void Print (std::ostream& os, const FAStr2IntA * pMap);

/// Array tools
public:
    /// reads from textual representation, memory is valid upto the next call
    void Read (std::istream& is, const int ** ppArr, int * pCount);
    /// prints out textual representation
    void Print (std::ostream& os, const int * pArr, const int Count);
    /// reads from textual representation, memory is valid upto the next call
    void Read (std::istream& is, const float ** ppArr, int * pCount);
    /// prints out textual representation
    void Print (std::ostream& os, const float * pArr, const int Count);
    /// reads from textual representation, memory is valid upto the next call
    void Read (std::istream& is, FAArray_cont_t < unsigned char > * pBuff, FAArray_cont_t < int > * pOffsets, bool fStringFormat = true);
    /// prints out textual representation
    void Print (std::ostream& os, const FAArray_cont_t < unsigned char > * pBuff, const FAArray_cont_t < int > * pOffsets, bool fStringFormat = true);

private:
    /// allocator
    FAAllocatorA * m_pAlloc;
    /// keeps temporary array data
    FAArray_cont_t < int > m_tmp_arr;
    FAArray_cont_t < float > m_tmp_farr;
};

}

#endif
