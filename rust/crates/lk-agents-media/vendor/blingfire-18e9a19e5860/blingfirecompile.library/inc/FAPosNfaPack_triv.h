/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_POSNFAPACK_TRIV_H_
#define _FA_POSNFAPACK_TRIV_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAMap_judy.h"
#include "FAChainsPack_triv.h"
#include "FAOffsetTablePack.h"
#include "FAIwMapPack.h"

namespace BlingFire
{

class FARSNfaA;
class FAMultiMapA;

///
/// Position Nfa binary format:
/// BEGIN
/// Header:
///   <offset to the encoded destination sets>     : int
///   <offset to the encoded trbr arrays>          : int
///   <global size of TrBr offset>                 : int
///   <offset to the encoded state2offset table>   : int
///   <InitialCount>                               : int
///   <array of initial states>                    : int * InitialCount
///   <SizeOfIw2IwMap>                             : int
///   <Iw2IwMap>                                   : see FAIwMapPack
/// Body:
///   <state_0>                                    : variable size, see below
///   <state_1>                                    : variable size, see below
///   ...
///   <state_N>                                    : variable size, see below
/// Tables:
///   <encoded destination sets with size>1 >      : see FAChainsPack_triv
///   <encoded traingular bracket sets>            : see FAChainsPack_triv
///   <encoded state2offset table>                 : see FAOffsetTablePack
/// END
///
/// State representation format:
/// BEGIN
///   <info>                                       : char
///   [<TrCount>]                                  : <IwSize>
///   <sorted array of internal Iws>               : <IwSize> * TrCount
///   <parallel array of destination states>       : <DstSize> * TrCount
///   [<BeginingTrBrOffset>]                       : see field 3 of the header
///   [<EndingTrBrOffset>]                         : see field 3 of the header
/// END
///
/// <info> bits description:
/// BEGIN
///   0   - indicates that state begins some TrBrs, e.g. has <BeginingTrBrOffset>
///   1   - indicates that state ends some TrBrs
///   2-3 - <IwSize> - 1, bytes
///   4-5 - <DstSize> - 1, bytes
///   6-7 - <TrCount> + 1, 0 indicates that <TrCount> follows <info>
/// END
///
/// Note:
/// See *_pack_triv container classes for the interpretation of this dump.
///

class FAPosNfaPack_triv {

public:
    FAPosNfaPack_triv (FAAllocatorA * pAlloc);

public:
    // sets up automaton for processing
    void SetNfa (const FARSNfaA * pNfa);
    /// sets up brackets beginning map
    void SetPos2BrBegin (const FAMultiMapA * pPos2BrBegin);
    /// sets up brackets ending map
    void SetPos2BrEnd (const FAMultiMapA * pPos2BrEnd);
    // builds dump
    void Process ();
    // returns output dump representation of the automaton (size and pointer)
    const int GetDump (const unsigned char ** ppDump) const;

private:
    // makes converter ready
    void Prepare ();
    // calculates OldIw -> NewIw  map s.t. 
    // foreach i != j if Freq[OldIw_i] > Freq[OldIw_j] then NewIw_i < NewIw_j
    void CalcIwMap ();
    // stores automaton alphabet into m_alphabet
    void CalcAlphabet ();
    // calculates dump for all destination sets which contain more than one 
    // destination state
    void CalcDestDump ();
    // calculates dump for arrays from the traingular bracket maps
    void CalcTrBrDump ();

    // helper, returns destination state if there is only one in set
    // or returns masked dest set offset masked
    const int GetDest (const int State, const int Iw);
    // helper, builds array of iws and dsts for the given state
    void BuildIwsDsts (const int State);

    // calculates the number of bytes required to store automaton transitions
    const unsigned int GetSize ();
    // returns size necessary to represent the state
    const unsigned int GetStateSize (const int State);
    // returns size necessary to represent one element of the Iws array
    // (relys on data from the prev BuildIwsDsts call)
    const unsigned int CalcIwSize () const;
    // returns size necessary to represent one element of the Dsts array
    // (relys on data from the prev BuildIwsDsts call)
    const unsigned int CalcDstSize () const;

    // stores array of the initial states
    void StoreInitials ();
    // stores alphabet and Iw 2 Iw map if necessary
    void StoreIwMap ();
    // stores sets of destination states
    void StoreDestSets ();
    // stores arrays of values of TrBr maps
    void StoreTrBrs ();
    // State -> Offset mapping
    void StoreOffsetTable ();
    // builds dump for the given state
    void EncodeState (const int State);
    // encodes <info>-rmation byte
    inline void EncodeInfo (
            const int State, 
            const int IwSize, 
            const int DstSize
        );
    // list of transitions
    // (assumes m_iws, m_dsts contain outgoing transitions sorted by Iw)
    inline void EncodeTrs (const int IwSize, const int DstSize);
    // encodes offsets to the TrBr map entries, if any
    inline void EncodeTrBr (const int State);
    // encodes Iw with respect to the necessary size
    inline void EncodeIw (const int Iw, const int IwSize);
    // encodes Dst as 1,2,3 or 4 bytes
    inline void EncodeDst (const int Dst, const int DstSize);

    // returns number of bytes necessary to encode Dst
    inline static const int SizeOfValue (const int Value);

private:
    // input automaton
    const FARSNfaA * m_pNfa;
    // mapping from position into ids of triangular brackets it begins
    const FAMultiMapA * m_pPos2BrBegin;
    // mapping from position into ids of triangular brackets it ends
    const FAMultiMapA * m_pPos2BrEnd;
    // output buffer pointer
    unsigned char * m_pOutBuff;
    // mapping from state number into offset
    FAArray_cont_t < unsigned int > m_state2offset;
    unsigned int m_LastOffset;
    // temporary arrays
    FAArray_cont_t < int > m_alphabet;
    FAArray_cont_t < int > m_iws;
    FAArray_cont_t < int > m_dsts;
    FAArray_cont_t < int > m_tmp_arr;
    FAArray_cont_t < int > m_iw2iw;
    // dump storage
    FAArray_cont_t < unsigned char > m_dump;
    // destination sets packer
    FAChainsPack_triv m_dsts2dump;
    // offset table packer
    FAOffsetTablePack m_offsets2dump;
    // triangular brackets information packer
    FAChainsPack_triv m_trbrs2dump;
    // keeps iw2iw map dump
    FAIwMapPack m_iws2dump;
    // keeps global size of TrBrOffset
    int m_SizeOfTrBrOffset;
};

}

#endif
