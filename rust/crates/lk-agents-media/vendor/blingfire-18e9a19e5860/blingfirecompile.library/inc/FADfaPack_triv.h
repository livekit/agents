/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DFAPACK_TRIV_H_
#define _FA_DFAPACK_TRIV_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAArray_t.h"
#include "FAMap_judy.h"
#include "FABitArray.h"
#include "FAChainsPack_triv.h"
#include "FAIwMapPack.h"
#include "FACalcIwEqClasses.h"

namespace BlingFire
{

class FARSDfaA;
class FAState2OwA;
class FAState2OwsA;
class FAMealyDfaA;

///
/// FADfaPack_triv automaton binary format:
///
/// BEGIN
/// Header:
///   <DstSize>                                  : int; valid values are 1..4
///   <offset of the encoded Ows sets>           : int; 0 if does not exist
///   if (RemapIws) {
///     0x80000000 | <AlphabetSize>              : int
///   } else {
///     <AlphabetSize>                           : int
///   }
///   <Alphabet>                                 : int * AlphabetSize
///   if (RemapIws) {
///     [<SizeOfIw2IwMap>]                       : int
///     [<Iw2IwMap>]                             : SizeOfIw2IwMap
///   }                                          : see FAIwMapPack
/// Body:
///   <state_0>                                  : variable size, see below
///   <state_1>                                  : variable size, see below
///   ...
///   <state_N>                                  : variable size, see below
/// Ows:
///   <encoded Ows sets>                         : see FAChainsPack_triv
/// END
///
/// State representation format:
/// BEGIN
///   <info>                                     : char
///   if (TRS_IMPL) {
///      <iw>                                    : <IwSize>
///    } else if (TRS_PARA) {
///      <TrCount>                               : <IwSize>
///      <sorted array of internal Iws>          : <IwSize> * TrCount
///      <parallel array of destination states>  : <DstSize> * TrCount
///    } else if (TRS_IWIA) {
///      <IwBase>                                : <IwSize>
///      <IwMax>                                 : <IwSize>
///      <array of destination states>           : <DstSize> * TrCount
///    } else if (TRS_RANGE) {
///      <Count>                                 : <IwSize>
///      <FromIws>                               : <IwSize> * Count
///      <ToIws>                                 : <IwSize> * Count
///      <parallel array of destination states>  : <DstSize> * Count
///    }
///    [<Ow> | <Ows set offset>]                 : <OwSize>
/// END
///
/// <info> bits description:
/// BEGIN
///   0-2 - 000 - only if (0 == TrCount)
///         010 - only if (1 == TrCount && Dst == Src + 1)
///         100 - Iws/Dsts are stored as two parallel arrays, Iws is sorted
///         110 - Iws/Dsts are stored as Iws indexed array of Dsts (IwIA)
///         001 - ranges of Iws, each of which corresponds to only one Dst
///   3-4 - <IwSize> - 1, bytes
///   5-6 - 0 - no Ow or Ows assosiated with this state
///       - 1 - OwsSize == 1
///       - 2 - OwsSize == 2
///       - 3 - OwsSize == 4
///   7   - indicates whether state is final
/// END
///
/// Note:
/// See FARSDfa_pack_triv, FAState2Ow_pack_triv, FAState2Ows_pack_triv,
/// FAMealyDfa_pack_triv and FAOw2Iw_pack_triv for interpretation of this dump.
///

class FADfaPack_triv {

public:
    FADfaPack_triv (FAAllocatorA * pAlloc);

public:
    /// sets up automaton for processing
    void SetDfa (const FARSDfaA * pDfa);
    /// sets up Moore automaton reaction
    void SetState2Ow (const FAState2OwA * pState2Ow);
    /// sets up Multi Moore automaton reaction
    void SetState2Ows (const FAState2OwsA * pState2Ows);
    /// sets up Mealy automaton reaction
    void SetSigma (const FAMealyDfaA * pSigma);
    /// sets up whether to remap alphabet or not, false by default
    void SetRemapIws (const bool RemapIws);
    /// set use Iw-indexed arrays whenever possible, false by default
    void SetUseIwIA (const bool UseIwIA);
    /// uses ranges whenever possible, false by default
    void SetUseRanges (const bool UseRanges);
    /// sets up Dst (DstOffset) size, 3 is used by default
    void SetDstSize (const int DstSize);
    /// builds dump
    void Process ();
    /// returns output dump representation of the automaton (size and pointer)
    const int GetDump (const unsigned char ** ppDump) const;

private:
    // makes converter ready
    void Prepare ();

    // returns destination state with respect to the Iws equivalence classes
    inline const int GetDest_eq (const int State, const int Iw) const;
    // returns alphabet with respect to the equivalence classes
    inline const int GetIWs_eq (const int ** ppIws) const;

    // calcs Iw equivalence classes, if needed
    void BuildEqs ();
    // calculates OldIw -> NewIw  map s.t. 
    // foreach i != j if Freq[OldIw_i] > Freq[OldIw_j] then NewIw_i < NewIw_j
    void BuildIwMap ();
    // builds memory dump for Iw2Iw map
    void BuildIw2IwDump ();
    // packs sets of Ows if m_pState2Ows was specified
    void PackOws ();
    // stores alphabet and Iw 2 Iw map if necessary
    void StoreIws ();
    // stores all the states
    void StoreStates ();
    // stores Ows sets, only if FAState2OwsA is specified
    void StoreOws ();

    // returns the transitions representation type (see above)
    // (relys on data from the prev BuildIwsDsts call)
    // returns FAFsmConst::TRS_PARA, FAFsmConst::TRS_IWIA
    const int GetTrType (const int State, const int IwSize);

    // calculates the number of bytes required to store automaton
    const unsigned int GetSize ();
    // returns size necessary to represent the state
    const unsigned int GetStateSize (const int State);
    // returns size necessary to represent one element of the Iws array
    // (relys on data from the prev BuildIwsDsts call)
    const unsigned int GetIwSize () const;
    // returns size of State if the transition can be represented implicitly
    // (relys on data from the prev BuildIwsDsts call)
    const unsigned int GetTrsSize_impl (const int IwSize) const;
    // returns size of State if it is two parallel arrays of Iws and Dsts
    // (relys on data from the prev BuildIwsDsts call)
    const unsigned int GetTrsSize_para (const int IwSize) const;
    // returns size of State if it is Iw-Indexed Array of Dst
    // (relys on data from the prev BuildIwsDsts call)
    const unsigned int GetTrsSize_iwia (const int IwSize) const;
    // returns size of State if it is ranges of Iws 
    // (relys on data from the prev BuildIwsDsts call)
    const unsigned int GetTrsSize_range (const int IwSize);

    // helper, builds array of iws and dsts for the given state
    inline void BuildIwsDsts (const int State);

    // builds dump for the given state
    void EncodeState (const int State);
    // encodes <info>-rmation byte
    inline void EncodeInfo (const int State, const int IwSize);
    // one implicit transition to the next state
    inline void EncodeTrs_impl (const int IwSize);
    // transitions as two parallel arrays
    // (assumes m_iws, m_dsts contain outgoing transitions sorted by Iw)
    inline void EncodeTrs_para (const int IwSize);
    // Iws-indexed array of Dsts
    // (assumes m_iws, m_dsts contain outgoing transitions sorted by Iw)
    inline void EncodeTrs_iwia (const int IwSize);
    // ranges of Iws
    // (assumes m_iws, m_dsts contain outgoing transitions sorted by Iw)
    inline void EncodeTrs_range (const int IwSize);
    // encodes Iw with respect to the necessary size
    inline void EncodeIw (const int Iw, const int IwSize);
    // returns encoded Iw with respect to the IwSize
    inline const int DecodeIw (const int Offset, const int IwSize) const;
    // encodes Dst as three bytes
    inline void EncodeDst (const int Dst);
    // returns encoded Dst
    inline const int DecodeDst (const int Offset) const;
    // encodes Ow
    inline void EncodeOw (const int State);
    // returns number of bytes necessary to encode Ow of the given State
    inline const unsigned int CalcOwSize (const int State);

    // changes state nums into corresponding memory offests
    void State2Offset (const int State);
    void State2Offset_para (int Offset, const int State, const int IwSize);
    void State2Offset_iwia (int Offset, const int State, const int IwSize);
    void State2Offset_range (int Offset, const int State, const int IwSize);

    // returns array of Mealy automaton reactions at a given state
    inline const int GetMealyOws (const int State, const int ** ppOws);

    // packs automaton's alpahbet
    void PackAlphabet ();

private:
    // input automaton
    const FARSDfaA * m_pDfa;
    // state -> Ow map if any
    const FAState2OwA * m_pState2Ow;
    // state -> { Ow } map if any
    const FAState2OwsA * m_pState2Ows;
    // mealy reaction
    const FAMealyDfaA * m_pSigma;
    // indicates whether to use alphabet remapping or not
    bool m_RemapIws;
    // indicates whether it is allowed to use IwIAs
    bool m_UseIwIA;
    // indicates whether Iw ranges are allowed to use
    bool m_UseRanges;
    // output buffer pointer
    unsigned char * m_pOutBuff;
    // mapping from state number into offset
    FAArray_t < unsigned int > m_state2offset;
    unsigned int m_LastOffset;
    // m_pDfa's final states
    FABitArray m_finals;
    // temporary arrays
    FAArray_cont_t < int > m_iws;
    FAArray_cont_t < int > m_dsts;
    FAArray_cont_t < int > m_tmp_arr;
    FAArray_cont_t < int > m_iw2iw;
    // dump storage
    FAArray_cont_t < unsigned char > m_dump;
    // multi-map packer
    FAChainsPack_triv m_ows2dump;
    // iw map packer
    FAIwMapPack m_iws2dump;
    // calcs Iw equivalence classes
    FACalcIwEqClasses m_iws2eqs;
    // Eq -> Iw mapping
    FAArray_cont_t < int > m_eq2iw;
    // Iw -> Eq mapping
    FAMap_judy m_iw2eq;
    // Eqs alphabet
    FAArray_cont_t < int > m_eqs;
    // temporary storage for Mealy Ows
    FAArray_cont_t < int > m_mealy_ows;
    // keeps packed alphabet
    FAArray_cont_t < int > m_alphabet;
    // DstSize, e.g. 1, 2, 3 or 4
    int m_DstSize;
    // e.g. 0x00ffffff, for m_DstSize == 3
    unsigned int m_DstMask;
};

}

#endif
