/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_LEXTOOLS_T_H_
#define _FA_LEXTOOLS_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAUtf32Utils.h"
#include "FARSDfaCA.h"
#include "FAState2OwCA.h"
#include "FAMultiMapCA.h"
#include "FAWbdConfKeeper.h"
#include "FALimits.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Lexical analyzer runtime. 
///
/// This runtime interprets the rules of the following format:
///
/// re* < re > re* --> Tag
/// re* < re > re* --> _call FnTag
/// re* < re > re* --> Tag _call FnTag
/// re* < re > re* --> _call FnTag1 FnTag2 ...
/// re* < re > re* --> Tag _call FnTag1 FnTag2 ...
///
/// _function FnTag
///  <same kind of rules>
/// _end
///
/// see doc\lex.html, doc\lex.ppt for description
///
/// Usage notes:
///
/// 1. The object supposed to be initialized before it's used, it is caller 
///  responsibility to guarantee this.
///
/// 2. It is possible to execute either all rules or a particular function
///  in the later case it is caller responsibility to know function tag values,
///  the values can be predefined (and frozen) in the tagset.txt file of the
///  the corresponding grammar.
///

template < class Ty >
class FALexTools_t {

public:
    FALexTools_t ();

public:
    /// sets up the data containers
    void SetConf (const FAWbdConfKeeper * pWbdConf);

    /// makes a processing
    const int Process (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        ) const;

    /// makes a processing, starting from a particular _function
    const int Process (
            const int FnTag,
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        ) const;

private:
    /// validates consitensy between data structures
    inline void Validate () const;

    // internal processing function, returns the size of the output array
    const int Process_int (
            const int Initial,
            const int Offset,
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize,
            const int RecDepth,
            const bool fOnce = false
        ) const;

private:
    /// input objects
    const FARSDfaCA * m_pDfa;
    const FAState2OwCA * m_pState2Ow;
    const FAMultiMapCA * m_pActs;
    bool m_IgnoreCase;
    int m_MaxDepth;
    /// maps function id into an initial state, or -1 if not valid
    const int * m_pFn2Ini;
    unsigned int m_Fn2IniSize;
    /// maximum token length
    int m_MaxTokenLength;
    /// constants
    enum {
        DefMaxDepth = 2,
        MinActSize = 3,
        DefSubIw = FAFsmConst::IW_EPSILON,
    };
};


template < class Ty >
FALexTools_t< Ty >::FALexTools_t () :
    m_pDfa (NULL),
    m_pState2Ow (NULL),
    m_pActs (NULL),
    m_IgnoreCase (false),
    m_MaxDepth (DefMaxDepth),
    m_pFn2Ini (NULL),
    m_Fn2IniSize (0),
    m_MaxTokenLength (FALimits::MaxWordLen)
{}


template < class Ty >
void FALexTools_t< Ty >::SetConf (const FAWbdConfKeeper * pWbdConf)
{
    if (pWbdConf) {

        m_pDfa = pWbdConf->GetRsDfa();
        m_pState2Ow = pWbdConf->GetState2Ow();
        m_IgnoreCase =  pWbdConf->GetIgnoreCase();
        m_MaxDepth = pWbdConf->GetMaxDepth ();
        m_pActs = pWbdConf->GetMMap ();
        m_Fn2IniSize = pWbdConf->GetFnIniStates (&m_pFn2Ini);
        m_MaxTokenLength = pWbdConf->GetMaxTokenLength ();

    } else {

        m_pDfa = NULL;
        m_pState2Ow = NULL;
        m_IgnoreCase = false;
        m_MaxDepth = DefMaxDepth;
        m_pActs = NULL;
        m_Fn2IniSize = 0;
        m_pFn2Ini = NULL;
        m_MaxTokenLength = FALimits::MaxWordLen;
    }

    Validate ();
}


template < class Ty >
inline void FALexTools_t< Ty >::Validate () const
{
    if (m_pActs && m_pDfa) {

        const int * pAct;
        int ActSize;
        int ActId = 0;

        while (-1 != (ActSize = m_pActs->Get (ActId++, &pAct))) {

            // invalid action
            LogAssert (pAct && MinActSize <= ActSize);

            const int LeftCx = pAct [0];
            const int RightCx = pAct [1];
            LogAssert (-FALimits::MaxTag <= LeftCx && LeftCx <= FALimits::MaxTag);
            LogAssert (-FALimits::MaxTag <= RightCx && RightCx <= FALimits::MaxTag);

            int i = ActSize;

            // just one tag
            if (MinActSize == ActSize && 0 != pAct [MinActSize - 1]) {
                continue;
            // delimiter and fn(s), but no tag
            } else if (MinActSize < ActSize && 0 == pAct [MinActSize - 1]) {
                i = MinActSize;
            // tag, delimiter and fn(s)
            } else if (MinActSize + 1 < ActSize && 0 == pAct [MinActSize]) {
                i = MinActSize + 1;
            } else {
                // invalid action
                LogAssert (false);
            }

            // validate the function id
            for (; i < ActSize; ++i) {
                const int FnId = pAct [i];
                // bad function id
                LogAssert (0 <= FnId && (unsigned) FnId < m_Fn2IniSize);
                LogAssert (m_pFn2Ini && 0 <= m_pFn2Ini [FnId]);
            }
        } // of while (-1 != (ActSize = ...

    } // of if (m_pActs && m_pDfa) ...
}


template < class Ty >
const int FALexTools_t< Ty >::
    Process_int (
            const int Initial,
            const int Offset,
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize,
            const int RecDepth,
            const bool fOnce
        ) const
{
    int OutSize = 0;
    int Iw;
    int Dst;

    if (m_MaxDepth < RecDepth) {
        return 0;
    }

    const int MaxTokenLength = m_MaxTokenLength;

    /// iterate thru all possible start positions
    for (int FromPos = -1; FromPos < InSize; ++FromPos) {

        int State = Initial;
        int FinalState = -1;
        int FinalPos = -1;

        int j = FromPos;

        // maximum token length bounds j
        int LengthBound = FromPos + MaxTokenLength;
        if (InSize < LengthBound) {
            LengthBound = InSize;
        }

        /// feed the left anchor, if appropriate
        if (-1 == j) {
            State = m_pDfa->GetDest (Initial, FAFsmConst::IW_L_ANCHOR);
            if (-1 == State) {
                State = m_pDfa->GetDest (Initial, FAFsmConst::IW_ANY);
                if (-1 == State)
                    continue;
            }
            j++;
        }

        /// feed the letters
        for (; j < LengthBound; ++j) {

            Iw = pIn [j];
            // prevent regular input weights to match control input weights
            if (FAFsmConst::IW_EPSILON > Iw) {
                Iw = DefSubIw;
            }
            if (m_IgnoreCase) {
                Iw = FAUtf32ToLower (Iw);
            }
            Dst = m_pDfa->GetDest (State, Iw);
            if (-1 == Dst) {
                Dst = m_pDfa->GetDest (State, FAFsmConst::IW_ANY);
                if (-1 == Dst)
                    break;
            }
            if (m_pDfa->IsFinal (Dst)) {
                FinalState = Dst;
                FinalPos = j;
            }
            State = Dst;

        } // of for (; j < InSize; ...

        /// feed the right anchor, if appropriate
        if (InSize == j) {
            DebugLogAssert (-1 != State);
            Dst = m_pDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
            if (-1 == Dst) {
                Dst = m_pDfa->GetDest (State, FAFsmConst::IW_ANY);
            }
            if (-1 != Dst && m_pDfa->IsFinal (Dst)) {
                FinalState = Dst;
                FinalPos = j;
            }
        }

        // use the FinalState and the deepest FinalPos(ition)
        if (-1 != FinalPos) {
            DebugLogAssert (-1 != FinalState);
            DebugLogAssert (FinalPos >= FromPos);

            const int Ow = m_pState2Ow->GetOw (FinalState);

            const int * pAct;
            const int ActSize = m_pActs->Get (Ow, &pAct);
            DebugLogAssert (MinActSize <= ActSize && pAct);

            const int LeftCx = pAct [0];
            const int RightCx = pAct [1];
            const int Tag = pAct [2];

            // From position in the data 
            int FromPos2 = FromPos + LeftCx;
            if (0 > FromPos2)
            {
                FromPos2 = 0;
            }
            else if (InSize <= FromPos2)
            {
                FromPos2 = InSize - 1;
            }

            // To position in the data 
            int ToPos2 = FinalPos - RightCx;
            if (0 > ToPos2)
            {
                ToPos2 = 0;
            }
            else if (InSize <= ToPos2)
            {
                ToPos2 = InSize - 1;
            }

            int FnIdx = MinActSize;

            // create the token, if Tag is specified
            if (0 != Tag) {
                if (OutSize + 3 <= MaxOutSize) {
                    pOut [OutSize++] = Tag;
                    pOut [OutSize++] = FromPos2 + Offset;
                    pOut [OutSize++] = ToPos2 + Offset;
                } else {
                    // stop processing, the output buffer is not enough
                    return OutSize;
                }
                FnIdx = MinActSize + 1;
            }

            // set "once" flag for called functions, if there is more than one
            const bool fFnOnce = 1 < (ActSize - FnIdx);
            // functions' starting position
            int FnFrom = FromPos2;

            // apply functions, if any
            for (; FnIdx < ActSize; ++FnIdx) {

                const int FnId = pAct [FnIdx];
                DebugLogAssert (0 <= FnId && (unsigned) FnId < m_Fn2IniSize);

                const int FnIni = m_pFn2Ini [FnId];
                DebugLogAssert (-1 != FnIni);

                const Ty * pFnIn = pIn + FnFrom;
                const int FnInSize = ToPos2 - FnFrom + 1;
                int * pFnOut = pOut + OutSize;
                const int FnMaxOutSize = MaxOutSize - OutSize;

                const int FnOutSize = Process_int (FnIni, FnFrom + Offset, \
                  pFnIn, FnInSize, pFnOut, FnMaxOutSize, RecDepth + 1, \
                  0 == FnId ? false : fFnOnce);
                DebugLogAssert (0 == FnOutSize % 3);

                DebugLogAssert (FnOutSize <= FnMaxOutSize);
                __analysis_assume (FnOutSize <= FnMaxOutSize);

                // see if any tokens have been extracted
                if (0 < FnOutSize) {
                    // updated the total output count
                    OutSize += FnOutSize;
                    // next function starts from the last token's To + 1
                    FnFrom = pOut [OutSize - 1] + 1 - Offset;
                    // see if no text is left
                    if (FnFrom > ToPos2) {
                        break;
                    }
                }
            } // of for (; FnIdx < ActSize; ...

            // check if the function is supposed to be called once
            if (fOnce) {
                return OutSize;
            }

            // see if we can move the FromPos to the right more than one step
            if (FinalPos - RightCx > FromPos)
            {
                FromPos = FinalPos - RightCx;
            }

        } // of if (-1 != FinalPos)

    } // of for (FromPos = 0;

    return OutSize;
}


template < class Ty >
const int FALexTools_t< Ty >::
    Process (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        ) const
{
    if (!m_pActs || !m_pDfa || !m_pState2Ow) {
        return -1;
    }

    const int Initial = m_pDfa->GetInitial ();

    const int OutSize = Process_int (Initial, 0, pIn, InSize, pOut, MaxOutSize, 1);

    return OutSize;
}


template < class Ty >
const int FALexTools_t< Ty >::
    Process (
            const int FnTag,
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        ) const
{
    if (!m_pActs || !m_pDfa || !m_pState2Ow) {
        return -1;
    }

    if (0 == FnTag) {

        const int Initial = m_pDfa->GetInitial ();
        const int OutSize = Process_int (Initial, 0, pIn, InSize, pOut, MaxOutSize, 1);
        return OutSize;

    } else if (0 < FnTag && (unsigned int) FnTag < m_Fn2IniSize) {

        const int FnIni = m_pFn2Ini [FnTag];
        if (-1 == FnIni) {
            // the function tag is unknown
            return -1;
        }
        const int OutSize = Process_int (FnIni, 0, pIn, InSize, pOut, MaxOutSize, 1);
        return OutSize;

    }

    // the function tag is unknown
    return -1;
}

}

#endif
