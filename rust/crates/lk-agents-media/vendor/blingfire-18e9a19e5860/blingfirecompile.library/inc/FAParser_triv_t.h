/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSER_TRIV_T_H_
#define _FA_PARSER_TRIV_T_H_

#include "FAConfig.h"
#include "FAParser_base_t.h"
#include "FAState2OwsCA.h"

namespace BlingFire
{

///
/// This parser does not allows nestings at single pass, if there are
/// conflicts it prefers left-most longest constituent, if there are more
/// than one constituent of that kind it takes one with bigger tag value.
///

template < class Ty >
class FAParser_triv_t : public FAParser_base_t < Ty > {

public:
    FAParser_triv_t (FAAllocatorA * pAlloc);
    ~FAParser_triv_t ();

public:
    void SetActions (const FAMultiMapCA * pRule2Act);

private:
    void Prepare ();
    const int UpdateResults (
            const int FromPos,
            const int ToPos,
            const int State
        );
    const bool ApplyResults ();

private:
    /// max left context
    int m_MaxLc;

    /// mapping From --> < Tag, To > pair
    FAArray_cont_t < int > m_From2To;
    int * m_pFrom2To;
    FAArray_cont_t < int > m_From2Tag;
    int * m_pFrom2Tag;

    /// mapping From --> span's < From, To >
    FAArray_cont_t < int > m_From2SpanFrom;
    int * m_pFrom2SpanFrom;
    FAArray_cont_t < int > m_From2SpanTo;
    int * m_pFrom2SpanTo;

    enum {
        DefActionSize = 3,      // e.g. < Lc, Rc, Tag >
    };

    typedef FAParser_base_t < Ty > BASE;
};


template < class Ty >
FAParser_triv_t< Ty >::FAParser_triv_t (FAAllocatorA * pAlloc) :
    FAParser_base_t < Ty > (pAlloc),
    m_MaxLc (0),
    m_pFrom2To (NULL),
    m_pFrom2Tag (NULL),
    m_pFrom2SpanFrom (NULL),
    m_pFrom2SpanTo (NULL)
{
    m_From2To.SetAllocator (pAlloc);
    m_From2To.Create ();

    m_From2Tag.SetAllocator (pAlloc);
    m_From2Tag.Create ();

    m_From2SpanFrom.SetAllocator (pAlloc);
    m_From2SpanFrom.Create ();

    m_From2SpanTo.SetAllocator (pAlloc);
    m_From2SpanTo.Create ();
}


template < class Ty >
FAParser_triv_t< Ty >::~FAParser_triv_t ()
{}


template < class Ty >
void FAParser_triv_t< Ty >::
    SetActions (const FAMultiMapCA * pRule2Act)
{
    BASE::m_pRule2Act = pRule2Act;
    m_MaxLc = 0;

    if (BASE::m_pRule2Act) {

        /// rule nums are contiguous values starting from 0
        int RuleNum = 0;
        const int * pAct;
        int Size = BASE::m_pRule2Act->Get (RuleNum++, &pAct);
        DebugLogAssert (DefActionSize == Size && pAct);

        while (-1 != Size) {

            const int Lc = *pAct;

            if (Lc > m_MaxLc) {
                m_MaxLc = Lc;
            }

            Size = BASE::m_pRule2Act->Get (RuleNum++, &pAct);
            DebugLogAssert ((-1 == Size) || (DefActionSize == Size && pAct));

        } // of while (-1 != Size) ...

        DebugLogAssert (0 <= m_MaxLc);

    } // of if (BASE::m_pRule2Act) ...
}


template < class Ty >
void FAParser_triv_t< Ty >::Prepare ()
{
    BASE::Prepare ();

    DebugLogAssert (m_From2To.size () == m_From2Tag.size ());

    const int OldSize = m_From2To.size ();

    m_From2To.resize (BASE::m_UpperCount);
    m_From2Tag.resize (BASE::m_UpperCount);
    m_From2SpanFrom.resize (BASE::m_UpperCount);
    m_From2SpanTo.resize (BASE::m_UpperCount);

    m_pFrom2To = m_From2To.begin ();
    m_pFrom2Tag = m_From2Tag.begin ();
    m_pFrom2SpanFrom = m_From2SpanFrom.begin ();
    m_pFrom2SpanTo = m_From2SpanTo.begin ();

    // initialize, what is necessary
    if (OldSize < BASE::m_UpperCount) {

        const int IniCount = BASE::m_UpperCount - OldSize;

        memset (m_pFrom2To + OldSize, -1, IniCount * sizeof (int));
        memset (m_pFrom2Tag + OldSize, -1, IniCount * sizeof (int));
        memset (m_pFrom2SpanFrom + OldSize, -1, IniCount * sizeof (int));
        memset (m_pFrom2SpanTo + OldSize, -1, IniCount * sizeof (int));
    }

// debug check
#ifndef NDEBUG
    for (int i = 0; i < BASE::m_UpperCount; ++i) {
        DebugLogAssert (-1 == m_From2To [i]);
        DebugLogAssert (-1 == m_From2Tag [i]);
        DebugLogAssert (-1 == m_From2SpanFrom [i]);
        DebugLogAssert (-1 == m_From2SpanTo [i]);
    }
#endif

}


template < class Ty >
const int FAParser_triv_t< Ty >::UpdateResults (
        const int FromPos, 
        const int ToPos, 
        const int State
    )
{
    DebugLogAssert (BASE::m_pState2Rules);
    DebugLogAssert (m_From2To.begin () == m_pFrom2To);
    DebugLogAssert (m_From2Tag.begin () == m_pFrom2Tag);
    DebugLogAssert (m_From2SpanFrom.begin () == m_pFrom2SpanFrom);
    DebugLogAssert (m_From2SpanTo.begin () == m_pFrom2SpanTo);
    DebugLogAssert (m_From2To.size () == m_From2Tag.size ());
    DebugLogAssert (m_From2SpanFrom.size () == m_From2Tag.size ());
    DebugLogAssert (m_From2SpanTo.size () == m_From2Tag.size ());
    DebugLogAssert ((unsigned int) BASE::m_UpperCount <= m_From2To.size ());

    bool NoLc = true;

    const int Count = BASE::m_pState2Rules->GetOws \
        (State, BASE::m_pRules, BASE::m_MaxRulesSize);
    DebugLogAssert (0 < Count && Count <= BASE::m_MaxRulesSize);

    const int * pAct;

    for (int i = 0; i < Count; ++i) {

        const int RuleNum = BASE::m_pRules [i];

#ifndef NDEBUG
        const int ActSize = 
#endif
            BASE::m_pRule2Act->Get (RuleNum, &pAct);
        DebugLogAssert (DefActionSize == ActSize && pAct);

        if (0 != pAct [0]) {
            NoLc = false;
        }

        // += Left Context Size
        int From2 = FromPos + pAct [0];
        if (0 > From2) {
            From2 = 0; // due to the included left anchor
        }
        // -= Right Context Size
        int To2 = ToPos - pAct [1];
        if (To2 >= BASE::m_UpperCount) {
            To2 = BASE::m_UpperCount - 1; // due to the included right anchor
        }
        // get action Tag
        const int Tag = pAct [2];

        DebugLogAssert (0 <= From2 && 0 <= To2 && From2 <= To2);
        DebugLogAssert (FALimits::MinTag <= Tag && FALimits::MaxTag >= Tag);

        const int LastTo2 = m_pFrom2To [From2];
        const int LastSpanFrom = m_pFrom2SpanFrom [From2];
        const int LastSpanTo = m_pFrom2SpanTo [From2];

        // due to FromPos never decreases from call to call
        DebugLogAssert (!(To2 == LastTo2 && FromPos < LastSpanFrom));

        if (To2 > LastTo2) {

            m_pFrom2Tag [From2] = Tag;
            m_pFrom2To [From2] = To2;
            m_pFrom2SpanFrom [From2] = FromPos;
            m_pFrom2SpanTo [From2] = ToPos;

        } else if (To2 == LastTo2 && FromPos == LastSpanFrom) {

            if (ToPos > LastSpanTo) {

                m_pFrom2Tag [From2] = Tag;
                m_pFrom2SpanTo [From2] = ToPos;

            } else if (ToPos == LastSpanTo && Tag > m_pFrom2Tag [From2]) {

                m_pFrom2Tag [From2] = Tag;
            }
        }

    } // of for (int i = 0; ...

    // see whether it is possible to skip something
    if (NoLc) {

        const int MaxTo = m_pFrom2To [FromPos];
        const int NextPos = MaxTo - m_MaxLc;

        if (NextPos > FromPos) {
            return NextPos;
        }
    }

    return FromPos;
}


template < class Ty >
const bool FAParser_triv_t< Ty >::ApplyResults ()
{
    DebugLogAssert (m_From2To.begin () == m_pFrom2To);
    DebugLogAssert (m_From2Tag.begin () == m_pFrom2Tag);
    DebugLogAssert (m_From2SpanFrom.begin () == m_pFrom2SpanFrom);
    DebugLogAssert (m_From2SpanTo.begin () == m_pFrom2SpanTo);
    DebugLogAssert (m_From2To.size () == m_From2Tag.size ());
    DebugLogAssert (m_From2SpanFrom.size () == m_From2Tag.size ());
    DebugLogAssert (m_From2SpanTo.size () == m_From2Tag.size ());
    DebugLogAssert ((unsigned int) BASE::m_UpperCount <= m_From2To.size ());

    bool ResFound = false;

    for (int From = 0; From < BASE::m_UpperCount; ++From) {

        DebugLogAssert (-1 != m_pFrom2To [From] || -1 == m_pFrom2Tag [From]);
        const int To = m_pFrom2To [From];

        if (-1 != To) {

            DebugLogAssert (To >= From);

            const int Tag = m_pFrom2Tag [From];
            DebugLogAssert (FALimits::MinTag <= Tag && FALimits::MaxTag >= Tag);

            BASE::m_pTree->AddNode (-Tag, From, To);

            From = To;

            ResFound = true;
        }
    }

    if (ResFound) {

        const int Count = BASE::m_UpperCount * sizeof (int);

        memset (m_pFrom2To, -1, Count);
        memset (m_pFrom2Tag, -1, Count);
        memset (m_pFrom2SpanFrom, -1, Count);
        memset (m_pFrom2SpanTo, -1, Count);
    }

    return ResFound;
}

}

#endif
