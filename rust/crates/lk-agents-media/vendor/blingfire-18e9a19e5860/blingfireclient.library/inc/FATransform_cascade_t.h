/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORM_CASCADE_T_H_
#define _FA_TRANSFORM_CASCADE_T_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Implements a composition of n (up to 4) transformations.
///

template < class Ty >
class FATransform_cascade_t : public FATransformCA_t < Ty > {

public:
    FATransform_cascade_t ();
    virtual ~FATransform_cascade_t ();

public:
    /// adds next transformation into a transformation chain
    void AddTransformation (const  FATransformCA_t < Ty > * pTr);

    /// returns object into initial state
    void Clear ();

    /// makes transformation, pOut = pTrN( ... pTr2(pTr1(pIn)) ... )
    const int Process (
            const Ty * pIn,
            const int InCount,
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;

private:
    enum {
        MaxTrCount = 4,
    };

    /// keeps an array of transformations
    const  FATransformCA_t < Ty > * m_pTrs [MaxTrCount];
    unsigned int m_TrCount;
};


template < class Ty >
FATransform_cascade_t< Ty>::FATransform_cascade_t () :
      m_TrCount (0)
{
}


template < class Ty >
FATransform_cascade_t< Ty>::~FATransform_cascade_t ()
{}


template < class Ty >
void FATransform_cascade_t< Ty>::
    AddTransformation (const  FATransformCA_t < Ty > * pTr)
{
    LogAssert (m_TrCount < MaxTrCount);

    m_pTrs [m_TrCount++] = pTr;
}


template < class Ty >
void FATransform_cascade_t< Ty>::Clear ()
{
    m_TrCount = 0;
}


template < class Ty >
const int FATransform_cascade_t< Ty>::
    Process (const Ty * pIn,
             const int InCount,
             __out_ecount(MaxOutSize) Ty * pOut,
             const int MaxOutSize) const
{
  DebugLogAssert (0 < m_TrCount && m_pTrs && *m_pTrs);

  // as __in_range(0, FALimits::MaxWordSize) const int InCount does not work
  __analysis_assume (0 < InCount && FALimits::MaxWordSize >= InCount && pIn);
  DebugLogAssert (0 < InCount && FALimits::MaxWordSize >= InCount && pIn);

  // apply first transformation pIn --> pOut
  int OutSize = (*m_pTrs)->Process (pIn, InCount, pOut, MaxOutSize);

  // see whether transformation cannot be applied
  if (-1 == OutSize) {

    // transformation is not done in-place
    if (pIn != pOut) {
      if (MaxOutSize >= InCount) {
        // copy input to the output
        memcpy (pOut, pIn, InCount * sizeof (Ty));
      } else {
        // output buffer if not large enough, cannot continue
        DebugLogAssert (0);
        return InCount;
      }
    }

    OutSize = InCount;
  }

  // apply all the other transformations
  for (unsigned int i = 1; i < m_TrCount; ++i) {

    const  FATransformCA_t < Ty > * pTr = m_pTrs [i];
    DebugLogAssert (pTr);

    // apply transformation in-place
    const int NewOutSize = pTr->Process (pOut, OutSize, pOut, MaxOutSize);

    if (-1 != NewOutSize) {
      // update the output size
      OutSize = NewOutSize;
    }
    if (MaxOutSize < OutSize) {
      // output buffer if not large enough, cannot continue
      DebugLogAssert (0);
      return OutSize;
    }
  }

  return OutSize;
}

}

#endif
