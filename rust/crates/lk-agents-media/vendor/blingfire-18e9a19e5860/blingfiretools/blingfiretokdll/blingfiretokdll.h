#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <mutex>
#include <assert.h>

namespace BlingFire 
{
// SENTENCE PIECE DELIMITER
#define __FASpDelimiter__ 0x2581

// DEFAULT HYPHEN
#define __FADefaultHyphen__ 0x2012

// WHITESPACE [\x0004-\x0020\x007F-\x009F\x00A0\x2000-\x200F\x202F\x205F\x2060\x2420\x2424\x3000\xFEFF]
#define __FAIsWhiteSpace__(C) ( \
        (C <=  0x20 || C == 0xa0   || (C >= 0x2000 && C <= 0x200f) || \
        C == 0x202f || C == 0x205f || C == 0x2060 || C == 0x2420 || \
        C == 0x2424 || C == 0x3000 || C == 0xfeff) \
    )

extern "C"
{
const int GetBlingFireTokVersion();
const int TextToSentencesWithOffsetsWithModel(const char * pInUtf8Str, int InUtf8StrByteCount,
    char * pOutUtf8Str, int * pStartOffsets, int * pEndOffsets, const int MaxOutUtf8StrByteCount,
    void * hModel);
const int TextToSentencesWithOffsets(const char * pInUtf8Str, int InUtf8StrByteCount,
    char * pOutUtf8Str, int * pStartOffsets, int * pEndOffsets, const int MaxOutUtf8StrByteCount);
const int TextToSentencesWithModel(const char * pInUtf8Str, int InUtf8StrByteCount, char * pOutUtf8Str, const int MaxOutUtf8StrByteCount, void * hModel);
const int TextToSentences(const char * pInUtf8Str, int InUtf8StrByteCount, char * pOutUtf8Str, const int MaxOutUtf8StrByteCount);
const int TextToWordsWithOffsetsWithModel(const char * pInUtf8Str, int InUtf8StrByteCount,
    char * pOutUtf8Str, int * pStartOffsets, int * pEndOffsets, const int MaxOutUtf8StrByteCount,
    void * hModel);
const int TextToWordsWithOffsets(const char * pInUtf8Str, int InUtf8StrByteCount,
    char * pOutUtf8Str, int * pStartOffsets, int * pEndOffsets, const int MaxOutUtf8StrByteCount);
const int TextToWordsWithModel(const char * pInUtf8Str, int InUtf8StrByteCount,
    char * pOutUtf8Str, const int MaxOutUtf8StrByteCount, void * hModel);
const int TextToWords(const char * pInUtf8Str, int InUtf8StrByteCount, char * pOutUtf8Str, const int MaxOutUtf8StrByteCount);
const int NormalizeSpaces(const char * pInUtf8Str, int InUtf8StrByteCount, char * pOutUtf8Str, const int MaxOutUtf8StrByteCount, const int uSpace = __FASpDelimiter__);
const int TextToHashes(const char * pInUtf8Str, int InUtf8StrByteCount, int32_t * pHashArr, const int MaxHashArrLength, int wordNgrams, int bucketSize = 2000000);
const int WordHyphenationWithModel(const char * pInUtf8Str, int InUtf8StrByteCount,
    char * pOutUtf8Str, const int MaxOutUtf8StrByteCount, void * hModel, const int uHy = __FADefaultHyphen__);
void* SetModel(const unsigned char * pImgBytes, int ModelByteCount);
void* LoadModel(const char * pszLdbFileName);
const int TextToIdsWithOffsets_wp(
        void* ModelPtr,
        const char * pInUtf8Str,
        int InUtf8StrByteCount,
        int32_t * pIdsArr, 
        int * pStartOffsets, 
        int * pEndOffsets,
        const int MaxIdsArrLength,
        const int UnkId = 0
);
const int TextToIds_wp(
        void* ModelPtr,
        const char * pInUtf8Str,
        int InUtf8StrByteCount,
        int32_t * pIdsArr,
        const int MaxIdsArrLength,
        const int UnkId = 0
);
const int TextToIdsWithOffsets_sp(
        void* ModelPtr,
        const char * pInUtf8Str,
        int InUtf8StrByteCount,
        int32_t * pIdsArr,
        int * pStartOffsets, 
        int * pEndOffsets,
        const int MaxIdsArrLength,
        const int UnkId = 0
);
const int TextToIds_sp(
        void* ModelPtr,
        const char * pInUtf8Str,
        int InUtf8StrByteCount,
        int32_t * pIdsArr,
        const int MaxIdsArrLength,
        const int UnkId = 0
);
const int TextToIdsWithOffsets(
        void* ModelPtr,
        const char * pInUtf8Str,
        int InUtf8StrByteCount,
        int32_t * pIdsArr,
        int * pStartOffsets, 
        int * pEndOffsets,
        const int MaxIdsArrLength,
        const int UnkId = 0
);
const int TextToIds(
        void* ModelPtr,
        const char * pInUtf8Str,
        int InUtf8StrByteCount,
        int32_t * pIdsArr,
        const int MaxIdsArrLength,
        const int UnkId = 0
);
int FreeModel(void* ModelPtr);
int SetNoDummyPrefix(void* ModelPtr, bool fNoDummyPrefix);
int IdsToText (void* ModelPtr, const int32_t * pIdsArr, const int IdsCount, char * pOutUtf8Str, const int MaxOutUtf8StrByteCount, bool SkipSpecialTokens);
}
}