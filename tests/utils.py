import jiwer as tr


def wer(hypothesis: str, reference: str) -> float:
    wer_standardize_contiguous = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.ExpandCommonEnglishContractions(),
            tr.RemoveKaldiNonWords(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
            tr.ReduceToListOfListOfWords(),
        ]
    )

    return tr.wer(
        reference,
        hypothesis,
        reference_transform=wer_standardize_contiguous,
        hypothesis_transform=wer_standardize_contiguous,
    )
