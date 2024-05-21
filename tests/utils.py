import jiwer


def wer(hypothesis: str, reference: str) -> float:
    import jiwer

    transformers = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    return jiwer.wer(
        reference,
        hypothesis,
        reference_transform=transformers,
        hypothesis_transform=transformers,
    )
