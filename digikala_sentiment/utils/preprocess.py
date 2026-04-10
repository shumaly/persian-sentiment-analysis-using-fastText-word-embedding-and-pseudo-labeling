import re


class TextPreprocessor:
    """Normalize and clean Persian review text before modeling."""

    _DIGIT_TRANSLATION = str.maketrans(
        "۱۲۳۴۵۶۷۸۹۰١٢٣٤٥٦٧٨٩٠",
        "12345678901234567890",
    )

    def clean(self, text: str) -> str:
        text = str(text).translate(self._DIGIT_TRANSLATION)
        text = re.sub(r"(?:\@|https?\://)\s+", " ", text)
        text = re.findall(r"[A-Za-z._]+|[^A-Za-z\W]+", text, re.UNICODE)
        text = " ".join(word for word in text)

        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\d", "", text)
        text = re.sub(r"\r?\n", ".", text)
        text = re.sub(r"-{2,}", "", text)
        text = re.sub(r"\s*\.{2,}", ".", text)
        text = re.sub(r"\s+(ن؟می)\s+", r"\1", text)
        text = re.sub(r"(!){2,}", r"\1", text)
        text = re.sub(r"(/ ){2,}", "", text)
        text = re.sub(r"( /){2,}", "", text)
        text = re.sub(r"(//){2,}", "", text)
        text = re.sub(r"(/){2,}", "", text)
        text = re.sub(r"(؟){2,}", r"\1", text)
        text = re.sub(r"_+", "", text)
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"([\n]+)[\t]*", r"\1", text)
        text = re.sub(r"\b[a-zA-Z]\b", "", text)
        text = re.sub(r"product", "", text)
        text = re.sub(r"dkp", "", text)
        text = re.sub(r"br", "", text)
        text = re.sub(r"mm", "", text)
        return text.strip()


# Backward-compatible alias for the original notebook-style API.
preprocess = TextPreprocessor
