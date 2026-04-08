import re


class preprocess:
    def clean(self, text):
        intab = "۱۲۳۴۵۶۷۸۹۰١٢٣٤٥٦٧٨٩٠"
        outtab = "12345678901234567890"
        translation_table = str.maketrans(intab, outtab)
        text = str(text).translate(translation_table)
        text = re.findall(r"[A-Za-z._]+|[^A-Za-z\W]+", text, re.UNICODE)
        text = " ".join(word for word in text)

        cleanr = re.compile("<.*?>")
        text = re.sub(cleanr, "", text)
        text = re.sub(r"\d", "", text)
        text = re.sub(r"\r?\n", ".", text)
        text = re.sub(r"-{3}", "", text)
        text = re.sub(r"-{2}", "", text)
        text = re.sub(r"\s*\.{3,}", ".", text)
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
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"product", "", text)
        text = re.sub(r"dkp", "", text)
        text = re.sub(r"br", "", text)
        text = re.sub(r"mm", "", text)
        return text.strip()
