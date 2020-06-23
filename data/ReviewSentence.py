import re
class ReviewSentence():
    
    def __init__(self, sentence_type, body, targets=[]):
        self.sentence_type = sentence_type
        self.body = body
        self.targets = targets

    @classmethod
    def parse(cls, sentence):
        s = sentence.strip()
        if not s:
            return None

        if s.startswith("*"):
            return None  # comment

        if s.startswith("[t]"):
            return ReviewSentence("title", s[len("[t]"):])

        attr_body = s.split("##")
        if len(attr_body) != 2:
            attr_body = s.split("#")
            if len(attr_body) != 2:
                return ReviewSentence("", "")

        attr, body = attr_body
        if not attr:
            targets = []
            return ReviewSentence("review", body, targets)

        attrs = attr.split(",")
        attrs = [attr.strip() for attr in attrs if attr != '']

        if len(attrs) == 0:  # no targets
            targets = []
        else:
            score_pattern = re.compile('\[[^)]*\]')
            targets = [score_pattern.sub('', item) for item in attrs]
        return ReviewSentence("review", body, targets)

    def to_row(self):
        return [self.body, self.targets]