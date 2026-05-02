import re
from better_profanity import profanity


class InputGuardrails:

    def validate(self, text):

        if len(text) < 2:
            return False, "Too short"

        if len(text) > 1000:
            return False, "Too long"

        if profanity.contains_profanity(text):
            return False, "Inappropriate language"

        patterns = [
            r"ignore .* instructions",
            r"system:",
            r"jailbreak"
        ]

        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return False, "Prompt injection detected"

        return True, text