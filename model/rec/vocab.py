# Vietnamese Character Set
# This string contains all characters that the model should be able to recognize.

# Standard lowercase Vietnamese vowels with tones
vowels = "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ"

# Consonants and other characters
consonants = "bcdđghklmnpqrstvx"

# Digits and symbols
digits = "0123456789"
symbols = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

# Full vocabulary
# Note: We include both lowercase and uppercase
vietnamese_vocab = vowels + vowels.upper() + consonants + consonants.upper() + digits + symbols

# Export as a simple string
VOCAB = "".join(sorted(list(set(vietnamese_vocab))))
