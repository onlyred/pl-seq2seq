from enum import Enum, unique

@unique
class Tokens(Enum):
    UNK = '<unk>'
    BOS = '<bos>'
    EOS = '<eos>'
    PAD = '<pad>'

@unique
class IDs(Enum):
    UNK = 0
    BOS = 1
    EOS = 2
    PAD = 3
