from enum import Enum


class Modes(str, Enum):
    seconds = "seconds"
    turns = "turns"


class Formalisms(str, Enum):
    bm = "b&m"
    bjorken_mtinwga = "bjorken-mtingwa"
    nagaitsev = "nagaitsev"
