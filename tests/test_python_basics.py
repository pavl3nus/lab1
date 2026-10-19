"""Собственные тесты для первой части лабораторной."""

import pytest

from grader_contracts.python_basics import (
    PositiveIntegerInput as PosInt,
    TextInput as Txt,
    VectorPairInput as Pair,
)
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number,
    mse, multiplicative_persistence, prime_factorization, pyramid,
)


def test_vowels_simple_cases():
    assert count_vowels(Txt("Programming")) == 3
    assert count_vowels(Txt("XYZ")) == 0
    assert count_vowels(Txt("")) == 0
    assert count_vowels(Txt("OIuyt")) == 3


def test_unique_cases():
    assert has_unique_characters(Txt("qwerty")) is True
    assert has_unique_characters(Txt("qwertyq")) is False
    assert has_unique_characters(Txt("")) is True
    assert has_unique_characters(Txt("Zz")) is True  # регистр различается


def test_bits():
    assert count_one_bits(PosInt(0)) == 0
    assert count_one_bits(PosInt(2)) == 1
    assert count_one_bits(PosInt(1023)) == 10
    assert count_one_bits(PosInt(4096)) == 1


def test_persistence():
    assert multiplicative_persistence(PosInt(7)) == 0
    assert multiplicative_persistence(PosInt(25)) == 2
    assert multiplicative_persistence(PosInt(39)) == 3
    assert multiplicative_persistence(PosInt(77)) == 4


def test_mse_values():
    assert mse(Pair([3, 3], [3, 3])) == 0.0
    assert mse(Pair([0, 0], [1, 1])) == 1.0
    assert mse(Pair([10], [4])) == 36.0


def test_mse_errors():
    with pytest.raises(ValueError):
        mse(Pair([1, 2], [1]))
    with pytest.raises(ValueError):
        mse(Pair([], []))


def test_factorization_cases():
    assert prime_factorization(PosInt(1)) == ""
    assert prime_factorization(PosInt(2)) == "(2)"
    assert prime_factorization(PosInt(100)) == "(2**2)(5**2)"
    assert prime_factorization(PosInt(86240)) == "(2**5)(5)(7**2)(11)"


def test_pyramid_cases():
    assert pyramid(PosInt(1)) == 1
    assert pyramid(PosInt(30)) == 4  # 1+4+9+16
    assert pyramid(PosInt(3)) == "It is impossible"
    assert pyramid(PosInt(7)) == "It is impossible"


def test_balanced_numbers():
    assert is_balanced_number(PosInt(9)) is True
    assert is_balanced_number(PosInt(34)) is True
    assert is_balanced_number(PosInt(1234006)) is True
    assert is_balanced_number(PosInt(123456)) is False
    assert is_balanced_number(PosInt(1292)) is False