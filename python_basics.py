"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

def count_vowels(data: TextInput) -> int:
    result = 0
    for symbol in data.value.lower():
        if symbol in "aeiou":
            result += 1
    return result


def has_unique_characters(data: TextInput) -> bool:
    seen = []
    for symbol in data.value:
        if symbol in seen:
            return False
        seen.append(symbol)
    return True


def count_one_bits(data: PositiveIntegerInput) -> int:
    n = data.value
    total = 0
    while n:
        total += n & 1
        n >>= 1
    return total


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    n = data.value
    iterations = 0
    while n > 9:
        digits_product = 1
        while n:
            digits_product *= n % 10
            n //= 10
        n = digits_product
        iterations += 1
    return iterations


def mse(data: VectorPairInput) -> float:
    a, b = data.predicted, data.expected
    if len(a) != len(b):
        raise ValueError("Векторы разной длины!")
    if not a:
        raise ValueError("Векторы пустые!")
    accumulator = 0
    for i in range(len(a)):
        diff = a[i] - b[i]
        accumulator += diff * diff
    return accumulator / len(a)


def prime_factorization(data: PositiveIntegerInput) -> str:
    n = data.value
    result = ""
    p = 2
    while p * p <= n:
        exponent = 0
        while n % p == 0:
            n //= p
            exponent += 1
        if exponent == 1:
            result += "(" + str(p) + ")"
        elif exponent > 1:
            result += "(" + str(p) + "**" + str(exponent) + ")"
        p += 1 if p == 2 else 2
    if n > 1:
        result += "(" + str(n) + ")"
    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cubes = data.value
    level = 1
    while cubes >= level * level:
        cubes -= level * level
        level += 1
    if cubes == 0 and level > 1:
        return level - 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = list(map(int, str(data.value)))
    n = len(digits)
    
    if n % 2 == 1:
        middle_index = n // 2
        left_part = digits[:middle_index]
        right_part = digits[middle_index + 1:]
    else:
        left_part = digits[:n // 2 - 1]
        right_part = digits[n // 2 + 1:]
        
    return sum(left_part) == sum(right_part)