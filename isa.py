from __future__ import annotations

from enum import Enum

STACK_SIZE = 512
MEMORY_SIZE = 1024
MAX_NUMBER = (1 << 31) - 1
MIN_NUMBER = -(1 << 31)
INPUT_PORT = 0
OUTPUT_PORT = 1


class Variable:
    """Представляет переменную в памяти."""

    def __init__(self, name: str, address: int, data: list[int], is_string: bool):
        self.name = name
        self.address = address
        self.data = data
        self.is_string = is_string


class Opcode(Enum):
    """Перечисление поддерживаемых инструкций с мнемониками и бинарным кодом."""

    ADD     = ("add",     "00001")
    SUB     = ("sub",     "00010")
    MUL     = ("mul",     "00011")
    DIV     = ("div",     "00100")
    MOD     = ("mod",     "00101")
    INC     = ("inc",     "00110")
    DEC     = ("dec",     "00111")
    DUP     = ("dup",     "01000")
    OVER    = ("over",    "01001")
    SWAP    = ("swap",    "01010")
    CMP     = ("cmp",     "01011")
    JMP     = ("jmp",     "01100")
    JZ      = ("jz",      "01101")
    JNZ     = ("jnz",     "01110")
    CALL    = ("call",    "01111")
    RET     = ("ret",     "10000")
    LIT     = ("lit",     "10001")
    PUSH    = ("push",    "10010")
    POP     = ("pop",     "10011")
    DROP    = ("drop",    "10100")
    EI      = ("ei",      "10101")
    DI      = ("di",      "10110")
    IRET    = ("iret",    "10111")
    HALT    = ("halt",    "11000")
    IN      = ("in",      "11001")
    OUT     = ("out",     "11010")

    def __init__(self, mnemonic: str, binary: str):
        self.mnemonic = mnemonic
        self.binary = binary

    @classmethod
    def from_string(cls, mnemonic: str) -> Opcode | None:
        """Находит Opcode по мнемонике."""
        return next((op for op in cls if op.mnemonic == mnemonic), None)

    @classmethod
    def from_binary(cls, binary: str) -> Opcode | None:
        """Находит Opcode по бинарному коду."""
        return next((op for op in cls if op.binary == binary), None)


class Command:
    """Представляет одну команду (инструкцию) в коде."""

    def __init__(self, opcode: Opcode, operand: int | None = None):
        self.opcode = opcode
        self.operand = operand


def write_commented_code(filename: str, commented_code: str) -> None:
    """Записывает человекочитаемую версию кода с комментариями в файл."""
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(commented_code)


def write_code(filename: str, code: str) -> None:
    """Записывает бинарный (скомпилированный) код в файл в UTF-8 формате."""
    with open(filename, mode="bw") as f:
        f.write(code.encode("utf-8"))


def read_code(filename: str) -> list[int]:
    """Читает бинарный код из файла и декодирует его в список целых чисел."""
    with open(filename, mode="rb") as f:
        code = f.read()
        lines = code.decode("utf-8").splitlines()
        return [binary32_to_int(line) for line in lines]


def value_to_binary32(value: int) -> str:
    """Кодирует знаковое целое число в 32-битную бинарную строку."""
    return format(value & 0xFFFFFFFF, "032b")


def command_to_binary32(command: Command) -> str:
    """Кодирует команду в 32-битную бинарную строку (только opcode)."""
    return command.opcode.binary + "0" * 27


def binary32_to_int(value: str) -> int:
    """Декодирует 32-битную бинарную строку в знаковое целое число."""
    num = int(value, 2)
    if value[0] == "1":
        num -= 1 << 32
    return num

class UnknownOpcodeError(ValueError):
    """Вызывается, когда переданный бинарный код не соответствует ни одному опкоду."""
    def __init__(self, binary: str):
        super().__init__(f"Unknown opcode binary: {binary}")
        self.binary = binary


def int_to_opcode(value: int) -> Opcode:
    """Извлекает opcode из верхних 5 бит 32-битного целого числа-инструкции."""
    bits = value_to_binary32(value)[:5]
    opcode = Opcode.from_binary(bits)
    if opcode is None:
        raise UnknownOpcodeError(bits)
    return opcode
