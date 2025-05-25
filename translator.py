from __future__ import annotations

import re
import sys
from typing import Optional, TypeAlias, Union, cast

from isa import (
    INPUT_PORT,
    OUTPUT_PORT,
    Command,
    Opcode,
    command_to_binary32,
    value_to_binary32,
    write_code,
    write_commented_code,
)

MEMORY_START: int = 2

ASTNode: TypeAlias = Union[str, list["ASTNode"]]


class SyntaxEOFError(SyntaxError):
    def __init__(self) -> None:
        super().__init__("Unexpected EOF while parsing")


class SyntaxUnexpectedTokenError(SyntaxError):
    def __init__(self) -> None:
        super().__init__("Unexpected ')' encountered unexpectedly")


class InvalidVariableDeclarationError(ValueError):
    def __init__(self) -> None:
        super().__init__("Invalid variable declaration")


class UnknownOpcodeError(ValueError):
    def __init__(self, opcode_name: str) -> None:
        super().__init__(f"Unknown instruction: {opcode_name}")
        self.opcode_name = opcode_name


class UnknownOperandError(ValueError):
    def __init__(self, operand: str) -> None:
        super().__init__(f"Unknown operand: {operand}")
        self.operand = operand


class InvalidOperandTypeError(TypeError):
    def __init__(self) -> None:
        super().__init__("Invalid operand type")


class InvalidASTError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MissingProgramError(InvalidASTError):
    def __init__(self) -> None:
        super().__init__("Исходный файл должен содержать одно S-выражение (program ...)")


class MissingDataSectionError(InvalidASTError):
    def __init__(self) -> None:
        super().__init__("Отсутствует секция data")


class MissingTextSectionError(InvalidASTError):
    def __init__(self) -> None:
        super().__init__("Отсутствует секция text")


def tokenize(text: str) -> list[str]:
    text = re.sub(r";[^\n]*", "", text)
    tokens: list[str] = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', text)
    return tokens


def parse(tokens: list[str]) -> ASTNode:
    if not tokens:
        raise SyntaxEOFError()

    token = tokens.pop(0)

    if token == "(":
        lst: list[ASTNode] = []
        while tokens and tokens[0] != ")":
            lst.append(parse(tokens))
        if not tokens:
            raise SyntaxEOFError()
        tokens.pop(0)
        return lst

    if token == ")":
        raise SyntaxUnexpectedTokenError()

    return token


def parse_source(source: str) -> list[ASTNode]:
    tokens = tokenize(source)
    ast: list[ASTNode] = []
    while tokens:
        ast.append(parse(tokens))
    return ast


class Translator:
    def __init__(self) -> None:
        self.variables: dict[str, int] = {}
        self.labels: dict[str, int] = {}
        self.commands: list[Command] = []
        self.addr_counter: int = MEMORY_START
        self.code_start: int = 0

    def translate_data(self, data_ast: list[ASTNode]) -> None:
        for var_def in data_ast:
            if not (
                isinstance(var_def, list)
                and len(var_def) == 3
                and isinstance(var_def[0], str)
                and var_def[0] == "var"
                and isinstance(var_def[1], str)
            ):
                raise InvalidVariableDeclarationError()
            name = var_def[1]
            self.variables[name] = self.addr_counter
            self.addr_counter += 1

    def translate_text(self, text_ast: list[ASTNode]) -> None:
        # Первый проход: сбор меток
        addr = self.addr_counter
        for item in text_ast:
            if (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], str)
                and item[0] == "label"
                and isinstance(item[1], str)
            ):
                self.labels[item[1]] = addr
            else:
                addr += 1
                if (
                    isinstance(item, list)
                    and len(item) >= 1
                    and isinstance(item[0], str)
                    and item[0] in ["jmp", "jz", "jnz", "call", "lit", "in", "out"]
                    and len(item) > 1
                ):
                    addr += 1
        self.code_start = self.addr_counter

        # Второй проход: генерация команд
        addr = self.addr_counter
        for item in text_ast:
            if (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], str)
                and item[0] == "label"
            ):
                continue
            if isinstance(item, list):
                opcode_name = cast(str, item[0])
                operand: Optional[Union[str, int]] = None
                if len(item) > 1:
                    operand = item[1] if isinstance(item[1], (str, int)) else str(item[1])
                cmd = self.make_command(opcode_name, operand)
                self.commands.append(cmd)
                addr += 1
                if cmd.operand is not None:
                    addr += 1
            else:
                cmd = self.make_command(cast(str, item), None)
                self.commands.append(cmd)
                addr += 1

    def make_command(self, opcode_name: str, operand: Optional[Union[str, int]]) -> Command:
        opcode = Opcode.from_string(opcode_name)
        if opcode is None:
            raise UnknownOpcodeError(opcode_name)

        if operand is None:
            return Command(opcode)

        val: int
        if isinstance(operand, str):
            if operand in self.variables:
                val = self.variables[operand]
            elif operand in self.labels:
                val = self.labels[operand]
            elif operand == "input_port":
                val = INPUT_PORT
            elif operand == "output_port":
                val = OUTPUT_PORT
            else:
                try:
                    val = int(operand)
                except ValueError as exc:
                    raise UnknownOperandError(operand) from exc
        elif isinstance(operand, int):
            val = operand
        else:
            raise InvalidOperandTypeError()

        return Command(opcode, val)

    def write_binary(self, filename: str) -> None:
        code_lines: list[str] = []
        commented_lines: list[str] = []
        char_for_index = 10

        code_lines.append(value_to_binary32(self.code_start))
        commented_lines.append(f"0{' ' * (char_for_index - 1)} {value_to_binary32(self.code_start)} start_address")

        interrupt_addr = self.labels.get(".int1", 0)
        code_lines.append(value_to_binary32(interrupt_addr))
        commented_lines.append(f"1{' ' * (char_for_index - 1)} {value_to_binary32(interrupt_addr)} int_vector_1")

        def get_var_addr(item: tuple[str, int]) -> int:
            return item[1]

        for name, addr in sorted(self.variables.items(), key=get_var_addr):
            code_lines.append(value_to_binary32(0))
            commented_lines.append(f"{addr}{' ' * (char_for_index - len(str(addr)))} {value_to_binary32(0)} var_{name}")

        addr = self.code_start
        for cmd in self.commands:
            code_lines.append(command_to_binary32(cmd))
            commented_lines.append(f"{addr}{' ' * (char_for_index - len(str(addr)))} {command_to_binary32(cmd)} {cmd.opcode.mnemonic}")
            addr += 1
            if cmd.operand is not None:
                code_lines.append(value_to_binary32(cmd.operand))
                commented_lines.append(f"{addr}{' ' * (char_for_index - len(str(addr)))} {value_to_binary32(cmd.operand)} operand")
                addr += 1

        write_code(filename, "\n".join(code_lines))
        write_commented_code(filename + ".txt", "\n".join(commented_lines))


def main(source_file: str, target_file: str) -> None:
    with open(source_file, encoding="utf-8") as f:
        source = f.read()

    ast = parse_source(source)
    if len(ast) != 1 or not isinstance(ast[0], list) or not ast[0] or not isinstance(ast[0][0], str) or ast[0][0] != "program":
        raise MissingProgramError()

    program_ast = cast(list[ASTNode], ast[0])
    if len(program_ast) < 3:
        raise MissingDataSectionError()

    data_ast = cast(list[ASTNode], program_ast[1])
    if not (isinstance(data_ast, list) and data_ast and isinstance(data_ast[0], str) and data_ast[0] == "data"):
        raise MissingDataSectionError()
    data_ast = data_ast[1:]

    text_ast = cast(list[ASTNode], program_ast[2])
    if not (isinstance(text_ast, list) and text_ast and isinstance(text_ast[0], str) and text_ast[0] == "text"):
        raise MissingTextSectionError()
    text_ast = text_ast[1:]

    translator = Translator()
    translator.translate_data(data_ast)
    translator.translate_text(text_ast)
    translator.write_binary(target_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 lisp_translator.py <source.lisp> <output.bin>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
