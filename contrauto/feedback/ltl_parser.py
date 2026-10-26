"""Parse CHASE LTL formula strings into a simple AST.

Handles the subset of LTL used in CHASE contracts:
    G  F  X  U  !  &  |  ->  <->
    Atoms: identifiers like at_shelf, r1_has_item
    Parentheses for grouping

Grammar (precedence low→high):
    iff     := impl (<-> impl)*
    impl    := or   (->  or  )*
    or      := and  (|   and )*
    and     := unary (&  unary)*
    unary   := G(φ) | F(φ) | X(φ) | !φ | until
    until   := primary (U primary)?
    primary := atom | (iff)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union

LTLNode = Union["Atom", "Not", "And", "Or", "Implies", "Iff", "G", "F", "X", "U_"]


@dataclass(frozen=True)
class Atom:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class Not:
    child: LTLNode
    def __repr__(self): return f"!{self.child}"

@dataclass(frozen=True)
class And:
    left: LTLNode
    right: LTLNode
    def __repr__(self): return f"({self.left} & {self.right})"

@dataclass(frozen=True)
class Or:
    left: LTLNode
    right: LTLNode
    def __repr__(self): return f"({self.left} | {self.right})"

@dataclass(frozen=True)
class Implies:
    left: LTLNode
    right: LTLNode
    def __repr__(self): return f"({self.left} -> {self.right})"

@dataclass(frozen=True)
class Iff:
    left: LTLNode
    right: LTLNode
    def __repr__(self): return f"({self.left} <-> {self.right})"

@dataclass(frozen=True)
class G:
    child: LTLNode
    def __repr__(self): return f"G({self.child})"

@dataclass(frozen=True)
class F:
    child: LTLNode
    def __repr__(self): return f"F({self.child})"

@dataclass(frozen=True)
class X:
    child: LTLNode
    def __repr__(self): return f"X({self.child})"

@dataclass(frozen=True)
class U_:
    left: LTLNode
    right: LTLNode
    def __repr__(self): return f"({self.left} U {self.right})"


class _Tokenizer:
    def __init__(self, text: str):
        self._text = text
        self._pos = 0
        self._tokens: list[str] = []
        self._idx = 0
        self._tokenize()

    def _tokenize(self):
        i = 0
        t = self._text
        while i < len(t):
            if t[i].isspace():
                i += 1
            elif t[i:i+3] == '<->':
                self._tokens.append('<->')
                i += 3
            elif t[i:i+2] == '->':
                self._tokens.append('->')
                i += 2
            elif t[i] in '()&|!':
                self._tokens.append(t[i])
                i += 1
            elif t[i].isalpha() or t[i] == '_':
                j = i
                while j < len(t) and (t[j].isalnum() or t[j] == '_'):
                    j += 1
                self._tokens.append(t[i:j])
                i = j
            else:
                i += 1

    def peek(self) -> str | None:
        if self._idx < len(self._tokens):
            return self._tokens[self._idx]
        return None

    def consume(self) -> str:
        tok = self._tokens[self._idx]
        self._idx += 1
        return tok

    def expect(self, tok: str):
        got = self.consume()
        if got != tok:
            raise ValueError(f"Expected '{tok}', got '{got}'")

    def done(self) -> bool:
        return self._idx >= len(self._tokens)


def parse(text: str) -> LTLNode:
    """Parse one LTL formula string (trailing ``;`` allowed) into an AST.

    Raises:
        ValueError: On a malformed formula.
    """
    text = text.strip().rstrip(';')
    tok = _Tokenizer(text)
    result = _parse_iff(tok)
    return result


def _parse_iff(tok: _Tokenizer) -> LTLNode:
    left = _parse_impl(tok)
    while tok.peek() == '<->':
        tok.consume()
        right = _parse_impl(tok)
        left = Iff(left, right)
    return left


def _parse_impl(tok: _Tokenizer) -> LTLNode:
    left = _parse_or(tok)
    while tok.peek() == '->':
        tok.consume()
        right = _parse_or(tok)
        left = Implies(left, right)
    return left


def _parse_or(tok: _Tokenizer) -> LTLNode:
    left = _parse_and(tok)
    while tok.peek() == '|':
        tok.consume()
        right = _parse_and(tok)
        left = Or(left, right)
    return left


def _parse_and(tok: _Tokenizer) -> LTLNode:
    left = _parse_unary(tok)
    while tok.peek() == '&':
        tok.consume()
        right = _parse_unary(tok)
        left = And(left, right)
    return left


def _parse_unary(tok: _Tokenizer) -> LTLNode:
    p = tok.peek()
    if p == '!':
        tok.consume()
        return Not(_parse_unary(tok))
    if p in ('G', 'F', 'X'):
        tok.consume()
        if tok.peek() == '(':
            tok.expect('(')
            child = _parse_iff(tok)
            tok.expect(')')
        else:
            child = _parse_unary(tok)
        if p == 'G': return G(child)
        if p == 'F': return F(child)
        return X(child)
    return _parse_until(tok)


def _parse_until(tok: _Tokenizer) -> LTLNode:
    left = _parse_primary(tok)
    if tok.peek() == 'U':
        tok.consume()
        right = _parse_primary(tok)
        return U_(left, right)
    return left


def _parse_primary(tok: _Tokenizer) -> LTLNode:
    p = tok.peek()
    if p == '(':
        tok.consume()
        node = _parse_iff(tok)
        tok.expect(')')
        return node
    if p is not None and (p[0].isalpha() or p[0] == '_') and p not in ('G', 'F', 'X', 'U'):
        tok.consume()
        return Atom(p)
    if p in ('G', 'F', 'X'):
        return _parse_unary(tok)
    raise ValueError(f"Unexpected token: {p}")
