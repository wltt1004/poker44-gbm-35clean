from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

TASK_ID = "fc-379fc029-green54-green-54-88740a5fc0-counterexample-v1"


def body_without_first_import_and_audit(path: str) -> str:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    removed_import = False
    output: list[str] = []
    for line in lines:
        if not removed_import and line.startswith("import "):
            removed_import = True
            continue
        if line.strip().startswith("#print axioms"):
            continue
        output.append(line)
    return "\n".join(output).strip()


def build_main() -> str:
    root = Path("Green54Check.lean").read_text(encoding="utf-8").strip()
    classification = body_without_first_import_and_audit("Green54Check/Classification.lean")
    final = body_without_first_import_and_audit("Green54Check/Final.lean")
    source = "\n\n".join((root, classification, final))

    kept: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            continue
        if stripped in {"namespace Green54Counterexample", "end Green54Counterexample"}:
            continue
        if stripped.startswith('local notation "γ"'):
            continue
        if stripped.startswith('local notation "g"'):
            continue
        if stripped.startswith("#print axioms"):
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = text.replace("/-!", "/-")
    text = text.replace("Ω", "Omega")
    text = text.replace("γ", "Green54.gaussianMeasureInf")
    text = re.sub(r"(?<![\w.])g(?![\w])", "(gaussianReal 0 1)", text)
    target = [
        "",
        "/- Exact counterexample-mode target required by the live task. -/",
        'theorem target : ¬ (fcTypeOfName% "Green54.green_54") := by',
        "  exact green54_counterexample",
        "",
    ]
    return text.rstrip() + "\n" + "\n".join(target)


@dataclass(frozen=True)
class Token:
    value: str
    line: int
    column: int
    depth: int
    line_start: bool


TOP_LEVEL_PROHIBITED = frozenset({
    "import", "prelude", "module", "axiom", "axioms", "unsafe", "extern",
    "foreign", "initialize", "elab", "syntax", "macro", "set_option", "run_tac",
})
ANYWHERE_PROHIBITED = frozenset({
    "import", "prelude", "module", "axiom", "axioms", "constant", "constants",
    "sorry", "admit", "sorryAx", "initialize", "builtin_initialize", "elab",
    "elab_rules", "syntax", "syntax_rules", "declare_syntax_cat", "macro",
    "macro_rules", "native_decide", "set_option", "include_str", "include_bytes",
    "implemented_by", "run_tac", "run_cmd", "attribute", "deriving",
    "compile_inductive", "init_quot", "docs_to_verso", "add_decl_doc",
    "register_tactic_tag", "tactic_extension", "recommended_spelling",
    "gen_injective_theorems", "notation", "infix", "infixl", "infixr", "prefix",
    "postfix", "mixfix", "meta", "partial", "coinductive", "instance", "unsafe",
    "extern", "foreign",
})


def lean_name_parts(value: str) -> tuple[str, ...]:
    rooted = value.removeprefix("_root_.")
    segments = re.findall(r"«[^»]*»|[^.]+", rooted)
    return tuple(part.removeprefix("«").removesuffix("»") for part in segments)


def lean_tokens(text: str) -> tuple[Token, ...]:
    tokens: list[Token] = []
    index = 0
    line = 1
    column = 1
    depth = 0
    line_start = True
    block_depth = 0
    length = len(text)
    while index < length:
        char = text[index]
        pair = text[index:index + 2]
        if block_depth:
            if pair == "/-":
                block_depth += 1; index += 2; column += 2
            elif pair == "-/":
                block_depth -= 1; index += 2; column += 2
            elif char == "\n":
                index += 1; line += 1; column = 1; line_start = True
            else:
                index += 1; column += 1
            continue
        if pair == "/-":
            block_depth = 1; index += 2; column += 2; continue
        if pair == "--":
            newline = text.find("\n", index + 2)
            index = length if newline < 0 else newline
            continue
        if char in {'"', "'"}:
            if char == '"' and index > 0 and text[index - 1] == "!":
                tokens.append(Token("__interpolated_string__", line, column, depth, line_start))
            quote = char; index += 1; column += 1
            while index < length:
                current = text[index]
                if current == "\\":
                    index += 2; column += 2
                elif current == quote:
                    index += 1; column += 1; break
                elif current == "\n":
                    index += 1; line += 1; column = 1
                else:
                    index += 1; column += 1
            continue
        if char == "\n":
            index += 1; line += 1; column = 1; line_start = True; continue
        if char.isspace():
            index += 1; column += 1; continue
        start_column = column
        start_line = line
        at_start = line_start
        if char.isalpha() or char == "_" or ord(char) > 127:
            end = index + 1
            while end < length and (
                text[end].isalnum() or text[end] in "_'." or ord(text[end]) > 127
            ):
                end += 1
            value = text[index:end]
            tokens.append(Token(value, start_line, start_column, depth, at_start))
            column += end - index; index = end; line_start = False; continue
        if char in "([{":
            tokens.append(Token(char, start_line, start_column, depth, at_start)); depth += 1
        elif char in ")]}" and depth:
            depth -= 1; tokens.append(Token(char, start_line, start_column, depth, at_start))
        else:
            tokens.append(Token(char, start_line, start_column, depth, at_start))
        index += 1; column += 1; line_start = False
    return tuple(tokens)


def check_policy(text: str) -> int:
    tokens = lean_tokens(text)
    violations: list[str] = []
    if len(tokens) > 200_000:
        violations.append("submission exceeds the 200000-token policy limit")
    if tokens and max(token.depth for token in tokens) > 4096:
        violations.append("submission exceeds the delimiter nesting limit")
    if any(len(line) > 100_000 for line in text.splitlines()):
        violations.append("submission exceeds the line-length limit")
    if any(unicodedata.category(char) in {"Cf", "Cs", "Co", "Cn"} for char in text):
        violations.append("submission contains prohibited Unicode")

    for index, token in enumerate(tokens):
        if token.value == "__interpolated_string__":
            violations.append(f"interpolated string at {token.line}:{token.column}")
        prohibited = next(
            (name for name in lean_name_parts(token.value) if name in ANYWHERE_PROHIBITED),
            None,
        )
        if prohibited:
            violations.append(f"{prohibited} is prohibited at {token.line}:{token.column}")
        if token.depth == 0 and token.line_start and token.value in TOP_LEVEL_PROHIBITED:
            violations.append(f"top-level {token.value} at {token.line}:{token.column}")
        next_value = tokens[index + 1].value if index + 1 < len(tokens) else ""
        if token.value == "@" and next_value == "[":
            violations.append(f"declaration attribute at {token.line}:{token.column}")
        if token.depth == 0 and token.value == "#" and next_value != "[":
            violations.append(f"top-level hash command at {token.line}:{token.column}")

    target_start = next((
        index for index in range(len(tokens) - 1)
        if tokens[index].value in {"theorem", "lemma"} and tokens[index + 1].value == "target"
    ), -1)
    target_assign = next((
        index for index in range(max(target_start, 0), len(tokens) - 1)
        if tokens[index].value == ":" and tokens[index + 1].value == "="
    ), -1)
    dependency_parts = lean_name_parts("Green54.green_54")
    dependency_short = dependency_parts[-1]
    references = tuple(
        index for index, token in enumerate(tokens)
        if lean_name_parts(token.value) == dependency_parts
        or dependency_short in lean_name_parts(token.value)
    )
    if any(not (target_start <= index < target_assign) for index in references):
        violations.append("source declaration dependency is prohibited: Green54.green_54")

    if violations:
        raise SystemExit("\n".join(dict.fromkeys(violations)))
    return len(tokens)


def build_wrapper(main: str) -> str:
    header = [
        'import FormalConjectures.GreensOpenProblems.«54»',
        "",
        'syntax (name := green54FcTypeTest) "fcTypeOfName% " str : term',
        "macro_rules",
        '  | `(fcTypeOfName% $_source:str) =>',
        "      `(True ↔",
        "        ∀ (K : Set (ℕ → ℝ)),",
        "          IsCompact K →",
        "            Balanced ℝ K →",
        "              (0.99 : ℝ≥0∞) ≤ Green54.gaussianMeasureInf K →",
        "                ∃ C : Set (ℕ → ℝ),",
        "                  IsCompact C ∧",
        "                    Convex ℝ C ∧",
        "                      C ⊆ (10 : ℝ) • K ∧",
        "                        (1e-2 : ℝ≥0∞) ≤ Green54.gaussianMeasureInf C)",
        "",
        "namespace Bounty",
        "",
    ]
    footer = ["", "#print axioms target", "", "end Bounty", ""]
    return "\n".join(header) + main + "\n".join(footer)


def main() -> None:
    dist = Path("dist")
    dist.mkdir(exist_ok=True)
    main_text = build_main()
    token_count = check_policy(main_text)
    (dist / "Main.lean").write_text(main_text, encoding="utf-8")
    (dist / "WrappedMain.lean").write_text(build_wrapper(main_text), encoding="utf-8")
    readme = "\n".join([
        "Green54 counterexample submission",
        "=================================",
        "",
        f"Task: {TASK_ID}",
        "",
        "Submit Main.lean only. The Conjectures CLI supplies imports and namespace.",
        "Run `conjectures verify` and `conjectures check` before paying.",
        "",
    ])
    (dist / "README.txt").write_text(readme, encoding="utf-8")
    print(f"Generated dist/Main.lean: {len(main_text.encode('utf-8'))} bytes")
    print(f"Static policy admitted: {token_count} Lean tokens")


if __name__ == "__main__":
    main()
