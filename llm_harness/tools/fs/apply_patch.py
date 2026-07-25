"""Minimal patch edit tool.

Reference from OpenClaw apply-patch tool:
https://github.com/openclaw/openclaw/blob/c30cabcca42d5a41b0e129a7ce9d438ff539e792/src/agents/apply-patch.ts
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

BEGIN_PATCH_MARKER = "*** Begin Patch"
END_PATCH_MARKER = "*** End Patch"
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CHANGE_CONTEXT_MARKER = "@@"
PUNCTUATION_TRANSLATION = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u00a0": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u205f": " ",
    "\u3000": " ",
}
type LineNormalizer = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class PatchChunk:
    """One local block of line changes inside a file patch.

    Example:
        In this file patch for `app.py`:

            *** Update File: app.py
            @@
            -def greet():
            -    print("hi")
            +def greet(name: str):
            +    print(f"hi {name}")
            @@
            -def bye():
            -    print("bye")
            +def bye(name: str):
            +    print(f"bye {name}")

        each `@@` section is one patch chunk.
    """

    change_context: str | None
    old_lines: list[str]
    new_lines: list[str]
    is_end_of_file: bool
    removed_lines: int
    inserted_lines: int


@dataclass(frozen=True, slots=True)
class FilePatch:
    """All patch chunks that apply to one file.

    Example:
        In this patch:

            *** Update File: app.py
            @@
            -def greet():
            -    print("hi")
            +def greet(name: str):
            +    print(f"hi {name}")
            @@
            -def bye():
            -    print("bye")
            +def bye(name: str):
            +    print(f"bye {name}")

        the whole `*** Update File: app.py` section is one file patch.
        Its two `@@` sections are the two patch chunks inside it.
    """

    path: str
    move_path: str | None
    chunks: list[PatchChunk]


@dataclass(frozen=True, slots=True)
class PatchStats:
    """Summarize how many chunks and line changes a parsed patch contains."""

    chunk_count: int
    lines_removed: int
    lines_inserted: int

    @property
    def lines_touched(self) -> int:
        """Return the number of distinct lines touched by a patch."""
        return self.lines_removed + self.lines_inserted


def parse_single_file_patch_with_stats(
    *,
    patch_text: str,
    target_path: str | None = None,
) -> tuple[FilePatch, PatchStats]:
    """Parse one-file patch text and return both the patch and its summary stats."""
    file_patches = _parse_file_patches(patch_text)
    if not file_patches:
        raise ValueError("No files were modified.")
    if len(file_patches) != 1:
        raise ValueError("Patch must update exactly one file.")

    file_patch = _validate_single_file_patch(file_patches[0], target_path=target_path)
    return file_patch, _collect_patch_stats(file_patches)


def _parse_file_patches(patch_text: str) -> list[FilePatch]:
    """Parse raw patch text into ordered per-file patch sections."""
    stripped = patch_text.strip()
    if not stripped:
        raise ValueError("Patch input is empty.")

    lines = stripped.splitlines()
    if lines[0].strip() != BEGIN_PATCH_MARKER:
        raise ValueError("The first line of the patch must be '*** Begin Patch'.")
    if lines[-1].strip() != END_PATCH_MARKER:
        raise ValueError("The last line of the patch must be '*** End Patch'.")

    file_patches: list[FilePatch] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if not line.startswith(UPDATE_FILE_MARKER):
            raise ValueError(f"Unsupported patch header: {lines[index]!r}")

        path = line[len(UPDATE_FILE_MARKER) :].strip()
        index += 1
        move_path: str | None = None
        if index < len(lines) - 1 and lines[index].strip().startswith(MOVE_TO_MARKER):
            move_path = lines[index].strip()[len(MOVE_TO_MARKER) :].strip()
            index += 1

        chunks: list[PatchChunk] = []
        while index < len(lines) - 1:
            current = lines[index]
            stripped_current = current.strip()
            if not stripped_current:
                index += 1
                continue
            if stripped_current.startswith("*** "):
                break

            chunk, consumed = _parse_patch_chunk(
                lines=lines[index : len(lines) - 1],
            )
            chunks.append(chunk)
            index += consumed

        if not chunks:
            raise ValueError(f"Update file patch for {path!r} is empty.")
        file_patches.append(FilePatch(path=path, move_path=move_path, chunks=chunks))

    return file_patches


def _parse_patch_chunk(
    *,
    lines: list[str],
) -> tuple[PatchChunk, int]:
    """Parse a single `@@` chunk and report how many source lines it consumed."""
    if not lines:
        raise ValueError("Patch chunk does not contain any lines.")

    index = 0
    change_context: str | None = None
    first = lines[index]
    if first == EMPTY_CHANGE_CONTEXT_MARKER:
        index += 1
    elif first.startswith(CHANGE_CONTEXT_MARKER):
        change_context = first[len(CHANGE_CONTEXT_MARKER) :]
        index += 1

    old_lines: list[str] = []
    new_lines: list[str] = []
    is_end_of_file = False
    change_count = 0
    removed_lines = 0
    inserted_lines = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped == EOF_MARKER:
            is_end_of_file = True
            index += 1
            break
        if _is_chunk_boundary(line):
            break
        if not line:
            raise ValueError("Patch change lines must start with ' ', '+', or '-'.")

        prefix = line[0]
        payload = line[1:]
        if prefix == " ":
            old_lines.append(payload)
            new_lines.append(payload)
        elif prefix == "-":
            old_lines.append(payload)
            removed_lines += 1
        elif prefix == "+":
            new_lines.append(payload)
            inserted_lines += 1
        else:
            raise ValueError(f"Invalid patch line prefix {prefix!r}.")
        change_count += 1
        index += 1

    if change_count == 0:
        raise ValueError("Patch chunk has no change lines.")

    return (
        PatchChunk(
            change_context=change_context,
            old_lines=old_lines,
            new_lines=new_lines,
            is_end_of_file=is_end_of_file,
            removed_lines=removed_lines,
            inserted_lines=inserted_lines,
        ),
        index,
    )


def _is_chunk_boundary(line: str) -> bool:
    """Return whether a line starts the next chunk or file-level patch marker."""
    stripped = line.strip()
    return (
        stripped.startswith("*** ")
        or line == EMPTY_CHANGE_CONTEXT_MARKER
        or line.startswith(CHANGE_CONTEXT_MARKER)
    )


def _validate_single_file_patch(
    file_patch: FilePatch,
    *,
    target_path: str | None,
) -> FilePatch:
    """Ensure the parsed patch targets the expected file and uses supported features."""
    if target_path is not None:
        expected_path = target_path.lstrip("/")
        actual_path = file_patch.path.lstrip("/")
        if actual_path != expected_path:
            raise ValueError(
                f"Patch targets {file_patch.path!r}, expected {target_path!r}."
            )
    if file_patch.move_path is not None:
        raise ValueError("Move operations are not supported.")
    return file_patch


def _collect_patch_stats(file_patches: list[FilePatch]) -> PatchStats:
    """Aggregate chunk and line-change totals across parsed file patches."""
    return PatchStats(
        chunk_count=sum(len(file_patch.chunks) for file_patch in file_patches),
        lines_removed=sum(
            chunk.removed_lines
            for file_patch in file_patches
            for chunk in file_patch.chunks
        ),
        lines_inserted=sum(
            chunk.inserted_lines
            for file_patch in file_patches
            for chunk in file_patch.chunks
        ),
    )


def apply_patch_chunks_to_text(
    *,
    original_text: str,
    file_path: str,
    chunks: list[PatchChunk],
) -> str:
    """Apply parsed patch chunks to file text while preserving trailing-newline state."""
    has_trailing_newline = original_text.endswith("\n")
    original_lines = original_text.split("\n")
    if original_lines and original_lines[-1] == "":
        original_lines.pop()

    replacements = _find_replacements(original_lines, file_path, chunks)
    new_lines = _apply_replacements(original_lines, replacements)
    if not new_lines:
        return ""
    return "\n".join(new_lines) + ("\n" if has_trailing_newline else "")


def _find_replacements(
    original_lines: list[str],
    file_path: str,
    chunks: list[PatchChunk],
) -> list[tuple[int, int, list[str]]]:
    """Resolve each patch chunk to a concrete list replacement against the original lines."""
    replacements: list[tuple[int, int, list[str]]] = []
    line_index = 0

    for chunk in chunks:
        if chunk.change_context is not None:
            context_index = _seek_sequence(
                lines=original_lines,
                pattern=[chunk.change_context],
                start=line_index,
                eof=False,
            )
            if context_index is None:
                raise ValueError(
                    f"Failed to find context {chunk.change_context!r} in {file_path}."
                )
            line_index = context_index + 1

        if not chunk.old_lines:
            insertion_index = len(original_lines)
            replacements.append((insertion_index, 0, chunk.new_lines))
            continue

        old_lines = chunk.old_lines
        replacement_lines = chunk.new_lines
        found = _seek_sequence(
            lines=original_lines,
            pattern=old_lines,
            start=line_index,
            eof=chunk.is_end_of_file,
        )
        if found is None and old_lines and old_lines[-1] == "":
            old_lines = old_lines[:-1]
            if replacement_lines and replacement_lines[-1] == "":
                replacement_lines = replacement_lines[:-1]
            found = _seek_sequence(
                lines=original_lines,
                pattern=old_lines,
                start=line_index,
                eof=chunk.is_end_of_file,
            )
        if found is None:
            joined = "\n".join(chunk.old_lines)
            raise ValueError(f"Failed to find expected lines in {file_path}:\n{joined}")

        replacements.append((found, len(old_lines), replacement_lines))
        line_index = found + len(old_lines)

    return sorted(replacements, key=lambda item: item[0])


def _apply_replacements(
    lines: list[str],
    replacements: list[tuple[int, int, list[str]]],
) -> list[str]:
    """Apply resolved replacements from bottom to top so earlier offsets stay stable."""
    result = list(lines)
    for start_index, old_len, new_lines in reversed(replacements):
        del result[start_index : start_index + old_len]
        result[start_index:start_index] = new_lines
    return result


def _seek_sequence(
    *,
    lines: list[str],
    pattern: list[str],
    start: int,
    eof: bool,
) -> int | None:
    """Find the next matching line sequence using progressively looser normalizers."""
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None

    max_start = len(lines) - len(pattern)
    search_start = max_start if eof and len(lines) >= len(pattern) else start
    if search_start > max_start:
        return None

    for normalize in LINE_NORMALIZERS:
        normalized_pattern = [normalize(expected) for expected in pattern]
        normalized_lines: list[str | None] = [None] * len(lines)
        for index in range(search_start, max_start + 1):
            if _lines_match(
                lines,
                normalized_pattern,
                start=index,
                normalize=normalize,
                normalized_lines=normalized_lines,
            ):
                return index
    return None


def _lines_match(
    lines: list[str],
    normalized_pattern: list[str],
    *,
    start: int,
    normalize: LineNormalizer,
    normalized_lines: list[str | None],
) -> bool:
    """Check whether normalized lines match the expected pattern at one start index."""
    for offset, expected in enumerate(normalized_pattern):
        line_index = start + offset
        actual = normalized_lines[line_index]
        if actual is None:
            actual = normalize(lines[line_index])
            normalized_lines[line_index] = actual
        if actual != expected:
            return False
    return True


def _normalize_punctuation(value: str) -> str:
    """Replace smart punctuation and unusual spaces with ASCII equivalents."""
    return "".join(PUNCTUATION_TRANSLATION.get(char, char) for char in value)


def _normalize_whitespace(value: str) -> str:
    """Collapse internal whitespace runs to single spaces for fuzzy matching."""
    return " ".join(value.split())


LINE_NORMALIZERS: tuple[LineNormalizer, ...] = (
    lambda value: value,
    lambda value: value.rstrip(),
    lambda value: value.strip(),
    lambda value: _normalize_whitespace(value.strip()),
    lambda value: _normalize_punctuation(value.strip()),
    lambda value: _normalize_whitespace(_normalize_punctuation(value.strip())),
)
