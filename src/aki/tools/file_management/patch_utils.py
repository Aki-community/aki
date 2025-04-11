"""Patch utilities for file modifications."""

import math
import re
from difflib import SequenceMatcher


def apply_patch(content: str, patch_string: str) -> str:
    """Apply a diff string to content string.

    Args:
        content: Original content to patch
        patch_string: Patch content as string (search/replace block)

    Returns:
        str: Patched content

    Raises:
        ValueError: If patch parsing or application fails
    """
    if not patch_string:
        raise ValueError("Patch string is required")

    try:
        # Parse the patch string to get original and updated content pairs
        edit_blocks = parse_patch_string(patch_string)

        # Apply each edit block sequentially
        result = content
        for original, updated in edit_blocks:
            if not original.strip():
                # For empty original content, append the update without extra newline
                result = result.rstrip("\n") + updated.rstrip("\n")
            else:
                # Otherwise try to replace existing content
                new_result = replace_most_similar_chunk(result, original, updated)
                if new_result is None:
                    raise ValueError("Could not find matching content to replace")
                result = new_result

        return result

    except ValueError as e:
        # Re-raise any ValueError with more context
        raise ValueError(f"Failed to apply patch: {str(e)}")
    except Exception as e:
        # Convert any other exception to ValueError
        raise ValueError(f"Failed to apply patch: Unexpected error - {str(e)}")


def parse_patch_string(patch_string: str) -> list[tuple[str, str]]:
    """Parse a patch string into list of (original, updated) content pairs.

    The patch string should contain one or more blocks in the format:
    <<<<<<< SEARCH
    original content
    =======
    updated content
    >>>>>>> REPLACE
    """
    HEAD = r"^<{5,9} SEARCH\s*$"
    DIVIDER = r"^={5,9}\s*$"
    UPDATED = r"^>{5,9} REPLACE\s*$"

    lines = patch_string.splitlines(keepends=True)

    # Find all edit blocks
    edit_blocks = []
    i = 0
    while i < len(lines):
        # Find start of block
        while i < len(lines) and not re.match(HEAD, lines[i].strip()):
            i += 1
        if i >= len(lines):
            break

        start_idx = i
        i += 1

        # Find divider
        while i < len(lines) and not re.match(DIVIDER, lines[i].strip()):
            i += 1
        if i >= len(lines):
            raise ValueError("Invalid patch format - missing divider")

        div_idx = i
        i += 1

        # Find end
        while i < len(lines) and not re.match(UPDATED, lines[i].strip()):
            i += 1
        if i >= len(lines):
            raise ValueError("Invalid patch format - missing end marker")

        end_idx = i
        i += 1

        original = "".join(lines[start_idx + 1 : div_idx])
        updated = "".join(lines[div_idx + 1 : end_idx])
        edit_blocks.append((original, updated))

    if not edit_blocks:
        raise ValueError("Invalid patch format - no edit blocks found")

    return edit_blocks


def replace_most_similar_chunk(whole: str, part: str, replace: str) -> str:
    """Find the most similar chunk of content and replace it.

    Args:
        whole: The complete content to search in
        part: The content to search for
        replace: The content to replace with

    Returns:
        The modified content with the replacement applied, or None if no suitable match found
    """

    # Prepare the content
    def prep(content):
        return content, content.splitlines(keepends=True)

    whole, whole_lines = prep(whole)
    part, part_lines = prep(part)
    replace, replace_lines = prep(replace)

    # Try exact match first
    res = perfect_replace(whole_lines, part_lines, replace_lines)
    if res is not None:
        return res

    # Try matching ignoring leading whitespace
    res = replace_with_flexible_whitespace(whole_lines, part_lines, replace_lines)
    if res is not None:
        return res

    # Try fuzzy matching as last resort
    res = replace_closest_edit_distance(whole_lines, part, part_lines, replace_lines)
    if res:
        return res

    raise ValueError("Could not find a suitable match for the content to replace")


def perfect_replace(whole_lines, part_lines, replace_lines):
    """Try to find and replace an exact match."""
    part_len = len(part_lines)
    part_tup = tuple(part_lines)

    for i in range(len(whole_lines) - part_len + 1):
        chunk = tuple(whole_lines[i : i + part_len])
        if chunk == part_tup:
            return "".join(
                whole_lines[:i] + replace_lines + whole_lines[i + part_len :]
            )

    return None


def replace_with_flexible_whitespace(whole_lines, part_lines, replace_lines):
    """Try to match ignoring differences in leading whitespace."""

    def get_content_with_indent(line):
        content = line.lstrip()
        indent = line[: len(line) - len(content)]
        return content, indent

    part_len = len(part_lines)

    for i in range(len(whole_lines) - part_len + 1):
        chunk = whole_lines[i : i + part_len]

        # Check if lines match ignoring leading whitespace
        matches = True
        common_indent = None

        for chunk_line, part_line in zip(chunk, part_lines):
            chunk_content, chunk_indent = get_content_with_indent(chunk_line)
            part_content, _ = get_content_with_indent(part_line)

            if chunk_content != part_content:
                matches = False
                break

            if chunk_content:  # Only consider non-empty lines
                if common_indent is None:
                    common_indent = chunk_indent
                elif chunk_indent != common_indent:
                    matches = False
                    break

        if matches:
            # Apply the common indentation to replacement lines
            if common_indent is None:
                common_indent = ""

            adjusted_replace = []
            for line in replace_lines:
                content, _ = get_content_with_indent(line)
                adjusted_replace.append(common_indent + content if content else line)

            return "".join(
                whole_lines[:i] + adjusted_replace + whole_lines[i + part_len :]
            )

    return None


def replace_closest_edit_distance(whole_lines, part, part_lines, replace_lines):
    """Find and replace the closest matching chunk based on edit distance."""
    similarity_thresh = 0.8

    max_similarity = 0
    best_start = -1
    best_end = -1

    # Allow for some variance in chunk size
    scale = 0.1
    min_len = math.floor(len(part_lines) * (1 - scale))
    max_len = math.ceil(len(part_lines) * (1 + scale))

    for length in range(min_len, max_len + 1):
        for i in range(len(whole_lines) - length + 1):
            chunk = "".join(whole_lines[i : i + length])
            similarity = SequenceMatcher(None, chunk, part).ratio()

            if similarity > max_similarity:
                max_similarity = similarity
                best_start = i
                best_end = i + length

    if max_similarity < similarity_thresh:
        return None

    return "".join(whole_lines[:best_start] + replace_lines + whole_lines[best_end:])
