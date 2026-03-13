from __future__ import annotations

from typing import List


def split_thoughts(
        text: str,
        min_chars=100,
        max_chars=300,
) -> List[str]:
    """
    Logic:
      1) Split by '\n\n'
      2) Iterate thoughts left→right
         - if MIN_CHARS <= len(thought) <= MAX_CHARS: keep
         - if len(thought) < MIN_CHARS: merge into next
         - if len(thought) > MAX_CHARS: split by single '\n'
    Notes:
      - Strips whitespace
      - Drops empty chunks
      - If the last chunk is < MIN_CHARS, it stays as-is (no next chunk to merge into)
    """
    if not text:
        return []

    # 1) initial split on double newline
    chunks = [c.strip() for c in text.split("\n\n")]
    chunks = [c for c in chunks if c]  # drop empties

    out: List[str] = []
    i = 0

    def _force_split_chunk(s: str, limit: int) -> List[str]:
        """Fallback splitter to ensure progress when no newlines exist.
        Tries sentence split, then word-chunking, finally hard slicing.
        """
        s = s.strip()
        if not s:
            return []
        # Try sentence boundaries
        import re

        sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
        if len(sentences) > 1 and all(len(x) <= limit for x in sentences):
            return sentences
        # Word-chunk up to limit
        words = s.split()
        out_chunks: List[str] = []
        cur = []
        cur_len = 0
        for w in words:
            wlen = len(w) + (1 if cur else 0)
            if cur_len + wlen <= limit:
                cur.append(w)
                cur_len += wlen
            else:
                if cur:
                    out_chunks.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
        if cur:
            out_chunks.append(" ".join(cur))
        # If still empty (extreme case), hard slice to ensure progress
        if not out_chunks:
            return [s[:limit]] + ([s[limit:]] if len(s) > limit else [])
        return out_chunks

    # Safety cap to avoid pathological infinite loops
    max_iters = max(100, 10 * (len(chunks) or 1))
    iters = 0

    while i < len(chunks):
        iters += 1
        if iters > max_iters:
            # flush remaining chunks to out to guarantee termination
            out.extend([c.strip() for c in chunks[i:] if c.strip()])
            break

        cur = chunks[i].strip()
        if not cur:
            i += 1
            continue

        # 3) too big -> split by single newline (and re-process subchunks)
        if len(cur) > max_chars:
            sub = [s.strip() for s in cur.split("\n")]
            sub = [s for s in sub if s]
            # If split didn't reduce, force a safe split to avoid infinite loop
            if len(sub) == 1 and sub[0] == cur:
                sub = _force_split_chunk(cur, max_chars)
            # Ensure progress: if still no change, move item to output to break loop
            if not sub:
                out.append(cur)
                i += 1
                continue
            # Insert subchunks in place of current chunk and continue
            chunks = chunks[:i] + sub + chunks[i + 1:]
            continue

        # 2) too small -> merge into next (if possible)
        if len(cur) < min_chars:
            if i + 1 < len(chunks):
                nxt = chunks[i + 1].strip()
                # merge with a blank line separator (keeps structure)
                merged = (cur + "\n\n" + nxt).strip() if nxt else cur
                chunks[i + 1] = merged
                i += 1  # skip current; it got merged into next
                continue
            else:
                # last chunk and too small: keep as-is
                out.append(cur)
                i += 1
                continue

        # in range -> keep
        out.append(cur)
        i += 1
    new_out = [out[0]]
    keep_t = ''
    for t in out[1:]:
        if len(t) < min_chars:
            new_out[-1] += t
        else:
            new_out.append(t)

    return new_out

# Example:
# thoughts = split_thoughts(generated_answer, min_chars=120, max_chars=600)
# print(len(thoughts))
# for t in thoughts: print("----\n", t)
