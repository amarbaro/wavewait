# fixtures/saxpy_renamed

The ATT saxpy capture (`../saxpy_att`) with **one instruction renamed** in `pcmap.tsv`:

    pc 6516   s_waitcnt vmcnt(0)   ->   s_waitcnt vmcnt(1)

The trace itself is unchanged (`blob_0.bin` is a symlink to the ATT capture), so the diff
of `fixtures/saxpy_att` against this fixture shows exactly one removed row (`-0x1974`,
the row of A that has no counterpart in B) and one inserted row (`+0x1974`, the renamed
row of B). Every other row is aligned with a delta of 0.0.

Why a rename and not a deleted instruction: the decoder walks the trace pc by pc through
`pcmap.tsv`, so an instruction missing from the map breaks the pc chain instead of
dropping a row. Renaming one instruction is the cheapest way to exercise the removed path
of the diff.
