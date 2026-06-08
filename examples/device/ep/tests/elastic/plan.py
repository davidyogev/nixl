# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json


# Plan-file formats accepted by this loader:
#
#   1. LEGACY (list-of-lists, single signed-integer encoding)
#
#        [
#          [0, 1, 2, 3],
#          [0, 1, -2, 3]
#        ]
#
#      Each phase is a list of signed ints. A positive (or zero, in
#      principle) value means "rank is present in this phase". A negative
#      value `-N` means "rank N is present in this phase AND killed during
#      it" (the victim's worker process still executes the phase up to the
#      kill point). The cardinal limitation of this encoding is that
#      `-0 == 0`, so rank 0 cannot be expressed as a victim.
#
#   2. EXPLICIT (list-of-dicts, separate `ranks` and `killed` arrays)
#
#        [
#          {"ranks": [0, 1, 2, 3]},
#          {"ranks": [0, 1, 2, 3], "killed": [0, 2]}
#        ]
#
#      `ranks` is the full participant list for the phase (everyone whose
#      worker is scheduled to execute, including soon-to-be-victims).
#      `killed` is the subset of `ranks` being killed during this phase;
#      it is optional (defaults to []). This encoding represents rank 0 as
#      a victim naturally.
#
# The two formats can be MIXED within a single plan file (auto-detected per
# phase by checking whether the entry is a list or a dict). Loading
# normalises every phase into a uniform internal form:
#
#     self._phases = [
#         {"all_ranks": [...], "victims": [...]},  # per phase
#         ...
#     ]
#
# Every method in this class reads from `_phases`, so legacy and explicit
# plans are observationally identical once loaded.


def _normalize_phase(raw_phase, phase_index: int):
    """Convert a single raw phase entry (legacy list-of-ints or explicit
    dict) into the uniform `{"all_ranks": [...], "victims": [...]}` form.
    """
    if isinstance(raw_phase, list):
        # Legacy: -N is a victim, all other non-negative ints are alive.
        all_ranks = [abs(r) for r in raw_phase]
        victims = [abs(r) for r in raw_phase if r < 0]
        return {"all_ranks": all_ranks, "victims": victims}
    if isinstance(raw_phase, dict):
        if "ranks" not in raw_phase:
            raise ValueError(
                f"plan phase {phase_index}: dict-form phase missing required "
                f"`ranks` key (got keys: {sorted(raw_phase.keys())})"
            )
        all_ranks = list(raw_phase["ranks"])
        victims = list(raw_phase.get("killed", []))
        # Reject negative entries on the explicit path -- the explicit form
        # exists precisely so the user does not need to use signed ints.
        for r in all_ranks:
            if r < 0:
                raise ValueError(
                    f"plan phase {phase_index}: dict-form `ranks` must be "
                    f"non-negative integers (got {r}). Use the `killed` key "
                    f"to mark victims; do not encode them as negatives."
                )
        for r in victims:
            if r < 0:
                raise ValueError(
                    f"plan phase {phase_index}: dict-form `killed` must be "
                    f"non-negative integers (got {r})."
                )
        # Every victim must also be a participant of the phase; this catches
        # typos like {"ranks": [0,1,2,3], "killed": [5]} where the killed
        # rank isn't even in the phase.
        extras = set(victims) - set(all_ranks)
        if extras:
            raise ValueError(
                f"plan phase {phase_index}: `killed` ranks {sorted(extras)} "
                f"are not in `ranks` {sorted(all_ranks)}; every killed rank "
                f"must participate in the phase before dying"
            )
        return {"all_ranks": all_ranks, "victims": victims}
    raise ValueError(
        f"plan phase {phase_index}: each phase must be either a list of "
        f"signed ints (legacy format) or a dict with `ranks` and optional "
        f"`killed` (explicit format); got {type(raw_phase).__name__}"
    )


class Plan:
    def __init__(self, plan_path: str, rank: int, start_phase: int = 0):
        """Initialize plan for a specific rank."""
        with open(plan_path, "r") as f:
            raw_phases = json.load(f)
        if not isinstance(raw_phases, list):
            raise ValueError(
                f"plan file {plan_path} must be a JSON array at the top "
                f"level; got {type(raw_phases).__name__}"
            )
        # Normalise every phase into the uniform internal representation.
        # Mixed legacy/explicit phases within one file are permitted.
        self._phases = [_normalize_phase(p, i) for i, p in enumerate(raw_phases)]
        self.rank = rank
        # Auto-detect starting phase for this rank
        self.current_phase = self._find_starting_phase(start_phase)
        self.starting_phase = self.current_phase  # Store the starting phase

    # -- internal helpers ----------------------------------------------------
    def _phase_all_ranks(self, idx: int):
        return self._phases[idx]["all_ranks"]

    def _phase_victims(self, idx: int):
        return self._phases[idx]["victims"]

    def _phase_survivors(self, idx: int):
        victims = set(self._phases[idx]["victims"])
        return [r for r in self._phases[idx]["all_ranks"] if r not in victims]

    # -- public API (unchanged signatures) -----------------------------------
    def _find_starting_phase(self, start_search_from_phase: int) -> int:
        """Find the first phase where this rank appears as a SURVIVOR after
        `start_search_from_phase`.

        Matches the legacy semantics: in the old list-of-ints form, a victim
        was encoded as `-N`, so `self.rank in self.phases[i]` returned False
        for the victim's kill phase. A rank's starting phase was therefore
        the first phase where it appeared positively (i.e. as a survivor of
        that phase). The explicit form preserves this: a rank that only
        appears as a victim has no starting phase.
        """
        for i in range(start_search_from_phase, len(self._phases)):
            if self.rank in self._phase_survivors(i):
                return i
        return -1

    def get_new_ranks(self):
        """Get ranks to connect to at current phase (only the new ones).

        Legacy semantic that we MUST preserve: a rank that died in the
        previous phase and is re-introduced alive in the current phase
        counts as "new" -- the survivors need to reconnect to it. The old
        list-of-ints code achieved this incidentally by doing set arithmetic
        on signed ints (so `-6` in prev was a different element than `6` in
        curr). We replicate the same semantic explicitly by computing the
        new arrivals as `curr_all - prev_SURVIVORS`, where prev_survivors
        excludes anyone who died in the previous phase.
        """
        curr_survivors = self._phase_survivors(self.current_phase)
        if self.current_phase == self.starting_phase:
            # First phase for this rank: connect to every other survivor of
            # this phase.
            return [r for r in curr_survivors if r != self.rank]
        prev_survivors = set(self._phase_survivors(self.current_phase - 1))
        curr_all = set(self._phase_all_ranks(self.current_phase))
        new_ranks = curr_all - prev_survivors
        # Connect only to NEW survivors (don't connect to a rank that's
        # arriving and dying in the same phase).
        return [r for r in new_ranks if r in curr_survivors and r != self.rank]

    def get_removed_ranks(self):
        """Get ranks to remove at current phase."""
        if self.current_phase == self.starting_phase:
            return []
        prev_all = set(self._phase_all_ranks(self.current_phase - 1))
        curr_all = set(self._phase_all_ranks(self.current_phase))
        curr_victims = set(self._phase_victims(self.current_phase))
        prev_victims = set(self._phase_victims(self.current_phase - 1))
        # Cleanly removed: in prev, not in curr, not killed in curr or prev.
        # A rank that died last phase isn't "cleanly removed" -- it died.
        cleanly_removed = list(prev_all - curr_all - curr_victims - prev_victims)
        return cleanly_removed

    def get_killed_ranks(self):
        """Get ranks that were killed at current phase."""
        return list(set(self._phase_victims(self.current_phase)))

    def get_active_ranks(self):
        """Get all active ranks in current phase (survivors AFTER the kill)."""
        return list(self._phase_survivors(self.current_phase))

    def get_max_rank(self):
        """Get the maximum participating rank index across all phases."""
        return max(max(p["all_ranks"]) for p in self._phases)

    def get_min_active_ranks(self):
        """Get the minimum number of active (= total participant) ranks in
        any phase. Mirrors the legacy implementation which used
        `len(phase)`; in the legacy list-of-ints form `len` counts both
        survivors and victims, so we use `len(all_ranks)` to match."""
        return min(len(p["all_ranks"]) for p in self._phases)

    def next(self):
        """Advance to next phase."""
        if self.current_phase < len(self._phases) - 1:
            self.current_phase += 1
            return True
        return False

    def get_phase(self):
        """Get current phase index."""
        return self.current_phase
