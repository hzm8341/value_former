"""Industrial Track: Task Phase, Contact State, and Mistake State labels.

Source: R&D plan V3.2, Section 2.3 and 2.5. These are categorical labels that
describe "what is currently happening" and are stored independently from the
continuous Physical Progress potential in industrial_progress.py -- Phase
must never be used to derive a numeric progress value.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class TaskPhase(str, Enum):
    S0_FREE_SPACE_APPROACH = "S0"
    S1_PRE_CONTACT_ALIGNMENT = "S1"
    S2_FIRST_OR_RENEWED_CONTACT = "S2"
    S3_CONTACT_LOCALIZATION_SEARCH = "S3"
    S4_INSERTION_ENGAGED = "S4"
    S5_INSERTION_DEPTH_INCREASING = "S5"
    S6_FULLY_SEATED = "S6"  # terminal, no allowed regression


PHASE_ALLOWS_REGRESSION = {
    TaskPhase.S0_FREE_SPACE_APPROACH: True,
    TaskPhase.S1_PRE_CONTACT_ALIGNMENT: True,
    TaskPhase.S2_FIRST_OR_RENEWED_CONTACT: True,
    TaskPhase.S3_CONTACT_LOCALIZATION_SEARCH: True,
    TaskPhase.S4_INSERTION_ENGAGED: True,
    TaskPhase.S5_INSERTION_DEPTH_INCREASING: True,
    TaskPhase.S6_FULLY_SEATED: False,
}


class ContactState(str, Enum):
    NO_CONTACT = "no-contact"
    LIGHT_CONTACT = "light-contact"
    CENTERED_CONTACT = "centered-contact"
    LEFT_CONTACT = "left-contact"
    RIGHT_CONTACT = "right-contact"
    FRONT_CONTACT = "front-contact"
    BACK_CONTACT = "back-contact"
    JAM = "jam"
    SLIP = "slip"
    INSERTION = "insertion"
    FULLY_SEATED = "fully-seated"


class MistakeState(str, Enum):
    NORMAL = "normal"
    RECOVERABLE_MISTAKE = "recoverable_mistake"
    TERMINAL_FAILURE = "terminal_failure"
    IGNORE_AMBIGUOUS = "ignore_ambiguous"


@dataclass(frozen=True)
class MistakeSegment:
    t_start: float
    t_end: float
    state: MistakeState


def build_mistake_segments(
    sample_times: Sequence[float],
    in_mistake: Sequence[bool],
    recovered: Sequence[bool],
) -> list[MistakeSegment]:
    """Turn a per-frame physical-failure trace into mistake segments.

    ``in_mistake[t]`` marks physical failure onset (jam / slip / misalignment
    beyond tolerance). ``recovered[t]`` marks the frame at which the episode
    re-enters a stable, effective insertion/search state. A segment that never
    recovers before the episode ends is TERMINAL_FAILURE; one that recovers is
    RECOVERABLE_MISTAKE, spanning onset to the recovery frame per Section 2.5.
    """
    if not (len(sample_times) == len(in_mistake) == len(recovered)):
        raise ValueError("sample_times, in_mistake, recovered must be same length")

    segments: list[MistakeSegment] = []
    onset_idx: int | None = None
    for i, bad in enumerate(in_mistake):
        if bad and onset_idx is None:
            onset_idx = i
        elif not bad and onset_idx is not None:
            segments.append(
                MistakeSegment(
                    t_start=sample_times[onset_idx],
                    t_end=sample_times[i],
                    state=MistakeState.RECOVERABLE_MISTAKE,
                )
            )
            onset_idx = None
        elif bad and onset_idx is not None and recovered[i]:
            segments.append(
                MistakeSegment(
                    t_start=sample_times[onset_idx],
                    t_end=sample_times[i],
                    state=MistakeState.RECOVERABLE_MISTAKE,
                )
            )
            onset_idx = i + 1 if i + 1 < len(sample_times) else None

    if onset_idx is not None:
        segments.append(
            MistakeSegment(
                t_start=sample_times[onset_idx],
                t_end=sample_times[-1],
                state=MistakeState.TERMINAL_FAILURE,
            )
        )
    return segments


def reaction_delay(t_intervention: float, t_physical_failure_onset: float) -> float:
    """reaction_delay = t_intervention - t_physical_failure_onset (Section 2.5)."""
    return t_intervention - t_physical_failure_onset
