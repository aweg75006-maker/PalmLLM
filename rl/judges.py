from dataclasses import dataclass


@dataclass
class JudgeOutput:
    score: float
    reason: str = ""


class HeuristicJudge:
    """
    Offline, dependency-free judge for RLAIF scaffolding.
    Produces a scalar score; higher is better.
    """

    def __init__(self, min_len: int = 8, max_len: int = 256, repetition_penalty: float = 0.2) -> None:
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.repetition_penalty = float(repetition_penalty)

    def score(self, prompt: str, answer: str) -> JudgeOutput:
        ans = answer.strip()
        if not ans:
            return JudgeOutput(score=-1e9, reason="empty")

        n = len(ans)
        # length shaping
        if n < self.min_len:
            base = -1.0
        elif n > self.max_len:
            base = -0.5
        else:
            base = 1.0

        # simple repetition penalty: unique char ratio
        uniq = len(set(ans))
        rep = 1.0 - (uniq / max(1, n))
        score = base - self.repetition_penalty * rep
        return JudgeOutput(score=float(score), reason=f"len={n} rep={rep:.3f}")


class PairwiseJudge:
    def pick(self, a: JudgeOutput, b: JudgeOutput) -> int:
        """
        Return 0 if a better, 1 if b better.
        """
        return 0 if a.score >= b.score else 1

