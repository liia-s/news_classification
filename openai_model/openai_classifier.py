import openai
import tenacity
import math
from dataclasses import dataclass

@dataclass
class ClassificationResult:
    probability: float
    num_tokens: int
    token_0_logprob: float | None
    token_1_logprob: float | None


class OpenAIClassifier:
    def __init__(self, *, api_key, model):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    @tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(6))
    def _get_completion_with_retry(self, prompt: str, seed: int):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
            {
              "role": "system",
              "content": "You are a helpful classification assistant. Your task is to read the given news text in Russian and provide the classification label ('0' or '1')."
            },
            {
              "role": "user",
              "content": "News text: " + prompt
            }],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
            seed=seed
        )
        return response

    def _compute_positive_class_probability(self, log_p_positive, log_p_negative):
        max_log_p = max(log_p_positive, log_p_negative)
        denominator = max_log_p + math.log(
            math.exp(log_p_positive - max_log_p) + math.exp(log_p_negative - max_log_p)
        )
        log_prob_positive = log_p_positive - denominator
        prob_positive = math.exp(log_prob_positive)
        return prob_positive

    def classify(self, text, seed=None):
        result = self._get_completion_with_retry(text, seed)
        token_logprobs = {t.token: t.logprob for t in result.choices[0].logprobs.content[0].top_logprobs}
        probability = self._compute_positive_class_probability(token_logprobs.get('1') or 0.0, token_logprobs.get('0') or 0.0,)
        return ClassificationResult(
            probability=probability,
            num_tokens=result.usage.prompt_tokens,
            token_0_logprob=token_logprobs.get('0'),
            token_1_logprob=token_logprobs.get('1')
        )
