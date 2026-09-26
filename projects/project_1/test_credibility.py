"""
test_credibility.py — contract tests for score_url().

    python test_credibility.py

These check the SHAPE of your output, not its quality. They must keep passing
however much you rewrite the internals — the app, the grader, and evaluate.py
all rely on this contract. Use `evaluate.py` to measure quality.

Deliverable 1 asks for "initial testing to validate input/output handling".
This file is that, and adding your own cases here is part of the deliverable.

No pytest required, deliberately — one less thing to install.
"""

from credibility import score_band, score_url

PASSED = 0
FAILED = 0


def check(condition: bool, description: str) -> None:
    """Assert-with-a-label so one failure doesn't stop the whole run."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {description}")
    else:
        FAILED += 1
        print(f"  FAIL  {description}")


print("\nContract: return shape")
result = score_url("https://www.nature.com/articles/example", use_llm=False)
check(isinstance(result, dict), "returns a dict")
check(set(result.keys()) == {"score", "explanation"}, "has exactly the keys 'score' and 'explanation'")
check(isinstance(result["score"], float), "score is a float")
check(isinstance(result["explanation"], str), "explanation is a str")
check(len(result["explanation"]) > 0, "explanation is not empty")

print("\nContract: score range")
for url in [
    "https://www.nature.com/x",
    "https://medium.com/@a/b",
    "http://unknown-site.xyz/page",
    "https://en.wikipedia.org/wiki/X",
]:
    score = score_url(url, use_llm=False)["score"]
    check(0.0 <= score <= 1.0, f"{url[:40]:<42} -> {score:.2f} is within [0, 1]")

print("\nContract: malformed input is handled, not raised")
for bad in ["", "   ", "not a url", "ftp://files.example.com/x", "javascript:alert(1)", "//example.com"]:
    try:
        bad_result = score_url(bad, use_llm=False)
        ok = isinstance(bad_result, dict) and 0.0 <= bad_result["score"] <= 1.0
        check(ok, f"{bad!r:<28} -> {bad_result['score']:.2f} (no exception)")
    except Exception as e:
        check(False, f"{bad!r:<28} raised {type(e).__name__}")

print("\nContract: determinism")
a = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)
b = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)
check(a == b, "same URL scored twice gives the same result")

print("\nSanity: ordering the baseline should already get right")
journal = score_url("https://www.nature.com/articles/x", use_llm=False)["score"]
blog = score_url("https://randomblog.blogspot.com/x", use_llm=False)["score"]
check(journal > blog, f"a journal ({journal:.2f}) outranks a personal blog ({blog:.2f})")

gov = score_url("https://www.census.gov/data", use_llm=False)["score"]
throwaway = score_url("http://whatever.xyz/page", use_llm=False)["score"]
check(gov > throwaway, f"a .gov source ({gov:.2f}) outranks a throwaway domain ({throwaway:.2f})")

print("\nContract: score_band")
check(score_band(0.9)[0] == "HIGH", "0.90 -> HIGH")
check(score_band(0.5)[0] == "MEDIUM", "0.50 -> MEDIUM")
check(score_band(0.1)[0] == "LOW", "0.10 -> LOW")

print("\nCustom tests: Improvement validation")
# Test 1: Preprints should score lower than peer-reviewed
preprint = score_url("https://www.biorxiv.org/content/10.1101/2024.01.01", use_llm=False)["score"]
published = score_url("https://www.nature.com/articles/nature12345", use_llm=False)["score"]
check(preprint < published, f"preprint ({preprint:.2f}) < published journal ({published:.2f})")

# Test 2: HTTPS should not penalize credibility
https_score = score_url("https://example.com/article", use_llm=False)["score"]
http_score = score_url("http://example.com/article", use_llm=False)["score"]
check(https_score >= http_score, f"HTTPS ({https_score:.2f}) >= HTTP ({http_score:.2f})")

# Test 3: Personal blogs should score lower than publishers
blog = score_url("https://myblog.blogspot.com/post", use_llm=False)["score"]
publisher = score_url("https://www.bbc.co.uk/news/article", use_llm=False)["score"]
check(blog < publisher, f"personal blog ({blog:.2f}) < BBC ({publisher:.2f})")

# Test 4: Government .int domain (WHO) should score high
who = score_url("https://www.who.int/news/alerts", use_llm=False)["score"]
check(who >= 0.80, f"WHO.int ({who:.2f}) is highly credible (>=0.80)")

# Test 5: Medical journal should score high
jama = score_url("https://www.jamanetwork.com/journals/jama", use_llm=False)["score"]
check(jama >= 0.85, f"JAMA medical journal ({jama:.2f}) is highly credible (>=0.85)")

print(f"\n{'=' * 60}")
print(f"  {PASSED} passed, {FAILED} failed")
print(f"{'=' * 60}\n")
raise SystemExit(1 if FAILED else 0)
