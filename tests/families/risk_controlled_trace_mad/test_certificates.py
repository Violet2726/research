from __future__ import annotations

from research_experiments.families.risk_controlled_trace_mad.certificates import verify_certificate


def test_arithmetic_certificate_passes_and_rejects_unbound_operand() -> None:
    passed = verify_certificate(question="Compute 2 + 3.", final_answer="5", certificate_type="arithmetic", payload={"expression": "2+3", "claimed_value": "5"})
    assert passed["status"] == "pass"
    unsupported = verify_certificate(question="Compute 2 + 3.", final_answer="105", certificate_type="arithmetic", payload={"expression": "2+3+100", "claimed_value": "105"})
    assert unsupported["status"] == "unsupported"
    mismatched = verify_certificate(question="Compute 2 + 3.", final_answer="7", certificate_type="arithmetic", payload={"expression": "2+3", "claimed_value": "5"})
    assert mismatched["status"] == "fail"


def test_symbolic_ordering_and_boolean_certificates() -> None:
    symbolic = verify_certificate(question="For real x compare x+x and 2*x.", final_answer="2*x", certificate_type="symbolic", payload={"left": "x+x", "right": "2*x", "substitutions": {}})
    assert symbolic["status"] == "pass"
    ordering = verify_certificate(question="Sort pear apple banana.", final_answer="apple banana pear", certificate_type="ordering", payload={"items": ["pear", "apple", "banana"], "ordered_items": ["apple", "banana", "pear"], "direction": "ascending"})
    assert ordering["status"] == "pass"
    boolean = verify_certificate(question="Evaluate True and False.", final_answer="False", certificate_type="boolean", payload={"expression": "True and False", "variables": {}, "claimed_value": False})
    assert boolean["status"] == "pass"


def test_self_consistent_payload_cannot_certify_an_unrelated_final_answer() -> None:
    symbolic = verify_certificate(question="For real x compare x+x and 2*x.", final_answer="3*x", certificate_type="symbolic", payload={"left": "x+x", "right": "2*x", "substitutions": {}})
    assert symbolic["status"] == "unsupported"
    ordering = verify_certificate(question="Sort pear apple banana.", final_answer="pear apple banana", certificate_type="ordering", payload={"items": ["pear", "apple", "banana"], "ordered_items": ["apple", "banana", "pear"], "direction": "ascending"})
    assert ordering["status"] == "fail"
    boolean = verify_certificate(question="Evaluate True and False.", final_answer="True", certificate_type="boolean", payload={"expression": "True and False", "variables": {}, "claimed_value": False})
    assert boolean["status"] == "fail"


def test_certificate_never_executes_calls_or_attributes() -> None:
    result = verify_certificate(question="Compute 1.", final_answer="1", certificate_type="arithmetic", payload={"expression": "__import__('os').system('x')", "claimed_value": "1"})
    assert result["status"] == "unsupported"
