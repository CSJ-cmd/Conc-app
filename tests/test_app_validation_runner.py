from structural_safety_core_v6 import run_validation_tests


def test_embedded_validation_runner_cases_all_pass():
    results = run_validation_tests()
    failed = [(name, detail) for name, passed, detail in results if not passed]
    assert not failed
    assert [name for name, _, _ in results] == [
        "TC0(입력 파서)",
        "TC1(엑셀 일치)",
        "TC2(기각 2개, 무효X)",
        "TC3(기각 5개, 무효)",
        "TC4(Ct 배율)",
        "TC5(입력값 검증)",
        "TC6(공식 선택 UX)",
        "TC7(측정점수 정책)",
    ]
