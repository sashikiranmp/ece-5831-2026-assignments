from logic_gate import LogicGate


def main():
    gate = LogicGate()

    input_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Expected outputs follow the input order above.
    tests = [
        ("and_gate", gate.and_gate, [0, 0, 0, 1]),
        ("nand_gate", gate.nand_gate, [1, 1, 1, 0]),
        ("or_gate", gate.or_gate, [0, 1, 1, 1]),
        ("nor_gate", gate.nor_gate, [1, 0, 0, 0]),
        ("xor_gate", gate.xor_gate, [0, 1, 1, 0]),
    ]

    total_passed = 0

    for name, function, expected_outputs in tests:
        print(f"\n{name}")
        print("x1 x2 | Actual Expected | Result")

        for (x1, x2), expected in zip(input_pairs, expected_outputs):
            actual = function(x1, x2)
            status = "PASS" if actual == expected else "FAIL"

            print(
                f" {x1}  {x2} |"
                f"   {actual}       {expected}    | {status}"
            )

            if actual != expected:
                raise AssertionError(
                    f"{name}({x1}, {x2}): "
                    f"expected {expected}, got {actual}"
                )

            total_passed += 1

    print(f"\nAll {total_passed} tests passed!")


if __name__ == "__main__":
    main()