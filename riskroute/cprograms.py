"""Generates short C programs whose output a student is asked to predict.

Each family targets a classic C misconception (integer division, post-increment,
switch fall-through, pass-by-value, ...). Programs are deterministic, free of undefined
behaviour and print a short output. The gold output is never written by hand: it comes
from actually compiling and running the program (riskroute.verifiers.run_c).
"""
import random

HEADER = "#include <stdio.h>\n"


def _main(body: str, extra_include: str = "", before_main: str = "") -> str:
    return (HEADER + extra_include + before_main
            + "int main(void) {\n" + body + "    return 0;\n}\n")


def int_division(r):
    a, b = r.randint(-30, 60), r.randint(2, 9)
    return _main(f"    int a = {a}, b = {b};\n"
                 f"    printf(\"%d %d\\n\", a / b, a % b);\n")


def precedence(r):
    a, b, c, d, e = (r.randint(1, 9) for _ in range(5))
    return _main(f"    int x = {a} + {b} * {c} - {d * e} / {e};\n"
                 f"    printf(\"%d\\n\", x);\n")


def increments(r):
    a = r.randint(0, 9)
    return _main(f"    int i = {a};\n    int j = i++;\n    int k = ++i;\n"
                 f"    printf(\"%d %d %d\\n\", i, j, k);\n")


def loop_sum(r):
    s, e, st = r.randint(0, 5), r.randint(8, 20), r.randint(1, 4)
    return _main(f"    int sum = 0;\n    for (int i = {s}; i < {e}; i += {st})\n"
                 f"        sum += i;\n    printf(\"%d\\n\", sum);\n")


def break_continue(r):
    n, skip, stop = r.randint(8, 15), r.randint(2, 4), r.randint(5, 7)
    return _main(f"    int count = 0;\n    for (int i = 1; i <= {n}; i++) {{\n"
                 f"        if (i % {skip} == 0) continue;\n"
                 f"        if (i > {stop}) break;\n        count += i;\n    }}\n"
                 f"    printf(\"%d\\n\", count);\n")


def pointer_arith(r):
    vals = [r.randint(1, 50) for _ in range(5)]
    k = r.randint(0, 3)
    return _main(f"    int a[] = {{{', '.join(map(str, vals))}}};\n    int *p = a + {k};\n"
                 f"    printf(\"%d %d\\n\", *p, *(p + 1));\n")


def char_arith(r):
    k = r.randint(0, 25)
    return _main(f"    char c = 'A' + {k};\n    printf(\"%c %d\\n\", c, c);\n")


def ternary_abs(r):
    a, b = r.randint(-10, 30), r.randint(-10, 30)
    return _main(f"    int x = {a}, y = {b};\n    int d = (x > y) ? x - y : y - x;\n"
                 f"    printf(\"%d\\n\", d);\n")


def switch_fallthrough(r):
    k = r.randint(1, 4)
    return _main(f"    int k = {k}, c = 0;\n    switch (k) {{\n"
                 f"        case 1: c += 1;\n        case 2: c += 2; break;\n"
                 f"        case 3: c += 3;\n        default: c += 10;\n    }}\n"
                 f"    printf(\"%d\\n\", c);\n")


def recursion(r):
    if r.random() < 0.5:
        n = r.randint(3, 8)
        fn = "int fact(int n) {\n    return n <= 1 ? 1 : n * fact(n - 1);\n}\n"
        return _main(f"    printf(\"%d\\n\", fact({n}));\n", before_main=fn)
    n = r.randint(5, 12)
    fn = "int fib(int n) {\n    return n < 2 ? n : fib(n - 1) + fib(n - 2);\n}\n"
    return _main(f"    printf(\"%d\\n\", fib({n}));\n", before_main=fn)


def bitwise(r):
    a, b = r.randint(1, 31), r.randint(1, 31)
    return _main(f"    int a = {a}, b = {b};\n"
                 f"    printf(\"%d %d %d %d\\n\", a & b, a | b, a ^ b, a << 2);\n")


def strings(r):
    word = r.choice(["pointer", "compiler", "student", "bengaluru", "variable", "function"])
    k = r.randint(1, 4)
    return _main(f"    char s[] = \"{word}\";\n"
                 f"    printf(\"%zu %s\\n\", strlen(s), s + {k});\n",
                 extra_include="#include <string.h>\n")


def float_pitfall(r):
    a, b = r.randint(5, 30), r.randint(2, 7)
    return _main(f"    int a = {a}, b = {b};\n    float f = a / b;\n    float g = (float)a / b;\n"
                 f"    printf(\"%.2f %.2f\\n\", f, g);\n")


def nested_pattern(r):
    n = r.randint(3, 4)
    return _main(f"    for (int i = 1; i <= {n}; i++) {{\n"
                 f"        for (int j = 1; j <= i; j++)\n            printf(\"*\");\n"
                 f"        printf(\"\\n\");\n    }}\n")


def static_var(r):
    n = r.randint(2, 5)
    fn = "int counter(void) {\n    static int c = 0;\n    c++;\n    return c;\n}\n"
    calls = "".join("    counter();\n" for _ in range(n - 1))
    return _main(calls + "    printf(\"%d\\n\", counter());\n", before_main=fn)


def swap_by_value(r):
    a, b = r.randint(1, 9), r.randint(10, 99)
    if r.random() < 0.5:
        fn = "void swap(int x, int y) {\n    int t = x; x = y; y = t;\n}\n"
        call = "swap(a, b)"
    else:
        fn = "void swap(int *x, int *y) {\n    int t = *x; *x = *y; *y = t;\n}\n"
        call = "swap(&a, &b)"
    return _main(f"    int a = {a}, b = {b};\n    {call};\n    printf(\"%d %d\\n\", a, b);\n",
                 before_main=fn)


FAMILIES = [int_division, precedence, increments, loop_sum, break_continue, pointer_arith,
            char_arith, ternary_abs, switch_fallthrough, recursion, bitwise, strings,
            float_pitfall, nested_pattern, static_var, swap_by_value]

QUESTION = "What is the exact output of this C program?\n\n```c\n{code}```"


def generate(n: int = 250, seed: int = 7):
    """Returns a list of (family_name, code) with no duplicate programs."""
    r = random.Random(seed)
    out, seen = [], set()
    i = 0
    while len(out) < n and i < n * 50:
        fam = FAMILIES[i % len(FAMILIES)]
        code = fam(r)
        i += 1
        if code not in seen:
            seen.add(code)
            out.append((fam.__name__, code))
    return out


def extract_code(query: str):
    """Pulls the C program back out of a question, or None."""
    if "```c" not in query:
        return None
    return query.split("```c", 1)[1].split("```", 1)[0].lstrip("\n")
