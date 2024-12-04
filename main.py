def list_lines_from_file(file_name: str) -> list[str]:
    with open(file_name) as file:
        return [line.rstrip() for line in file]


def to_pair_of_lists(input_list: list[str]) -> tuple[list[int], list[int]]:
    listA, listB = [], []
    for line in input_list:
        pair = line.split()
        listA.append(int(pair[0]))
        listB.append(int(pair[1]))
    return listA, listB


def day_01_1(input_list: list[str]) -> int:
    listA, listB = to_pair_of_lists(input_list)
    listA.sort()
    listB.sort()
    return sum([abs(a - b) for a, b in zip(listA, listB)])


def day_01_2(input_list: list[str]) -> int:
    listA, listB = to_pair_of_lists(input_list)
    countB = dict()
    for b in listB:
        countB[b] = countB.get(b, 0) + 1
    return sum([a * countB.get(a, 0) for a in listA])


def to_list_of_ints(line: str) -> list[int]:
    return [int(i) for i in line.split()]


def is_report_safe(report: list[int]) -> int:
    """
    Return 1 when a report is safe, and 0 when unsafe.

    A report only is safe if both of the following are true:
    - The levels are either all increasing or all decreasing.
    - Any two adjacent levels differ by at least one and at most three.
    """
    prev_item = report[0]
    signs = 0
    for i in range(1, len(report)):
        delta = report[i] - prev_item
        if abs(delta) < 1 or abs(delta) > 3:
            return 0
        signs += 1 if delta > 0 else -1
        if abs(signs) != i:
            return 0
        prev_item = report[i]
    return 1


def is_report_safe_tolerantly(report: list[int]) -> int:
    """
    Same as is_report_safe but tolerates one bad element
    """
    for i in range(0, len(report)):
        sub_report = report[:i] + report[i + 1:]
        if is_report_safe(sub_report):
            return 1
    return 0


def day_02_1(input_list: list[str]) -> int:
    """ How many reports are safe? """
    return sum([is_report_safe(to_list_of_ints(line)) for line in input_list])


def day_02_2(input_list: list[str]) -> int:
    """ How many reports are safe? """
    return sum([is_report_safe_tolerantly(to_list_of_ints(line)) for line in input_list])


def pop_number(parent_string: str, separator: str) -> tuple[int, str]:
    """ Return 1-3 digit number from the first part of the string before separator """
    possible_pair = parent_string.split(separator)
    if len(possible_pair) < 2:
        return 0, ""
    possible_number = possible_pair[0]
    if 1 <= len(possible_number) <= 3 and possible_number.isdigit():
        return int(possible_number), possible_pair[1]
    return 0, ""


def mul_candidate(candidate: str) -> int:
    """ Returns X*Y from a candidate string: "X,Y)", where X and Y are each 1-3 digit numbers """
    first_number, rest_of_candidate = pop_number(candidate, ",")
    second_number, _ = pop_number(rest_of_candidate, ")")
    return first_number * second_number


def sum_mul_candidates(candidates_list: list[str]) -> int:
    """ Candidate string: "X,Y)", where X and Y are each 1-3 digit numbers """
    return sum([mul_candidate(candidate) for candidate in candidates_list])


def day_03_1(input_list: list[str]) -> int:
    """ mul(X,Y), where X and Y are each 1-3 digit numbers """
    return sum([sum_mul_candidates(line.split("mul(")) for line in input_list])


def day_03_2(input_list: list[str]) -> int:
    """ mul(X,Y), where X and Y are each 1-3 digit numbers + do() and don't()"""
    return day_03_1([do_candidate.split("don't()")[0] for do_candidate in "".join(input_list).split("do()")])


if __name__ == '__main__':
    lines = list_lines_from_file("input/Day03.txt")
    day_function = day_03_2
    print(day_function(lines))
