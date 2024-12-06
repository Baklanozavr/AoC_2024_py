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
    first_number, rest_of_candidate = pop_number(candidate, ",")
    second_number, _ = pop_number(rest_of_candidate, ")")
    return first_number * second_number


def sum_mul_candidates(candidates_list: list[str]) -> int:
    return sum([mul_candidate(candidate) for candidate in candidates_list])


def day_03_1(input_list: list[str]) -> int:
    """ mul(X,Y), where X and Y are each 1-3 digit numbers """
    return sum([sum_mul_candidates(line.split("mul(")) for line in input_list])


def day_03_2(input_list: list[str]) -> int:
    """ mul(X,Y), where X and Y are each 1-3 digit numbers + do() and don't()"""
    return day_03_1([do_candidate.split("don't()")[0] for do_candidate in "".join(input_list).split("do()")])


def count_xmas_in_line(line: str) -> int:
    return len(line.split("XMAS")) - 1


def count_xmas(matrix: list[str]) -> int:
    return sum([count_xmas_in_line(line) + count_xmas_in_line(line[::-1]) for line in matrix])


def transpose(matrix: list[str]) -> list[str]:
    transposed = ['' for _ in range(len(matrix))]
    for line in matrix:
        for i, item in enumerate(line):
            transposed[i] += item
    return transposed


def reverse_lines(matrix: list[str]) -> list[str]:
    return [line[::-1] for line in matrix]


def get_minor_diagonals(matrix: list[str]) -> list[str]:
    diagonals = ['' for _ in range(2 * len(matrix) - 1)]
    for i, line in enumerate(matrix):
        for j, item in enumerate(line):
            diagonals[i + j] += item
    return diagonals


def get_main_diagonals(matrix: list[str]) -> list[str]:
    return get_minor_diagonals(reverse_lines(matrix))


def get_matrix_from_minor_diagonals(diagonals: list[str]) -> list[str]:
    N = len(diagonals) // 2 + 1
    matrix = ['' for _ in range(N)]
    for i, line in enumerate(diagonals):
        shift = 0 if i < N else i % N + 1
        for j, item in enumerate(line):
            matrix[shift + j] += item
    return matrix


def get_matrix_from_main_diagonals(diagonals: list[str]) -> list[str]:
    return reverse_lines(get_matrix_from_minor_diagonals(diagonals))


def mark_line(line: str) -> str:
    """ mark: MAS or SAM, encodes all characters to 0, and A in mark as 1 """

    def mark(sam_or_mas: str) -> str:
        return "010".join(['0' * len(shard) for shard in line.split(sam_or_mas)])

    marked_line = ''
    for sam, mas in zip(mark("SAM"), mark("MAS")):
        marked_line += '1' if sam == '1' or mas == '1' else '0'
    return marked_line


def day_04_1(input_list: list[str]) -> int:
    return (count_xmas(input_list)
            + count_xmas(transpose(input_list))
            + count_xmas(get_main_diagonals(input_list))
            + count_xmas(get_minor_diagonals(input_list)))


def day_04_2(input_list: list[str]) -> int:
    main = get_matrix_from_main_diagonals([mark_line(d) for d in get_main_diagonals(input_list)])
    minor = get_matrix_from_minor_diagonals([mark_line(d) for d in get_minor_diagonals(input_list)])
    intersections = 0
    for main_line, minor_line in zip(main, minor):
        for x, y in zip(main_line, minor_line):
            intersections += 1 if x == '1' and y == '1' else 0
    return intersections


def extract_rules_and_updates(input_list: list[str]) -> tuple[dict[int, set[int]], list[list[int]]]:
    rules = {}
    updates = []
    read_rules = True
    for line in input_list:
        if not line:
            read_rules = False
        elif read_rules:
            l, r = line.split('|')
            left, right = int(l), int(r)
            rules_set = rules.get(left, set())
            rules_set.add(right)
            rules[left] = rules_set
        else:
            updates.append([int(x) for x in line.split(',')])
    return rules, updates


def check_update(rules: dict[int, set[int]], update: list[int]) -> int:
    """ Returns the middle page number from a correctly-ordered update or 0 """
    for i, page in enumerate(update):
        page_rules = rules.get(page, set())
        if page_rules.intersection(update[:i + 1]):
            return 0
    return update[len(update) // 2]


def fix_update(rules: dict[int, set[int]], update: list[int]) -> int:
    """ Returns the middle page number from a correctly-ordered update """
    fixed_update = []
    for page in update:
        page_rules = rules.get(page, set())
        if page_rules.intersection(fixed_update):
            for i, anchor in enumerate(fixed_update):
                if anchor in page_rules:
                    fixed_update.insert(i, page)
                    break
        else:
            fixed_update.append(page)
    return fixed_update[len(update) // 2]


def day_05_1(input_list: list[str]) -> int:
    rules, updates = extract_rules_and_updates(input_list)
    return sum([check_update(rules, update) for update in updates])


def day_05_2(input_list: list[str]) -> int:
    rules, updates = extract_rules_and_updates(input_list)
    updates_to_fix = []
    for update in updates:
        if not check_update(rules, update):
            updates_to_fix.append(update)
    return sum([fix_update(rules, update) for update in updates_to_fix])


def find_guard_position(input_list: list[str]) -> tuple[int, int]:
    for n, line in enumerate(input_list):
        for i, symbol in enumerate(line):
            if symbol == '^':
                return n, i
    return -1, -1


def get_next_position(position: tuple[int, int], direction: str) -> tuple[int, int]:
    n, i = position
    if direction == 'UP':
        return n - 1, i
    elif direction == 'RIGHT':
        return n, i + 1
    elif direction == 'DOWN':
        return n + 1, i
    elif direction == 'LEFT':
        return n, i - 1
    else:
        raise Exception("Unexpected direction: {}", direction)


def get_next_direction(direction: str) -> str:
    return {
        'UP': 'RIGHT',
        'RIGHT': 'DOWN',
        'DOWN': 'LEFT',
        'LEFT': 'UP'
    }.get(direction)


def get_cell(area: list[str], position: tuple[int, int]) -> str:
    n, i = position
    return area[n][i] if 0 <= n < len(area) and 0 <= i < len(area) else ''


class GuardState:
    def __init__(self, position: tuple[int, int], direction='UP'):
        self.position = position
        self.direction = direction

    def __eq__(self, other):
        if isinstance(other, GuardState):
            return (self.position == other.position) and (self.direction == other.direction)
        else:
            return False

    def __hash__(self):
        return hash(self.position) + hash(self.direction)


def move(state: GuardState, area: list[str]) -> GuardState:
    next_position = get_next_position(state.position, state.direction)
    next_cell = get_cell(area, next_position)
    if not next_cell:
        return GuardState((-1, -1))
    if next_cell == '#' or next_cell == 'O':
        return GuardState(state.position, get_next_direction(state.direction))
    return GuardState(next_position, state.direction)


def copy_area(area: list[str], obstacle_position: tuple[int, int]) -> list[str]:
    copied_area = []
    for n, line in enumerate(area):
        new_line = ''
        for i, symbol in enumerate(line):
            new_line += 'O' if (n, i) == obstacle_position else symbol
        copied_area.append(new_line)
    return copied_area


def collect_guard_path(area: list[str], initial_state: GuardState) -> tuple[list[GuardState], bool]:
    state_history = []
    states_set = set()
    guard_state = initial_state
    while guard_state.position[0] >= 0 and guard_state.position[1] >= 0:
        state_history.append(guard_state)
        states_set.add(guard_state)
        guard_state = move(guard_state, area)
        if guard_state in states_set:
            return state_history, True
    return state_history, False


def is_area_with_cycle(area: list[str], initial_state: GuardState) -> bool:
    guard_state = initial_state
    while guard_state.position[0] >= 0 and guard_state.position[1] >= 0:
        guard_state = move(guard_state, area)
    return guard_state.position[0] == -10


def day_06_1(input_list: list[str]) -> int:
    guard_state = GuardState(find_guard_position(input_list))
    path, _ = collect_guard_path(input_list, guard_state)
    return len(set(state.position for state in path))


def day_06_2(input_list: list[str]) -> int:
    guard_state = GuardState(find_guard_position(input_list))
    state_history, _ = collect_guard_path(input_list, guard_state)
    checked_positions = set()
    cycling_obstacles = set()
    prev_state = state_history[0]
    for state in state_history[1::]:  # ignore initial position
        test_guard_state = prev_state
        prev_state = state
        if state.position in checked_positions:
            continue
        checked_positions.add(state.position)
        test_area = copy_area(input_list, state.position)
        _, has_cycle = collect_guard_path(test_area, test_guard_state)
        if has_cycle:
            cycling_obstacles.add(state.position)
            print(state.position)
    return len(cycling_obstacles)


if __name__ == '__main__':
    lines = list_lines_from_file("input/Day06.txt")
    day_function = day_06_2
    print(day_function(lines))
