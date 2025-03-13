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


def find_symbol_position(symbol: str, input_list: list[str]) -> tuple[int, int]:
    for n, line in enumerate(input_list):
        for i, character in enumerate(line):
            if character == symbol:
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


def move_guard(state: GuardState, area: list[str]) -> GuardState:
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
        guard_state = move_guard(guard_state, area)
        if guard_state in states_set:
            return state_history, True
    return state_history, False


def is_area_with_cycle(area: list[str], initial_state: GuardState) -> bool:
    guard_state = initial_state
    while guard_state.position[0] >= 0 and guard_state.position[1] >= 0:
        guard_state = move_guard(guard_state, area)
    return guard_state.position[0] == -10


def day_06_1(input_list: list[str]) -> int:
    guard_state = GuardState(find_symbol_position('^', input_list))
    path, _ = collect_guard_path(input_list, guard_state)
    return len(set(state.position for state in path))


def day_06_2(input_list: list[str]) -> int:
    guard_state = GuardState(find_symbol_position('^', input_list))
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


def parse_test_data_07(line: str) -> tuple[int, list[int]]:
    str_answer, str_list = tuple(line.split(':'))
    return int(str_answer), [int(str_num.strip()) for str_num in str_list.strip().split(' ')]


def solve_equation_01(test_value: int, numbers: list[int]) -> bool:
    if not numbers:
        raise Exception("Unexpected empty numbers")
    test_number = numbers[-1]
    sub_product = test_value - test_number
    div_product = test_value // test_number if test_value % test_number == 0 else -1
    if len(numbers) == 1:
        return sub_product == 0 or div_product == 1
    return (sub_product > 0 and solve_equation_01(sub_product, numbers[:-1])
            or div_product > 0 and solve_equation_01(div_product, numbers[:-1]))


def un_concat(test_value: int, test_number: int) -> int:
    if test_value == test_number:
        return 0
    str_value, str_number = str(test_value), str(test_number)
    if str_value.endswith(str_number):
        return int(str_value.removesuffix(str_number))
    return -1


def solve_equation_02(test_value: int, numbers: list[int]) -> bool:
    if not numbers:
        raise Exception("Unexpected empty numbers")
    test_number = numbers[-1]
    split_product = un_concat(test_value, test_number)
    sub_product = test_value - test_number
    div_product = test_value // test_number if test_value % test_number == 0 else -1
    if len(numbers) == 1:
        return split_product == 0 or sub_product == 0 or div_product == 1
    return (split_product > 0 and solve_equation_02(split_product, numbers[:-1])
            or sub_product > 0 and solve_equation_02(sub_product, numbers[:-1])
            or div_product > 0 and solve_equation_02(div_product, numbers[:-1]))


def day_07_1(input_list: list[str]) -> int:
    answer = 0
    for line in input_list:
        test_value, numbers = parse_test_data_07(line)
        if solve_equation_01(test_value, numbers):
            answer += test_value
    return answer


def day_07_2(input_list: list[str]) -> int:
    answer = 0
    for line in input_list:
        test_value, numbers = parse_test_data_07(line)
        if solve_equation_02(test_value, numbers):
            answer += test_value
    return answer


def get_antennas_coordinates(input_list: list[str]) -> dict[str, set[tuple[int, int]]]:
    """ key - antenna symbol, value - a set of coordinates """
    antennas_map = {}
    for n, line in enumerate(input_list):
        for i, symbol in enumerate(line):
            if symbol != '.':
                coordinates = antennas_map.get(symbol, set())
                coordinates.add((n, i))
                antennas_map[symbol] = coordinates
    return antennas_map


def get_unique_pairs_set(coordinates: set[any]) -> set[tuple[any, any]]:
    combinations = set()
    for left in coordinates:
        for right in coordinates:
            if left != right and (right, left) not in combinations:
                combinations.add((left, right))
    return combinations


def is_in_area(position: tuple[int, int], area_size: int) -> bool:
    return 0 <= position[0] < area_size and 0 <= position[1] < area_size


def shift_forward(vector: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = vector
    delta_x, delta_y = b[0] - a[0], b[1] - a[1]
    return b, (b[0] + delta_x, b[1] + delta_y)


def shift_backward(vector: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = vector
    delta_x, delta_y = b[0] - a[0], b[1] - a[1]
    return (a[0] - delta_x, a[1] - delta_y), a


def get_antinodes_1(antennas_pair: tuple[tuple[int, int], tuple[int, int]], area_size: int) -> set[tuple[int, int]]:
    _, first_candidate = shift_forward(antennas_pair)
    second_candidate, _ = shift_backward(antennas_pair)
    return set(filter(lambda x_y: is_in_area(x_y, area_size), [first_candidate, second_candidate]))


def get_antinodes_2(antennas_pair: tuple[tuple[int, int], tuple[int, int]], area_size: int) -> set[tuple[int, int]]:
    antinodes = set()
    first, second = antennas_pair
    while is_in_area(second, area_size):
        antinodes.add(second)
        first, second = shift_forward((first, second))
    first, second = antennas_pair
    while is_in_area(first, area_size):
        antinodes.add(first)
        first, second = shift_backward((first, second))
    return antinodes


def day_08(input_list: list[str], get_antinodes) -> int:
    area_size = len(input_list)
    antennas_map = get_antennas_coordinates(input_list)
    antinodes = set()
    for coordinates in antennas_map.values():
        for pair in get_unique_pairs_set(coordinates):
            antinodes.update(get_antinodes(pair, area_size))
    return len(antinodes)


def day_08_1(input_list: list[str]) -> int:
    return day_08(input_list, get_antinodes_1)


def day_08_2(input_list: list[str]) -> int:
    return day_08(input_list, get_antinodes_2)


def block_checksum(block_size: int, block_index: int, block_position: int) -> int:
    return (block_index // 2) * sum([block_position + i for i in range(block_size)])


def day_09_01(input_list: list[str]) -> int:
    disk_line = [int(block) for block in input_list[0]]
    left, right = 0, len(disk_line) - 1
    position, checksum = 0, 0
    used_empty_space, relocated_right_count = 0, 0
    while left <= right:
        left_block = disk_line[left]
        if left % 2 == 0:
            file_block = left_block if left != right else (left_block - relocated_right_count)
            checksum += block_checksum(file_block, left, position)
            position += file_block
            left += 1
        else:
            empty_block = left_block - used_empty_space
            right_block = disk_line[right] - relocated_right_count
            size_to_move = min(empty_block, right_block)
            checksum += block_checksum(size_to_move, right, position)
            position += size_to_move
            if right_block == empty_block:
                used_empty_space, relocated_right_count = 0, 0
                left += 1
                right -= 2
            elif right_block > empty_block:
                relocated_right_count += empty_block
                if relocated_right_count >= disk_line[right]:
                    relocated_right_count = 0
                    right -= 2
                used_empty_space = 0
                left += 1
            else:
                relocated_right_count = 0
                used_empty_space += right_block
                if used_empty_space >= left_block:
                    used_empty_space = 0
                    left += 1
                right -= 2
    return checksum


def get_positions_shifts(disk_line: list[int]) -> list[int]:
    position = 0
    positions_shifts = []
    for block in disk_line:
        positions_shifts.append(position)
        position += block
    return positions_shifts


def get_free_space_map(disk_line: list[int]) -> dict[int, list[int]]:
    free_space_map = {}
    for i, space_size in enumerate(disk_line):
        if i % 2 != 0 and space_size > 0:
            free_space_entry = free_space_map.get(space_size, [])
            free_space_entry.append(i)
            free_space_map[space_size] = free_space_entry
    return free_space_map


def insert_sorted(sorted_list: list[int], number: int):
    """ sorted_list is a sorted list of increasing numbers """
    if len(sorted_list) == 0 or sorted_list[-1] < number:
        sorted_list.append(number)
        return
    if sorted_list[0] > number:
        sorted_list.insert(0, number)
        return
    left_search_index, right_search_index = 0, len(sorted_list) - 1
    while True:
        if right_search_index - left_search_index == 0:
            break
        delta = max((right_search_index - left_search_index) // 2, 1)
        if sorted_list[left_search_index] < number < sorted_list[right_search_index]:
            right_search_index -= delta
        elif sorted_list[right_search_index] < number:
            left_search_index = right_search_index
            right_search_index = min(right_search_index + delta, len(sorted_list) - 1)
        elif sorted_list[left_search_index] > number:
            right_search_index = left_search_index
            left_search_index = max(left_search_index - delta, 0)
        else:
            raise Exception("{} {}", left_search_index, right_search_index)
    sorted_list.insert(left_search_index + 1, number)


def get_free_list(free_space_map: dict[int, list[int]], block_size: int) -> tuple[int, list[int]]:
    """ :return used_key, list[index] """
    candidates = {}
    for empty_size, i_list in free_space_map.items():
        if empty_size >= block_size and i_list:
            candidates[i_list[0]] = (empty_size, i_list)
    if not candidates:
        return -1, []
    return candidates[min(candidates.keys())]


def day_09_02(input_list: list[str]) -> int:
    disk_line = [int(block) for block in input_list[0]]
    moved_blocks = [False for _ in range(len(disk_line))]
    free_space_map = get_free_space_map(disk_line)
    positions_shifts = get_positions_shifts(disk_line)
    checksum = 0
    for inverted_index, block_to_move in enumerate(disk_line[:0:-2]):
        right = len(disk_line) - 1 - (2 * inverted_index)
        free_size, free_list = get_free_list(free_space_map, block_to_move)
        if not free_list:
            continue
        free_index = free_list.pop(0)
        if free_index > right:
            continue
        free_size_left = free_size - block_to_move
        if free_size_left > 0:
            insert_sorted(free_space_map[free_size_left], free_index)
        moved_blocks[right] = True
        checksum += block_checksum(block_to_move, right, positions_shifts[free_index])
        positions_shifts[free_index] += block_to_move
    for i, is_moved in enumerate(moved_blocks):
        if not is_moved and (i % 2 == 0):
            checksum += block_checksum(disk_line[i], i, positions_shifts[i])
    return checksum


class Tile1:
    def __init__(self, height: int, position: tuple[int, int]):
        self.height = height
        self.reachable_nines = set()
        if height == 9:
            self.reachable_nines.add(position)


class Tile2:
    def __init__(self, height: int):
        self.height = height
        self.reachable_nines = 1 if height == 9 else 0


def get_hiking_map_1(input_list: list[str]) -> list[list[Tile1]]:
    return [[Tile1(int(h_str), (n, i)) for i, h_str in enumerate(line)] for n, line in enumerate(input_list)]


def get_hiking_map_2(input_list: list[str]) -> list[list[Tile2]]:
    return [[Tile2(int(h_str)) for h_str in line] for line in input_list]


def count_trailheads_1(hiking_map: list[list[Tile1]]) -> int:
    return sum([sum([len(tile.reachable_nines) if tile.height == 0 else 0 for tile in tile_line])
                for tile_line in hiking_map])


def count_trailheads_2(hiking_map: list[list[Tile2]]) -> int:
    return sum([sum([tile.reachable_nines if tile.height == 0 else 0 for tile in tile_line])
                for tile_line in hiking_map])


def fill_reachable_nines_on_slope_1(slope_height: int, hiking_map: list[list[Tile1]]):
    for n, tile_line in enumerate(hiking_map):
        for i, tile in enumerate(tile_line):
            if tile.height == slope_height:
                for neighbor_position in [(n - 1, i), (n + 1, i), (n, i - 1), (n, i + 1)]:
                    if is_in_area(neighbor_position, len(hiking_map)):
                        neighbor = hiking_map[neighbor_position[0]][neighbor_position[1]]
                        if neighbor.height - tile.height == 1:
                            tile.reachable_nines.update(neighbor.reachable_nines)


def fill_reachable_nines_on_slope_2(slope_height: int, hiking_map: list[list[Tile2]]):
    for n, tile_line in enumerate(hiking_map):
        for i, tile in enumerate(tile_line):
            if tile.height == slope_height:
                for neighbor_position in [(n - 1, i), (n + 1, i), (n, i - 1), (n, i + 1)]:
                    if is_in_area(neighbor_position, len(hiking_map)):
                        neighbor = hiking_map[neighbor_position[0]][neighbor_position[1]]
                        if neighbor.height - tile.height == 1:
                            tile.reachable_nines += neighbor.reachable_nines


def day_10_01(input_list: list[str]) -> int:
    hiking_map = get_hiking_map_1(input_list)
    for slope_height in range(9)[::-1]:
        fill_reachable_nines_on_slope_1(slope_height, hiking_map)
    return count_trailheads_1(hiking_map)


def day_10_02(input_list: list[str]) -> int:
    hiking_map = get_hiking_map_2(input_list)
    for slope_height in range(9)[::-1]:
        fill_reachable_nines_on_slope_2(slope_height, hiking_map)
    return count_trailheads_2(hiking_map)


MAX_STONE_SIZE = 7


def change_stone(stone: int) -> list[int]:
    if stone == 0:
        return [1]
    digits = str(stone)
    number_of_digits = len(digits)
    global MAX_STONE_SIZE
    if number_of_digits > MAX_STONE_SIZE:
        MAX_STONE_SIZE = number_of_digits
        print("digits:", number_of_digits, digits)
    if number_of_digits % 2 == 0:
        first_part, second_part = digits[:number_of_digits // 2], digits[number_of_digits // 2:]
        return [int(first_part), int(second_part)]
    return [stone * 2024]


def day_11_01(input_list: list[str]) -> int:
    FRAME_LIMIT = 25
    stones_by_frames = dict([(i, []) for i in range(FRAME_LIMIT + 1)])
    stones_by_frames[0] = to_list_of_ints(input_list[0])
    stones_count, frame = 0, 0
    while frame >= 0:
        stones = stones_by_frames[frame]
        if not stones:
            frame -= 1
        elif frame == FRAME_LIMIT:
            stones_count += len(stones)
            stones_by_frames[frame] = []
            frame -= 1
        else:
            frame += 1
            stones_by_frames[frame] += change_stone(stones.pop())
    return stones_count


def count_stones(stone: int, frames: int, stone_frames_cache: dict[tuple[int, int], int]) -> int:
    if frames == 0:
        return 1
    cashed_count = stone_frames_cache.get((stone, frames))
    if cashed_count is not None:
        return cashed_count
    return sum([count_stones(new_stone, frames - 1, stone_frames_cache) for new_stone in change_stone(stone)])


def day_11_02(input_list: list[str]) -> int:
    stone_frames_cache = {}

    return 0


class Region:
    def __init__(self, plant: str):
        self.plant = plant
        self.region_id = -1  # -1 means no id
        self.perimeter = 4
        self.vertices = 0


def calculate_fence_price_1(garden: list[list[Region]]) -> int:
    square_and_perimeter_by_region_id = {}
    for line in garden:
        for region in line:
            square, perimeter = square_and_perimeter_by_region_id.get(region.region_id, (0, 0))
            square_and_perimeter_by_region_id[region.region_id] = (square + 1, perimeter + region.perimeter)
    return sum([square * perimeter for square, perimeter in square_and_perimeter_by_region_id.values()])


def calculate_fence_price_2(garden: list[list[Region]]) -> int:
    square_and_vertices_by_region_id = {}
    for line in garden:
        for region in line:
            square, vertices = square_and_vertices_by_region_id.get(region.region_id, (0, 0))
            square_and_vertices_by_region_id[region.region_id] = (square + 1, vertices + region.vertices)
    return sum([square * vertices for square, vertices in square_and_vertices_by_region_id.values()])


def visit_1(region_id: int, position: tuple[int, int], garden: list[list[Region]]):
    n, i = position
    region = garden[n][i]
    region.region_id = region_id
    for neighbor_position in [(n - 1, i), (n + 1, i), (n, i - 1), (n, i + 1)]:
        if is_in_area(neighbor_position, len(garden)):
            neighbor = garden[neighbor_position[0]][neighbor_position[1]]
            if neighbor.plant == region.plant:
                region.perimeter -= 1
                if neighbor.region_id == -1:
                    visit_1(region_id, neighbor_position, garden)


def visit_2(position: tuple[int, int], garden: list[list[Region]]):
    n, i = position
    region = garden[n][i]

    def from_same_region(n_pos: int, i_pos: int) -> bool:
        return is_in_area((n_pos, i_pos), len(garden)) and garden[n_pos][i_pos].region_id == region.region_id

    for neighbor_pair_shift in [
        ((0, -1), (-1, 0)),  # LEFT + UP
        ((-1, 0), (0, 1)),  # UP + RIGHT
        ((0, 1), (1, 0)),  # RIGHT + DOWN
        ((1, 0), (0, -1))  # DOWN + LEFT
    ]:
        shift_1, shift_2 = neighbor_pair_shift
        d_n_1, d_i_1 = shift_1
        d_n_2, d_i_2 = shift_2
        diagonal_position = (n + d_n_1 + d_n_2, i + d_i_1 + d_i_2)
        if (not from_same_region(n + d_n_1, i + d_i_1) and not from_same_region(n + d_n_2, i + d_i_2)
                or from_same_region(n + d_n_1, i + d_i_1)
                and from_same_region(n + d_n_2, i + d_i_2)
                and not from_same_region(*diagonal_position)):
            region.vertices += 1


def day_12_01(input_list: list[str]) -> int:
    garden = [[Region(plant) for plant in line] for line in input_list]
    region_counter = 0
    for n in range(len(garden)):
        for i in range(len(garden)):
            if garden[n][i].region_id == -1:
                visit_1(region_counter, (n, i), garden)
                region_counter += 1
    return calculate_fence_price_1(garden)


def day_12_02(input_list: list[str]) -> int:
    garden = [[Region(plant) for plant in line] for line in input_list]
    region_counter = 0
    for n in range(len(garden)):
        for i in range(len(garden)):
            if garden[n][i].region_id == -1:
                visit_1(region_counter, (n, i), garden)
                region_counter += 1
    for n in range(len(garden)):
        for i in range(len(garden)):
            visit_2((n, i), garden)
    return calculate_fence_price_2(garden)


class Robot:
    def __init__(self, position: tuple[int, int], velocity: tuple[int, int]):
        self.position = position
        self.velocity = velocity


def collect_robots(input_list: list[str]) -> list[Robot]:
    def extract_tuple(eq_str: str) -> tuple[int, int]:
        _, tuple_str = eq_str.split('=')
        x, y = tuple_str.split(',')
        return int(y), int(x)  # I want line number as the first element

    return [Robot(*map(extract_tuple, line.split(' '))) for line in input_list]


# SPACE_WIDTH, SPACE_TALL = 11, 7
SPACE_WIDTH, SPACE_TALL = 101, 103


def move_robot(robot: Robot, seconds: int):
    n, i = robot.position
    d_n, d_i = robot.velocity
    robot.position = ((n + d_n * seconds) % SPACE_TALL, (i + d_i * seconds) % SPACE_WIDTH)


def safety_factor(robots: list[Robot]) -> int:
    top_L, top_R, bottom_L, bottom_R = 0, 0, 0, 0
    middle_n, middle_i = SPACE_TALL // 2, SPACE_WIDTH // 2
    for robot in robots:
        n, i = robot.position
        if n < middle_n and i < middle_i:
            top_L += 1
        elif n < middle_n and i > middle_i:
            top_R += 1
        elif n > middle_n and i < middle_i:
            bottom_L += 1
        elif n > middle_n and i > middle_i:
            bottom_R += 1
    return top_L * top_R * bottom_L * bottom_R


def day_14_01(input_list: list[str]) -> int:
    robots = collect_robots(input_list)
    for robot in robots:
        move_robot(robot, 100)
    return safety_factor(robots)


A_COST = 3
B_COST = 1


def get_prize_cost(button_a: tuple[int, int], button_b: tuple[int, int], prize: tuple[int, int]) -> int:
    a_dx, a_dy = button_a
    b_dx, b_dy = button_b
    p_x, p_y = prize
    dividend = a_dx * p_y - a_dy * p_x
    divisor = a_dx * b_dy - a_dy * b_dx
    if dividend % divisor != 0:
        return 0
    N_b = dividend // divisor
    if N_b < 0:
        return 0
    if (p_x - N_b * b_dx) % a_dx != 0:
        return 0
    N_a = (p_x - N_b * b_dx) // a_dx
    if N_a < 0:
        return 0
    return A_COST * N_a + B_COST * N_b


def parse_button(line: str) -> tuple[int, int]:
    tuple_str = line.split(', ')
    x = tuple_str[0].split('+')[1]
    y = tuple_str[1].split('+')[1]
    return int(x), int(y)


def parse_prize(line: str) -> tuple[int, int]:
    tuple_str = line.split(', ')
    x = tuple_str[0].split('=')[1]
    y = tuple_str[1].split('=')[1]
    return int(x), int(y)


def get_tokens_count(input_list: list[str], prize_offset: int) -> int:
    tokens_count = 0
    button_a, button_b = (0, 0), (0, 0)
    state = 'A'  # B and P
    for line in input_list:
        if not line:
            continue
        elif state == 'A':
            button_a = parse_button(line)
            state = 'B'
        elif state == 'B':
            button_b = parse_button(line)
            state = 'P'
        elif state == 'P':
            prize_x, prize_y = parse_prize(line)
            tokens_count += get_prize_cost(button_a, button_b, (prize_x + prize_offset, prize_y + prize_offset))
            state = 'A'
    return tokens_count


def day_13_01(input_list: list[str]) -> int:
    return get_tokens_count(input_list, 0)


def day_13_02(input_list: list[str]) -> int:
    return get_tokens_count(input_list, 10000000000000)


def get_move_position(move: str, position: tuple[int, int]) -> tuple[int, int]:
    n, i = position
    if move == '<':
        return n, i - 1
    if move == '^':
        return n - 1, i
    if move == '>':
        return n, i + 1
    if move == 'v':
        return n + 1, i
    raise Exception('Unexpected move')


def get_free_position(move: str, start_position: tuple[int, int], warehouse: list[list[str]]) -> tuple[bool, tuple[int, int]]:
    target = warehouse[start_position[0]][start_position[1]]
    if target == '#':
        return False, (-1, -1)
    if target == 'O':
        return get_free_position(move, get_move_position(move, start_position), warehouse)
    return True, start_position


def calculate_sum_of_goods_positions(warehouse: list[list[str]]) -> int:
    sum_of_goods_positions = 0
    for n, line in enumerate(warehouse):
        for i, box in enumerate(line):
            if box == 'O':
                sum_of_goods_positions += 100 * n + i
    return sum_of_goods_positions


def day_15_01(input_list: list[str]) -> int:
    warehouse_size = len(input_list[0])
    warehouse = [list(line) for line in input_list[:warehouse_size:]]
    moves = ''.join(input_list[warehouse_size::])
    robot_position = find_symbol_position('@', input_list[:warehouse_size:])
    for move in moves:
        next_position = get_move_position(move, robot_position)
        is_free, free_position = get_free_position(move, next_position, warehouse)
        if is_free:
            robot_position = next_position
            if next_position != free_position:
                warehouse[next_position[0]][next_position[1]] = '.'
                warehouse[free_position[0]][free_position[1]] = 'O'
    return calculate_sum_of_goods_positions(warehouse)


if __name__ == '__main__':
    lines = list_lines_from_file("input/Day15.txt")
    day_function = day_15_01
    print(day_function(lines))
