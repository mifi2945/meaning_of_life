import argparse
import tracemalloc
import time
from colorama import Fore

from environment import Agent, Dust, Thing, Wall, Walkable
from problem import MazeNavigation
from maze_visualizer import MazeVisualizer

WIN_WIDTH = 600
WIN_HEIGHT = 600
UPDATE_INTERVAL = 500

TEST_STARTER = True
if TEST_STARTER:
    from search import random_search, breadth_first_search, depth_first_search, astar_search, greedy_search, \
        iterative_deepening, uniform_search, DustMaze
else:
    try:
        from search_solution import random_search, breadth_first_search, depth_first_search, astar_search, greedy_search, \
            iterative_deepening, uniform_search, DustMaze
        print(f"Loading solution file.")
    except:
        from search import random_search, breadth_first_search, depth_first_search, astar_search, greedy_search, \
            iterative_deepening, uniform_search, DustMaze


uninformed_mazes = ['basic', 'uninformed1', 'uninformed2', 'nonuniform', 'unsolvable','large_maze1','large_maze2']
informed_mazes = ['informed1']
dust_mazes = ["dust1"]
maze_types = uninformed_mazes + informed_mazes + dust_mazes

Open = Walkable

N = MazeNavigation.DIRECTIONS[0]
E = MazeNavigation.DIRECTIONS[1]
S = MazeNavigation.DIRECTIONS[2]
W = MazeNavigation.DIRECTIONS[3]

def to_index(width:int, x:int, y:int) -> int:
    return y * width + x

def maze_with_walls(width:int, height:int) -> list[Thing]:
    initial_state:list[Thing] = [Open() for _ in range(width * height)]
    # walls around the perimeter
    for j in range(width):
        x = j
        y1 = 0
        y2 = height - 1
        initial_state[y1 * width + x] = Wall()
        initial_state[y2 * width + x] = Wall()
    for i in range(height):
        x1 = 0
        x2 = width - 1
        y = i
        initial_state[y * width + x1] = Wall()
        initial_state[y * width + x2] = Wall()
    return initial_state

def check_location(things:tuple[Thing], location:tuple[int, int]):
    for thing in things:
        if thing.location == location:
            return True
    return False

def create_maze_problem(maze_type: str, search_type: str) -> [MazeNavigation, tuple[str, ...]]:
    width:int
    height:int
    initial_state:list[Thing]
    goal_state:tuple[int, int]
    thing_locations:tuple[tuple[int, int], ...]

    if maze_type == 'basic':
        """
        Basic maze used for testing and debugging.
        """
        width = 6
        height = 5
        maze \
            = [Wall(), Wall(), Wall(), Wall(), Wall(), Wall(),
               Wall(), Open(), Open(), Open(), Open(), Wall(),
               Wall(), Wall(), Wall(), Wall(), Open(), Wall(),
               Wall(), Open(), Open(), Open(), Open(), Wall(),
               Wall(), Wall(), Wall(), Wall(), Wall(), Wall()]
        # agent location
        thing_locations = ((1,1),)
        goal_state = (1,3)
        solution = [E, E, E, S, S, W, W, W]
    elif maze_type == 'uninformed1' or maze_type == 'uninformed2':
        """
        Mazes that show the difference in returned path between DFS and the other Uninformed Searches. 
        Why does the returned path of DFS change based on the starting and end locations?
        """
        width = 9
        height = 7
        maze \
            = [Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(),
               Wall(), Open(), Open(), Open(), Wall(), Open(), Open(), Open(), Wall(),
               Wall(), Open(), Wall(), Open(), Wall(), Open(), Wall(), Open(), Wall(),
               Wall(), Open(), Wall(), Open(), Open(), Open(), Wall(), Open(), Wall(),
               Wall(), Open(), Wall(), Wall(), Wall(), Wall(), Wall(), Open(), Wall(),
               Wall(), Open(), Open(), Open(), Open(), Open(), Open(), Open(), Wall(),
               Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall()]
        if maze_type == 'uninformed1':
            thing_locations = ((1,1),)
            goal_state = (7, 5)
            solution = [S, S, S, S, E, E, E, E, E, E]
        else: # 'uninformed2'
            thing_locations = ((7, 1),)
            goal_state = (1, 5)
            if search_type == "DFS":
                solution = [W, W, S, S, W, W, N, N, W, W, S, S, S, S]
            else:  #BFS or ID
                solution = [S, S, S, S, W, W, W, W, W, W]
    elif maze_type == 'nonuniform':
        """
        Mazes that show the difference in returned path between Uniform and the other Uninformed Searches. 
        In this maze the cost to move to certain tiles is changed from the default value of 1.
        Do the returned path for Uniform make sense?
        """
        width = 5
        height = 7
        maze \
            = [Wall(), Wall(), Wall(), Wall(), Wall(),
               Wall(), Open(), Open(0.5), Open(), Wall(),
               Wall(), Open(9), Wall(), Open(), Wall(),
               Wall(), Open(), Wall(), Open(), Wall(),
               Wall(), Open(), Wall(), Open(), Wall(),
               Wall(), Open(), Open(), Open(), Wall(),
               Wall(), Wall(), Wall(), Wall(), Wall()]
        thing_locations = ((1,1),)
        goal_state = (1, 3)
        if search_type == 'Uniform':
            solution = [E, E, S, S, S, S, W, W, N, N]
        else:
            solution = [S, S]
    elif maze_type == "unsolvable":
        """
        Maze to verify that your code will terminate and return failure 
        (i.e., []) if a solution can not be found.
        """
        width = 6
        height = 5
        maze \
            = [Wall(), Wall(), Wall(), Wall(), Wall(), Wall(),
               Wall(), Open(), Open(), Open(), Open(), Wall(),
               Wall(), Wall(), Wall(), Wall(), Open(), Wall(),
               Wall(), Open(), Wall(), Open(), Open(), Wall(),
               Wall(), Wall(), Wall(), Wall(), Wall(), Wall()]
        thing_locations = ((1, 3),)
        goal_state = (-1, -1) #unreachable
        solution = []
    elif maze_type == 'large_maze1' or maze_type == 'large_maze2':
        """
        Large mazes to help demonstrate the slowdown with searches like BFS compared to DFS. 
        Either of these mazes will turn the verbose time and space recording in run_search()
        """
        width = 25
        height = 25
        maze = maze_with_walls(width, height)

        center = (width // 2, height // 2)
        far_left = (1, height//2)
        far_south = (width//2, height - 2)

        thing_locations = ((center[0], center[1]), )

        if maze_type == 'large_maze1':
            goal_state = far_left
            solution = [W for _ in range(center[0] - far_left[0])]
        else:
            goal_state = far_south
            if search_type == "DFS":
                solution = ([W for _ in range(center[0] - far_left[0])] +
                            [S for _ in range(height - 2 - far_left[1])] +
                            [E for _ in range(center[0] - far_left[0])])
            else: #BFS and ID
                solution = [S for _ in range(far_south[1] - center[1])]
    elif maze_type == 'informed1':
        """
        Mazes that illustrate the difference in paths between A* and Greedy.
        """
        width = 7
        height = 7
        maze = maze_with_walls(width, height)

        # create two horizontal walls
        for j in range(2, 6):
            index = to_index(width, j, 2)
            maze[index] = Wall()
        for j in range(2, 5):
            index = to_index(width, j, 4)
            maze[index] = Wall()

        thing_locations = ((3,5),)
        goal_state = (5,1)
        if search_type == "Greedy":
            solution = [E, E, N, N, W, W, W, W, N, N, E, E, E, E]
        else: #AStar
            solution = [W, W, N, N, N, N, E, E, E, E]
    elif maze_type == 'dust1':
        """
        Maze contains dust objects and is used with DustMaze. 
        These mazes also illustrate the difference in paths between A* and Greedy.
        """

        width = 9
        height = 7
        maze \
            = [Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(),
               Wall(), Open(), Open(), Open(), Wall(), Open(), Open(), Open(), Wall(),
               Wall(), Open(), Wall(), Open(), Wall(), Open(), Wall(), Open(), Wall(),
               Wall(), Open(), Wall(), Open(), Open(), Open(), Wall(), Open(), Wall(),
               Wall(), Open(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(),
               Wall(), Open(), Open(), Open(), Open(), Open(), Open(), Open(), Wall(),
               Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall()]
        agent_location = ((1,2),)
        dust_locations = ((1,1), (5,1), (3,2), (7, 3), (1, 5), (7, 5))
        thing_locations = agent_location + dust_locations
        goal_state = (-1, -1) #goal state doesn't matter here
        if search_type == "AStar":
            solution = [S, S, S, E, E, E, E, E, E, W, W, W, W, W, W, N, N, N, N, E, E, S, S, E, E, N, N, E, E, S, S]
        else: # Greedy
            solution = [N, E, E, S, S, E, E, N, N, E, E, S, S, N, N, W, W, S, S, W, W, N, N, W, W, S, S, S, S, E, E, E, E, E, E]
    else:
        raise ValueError(f"Error, invalid maze type {maze_type}")

    if "dust" in maze_type:
        return [DustMaze(thing_locations, goal_state, maze, width, height), solution]
    else:
        return [MazeNavigation(thing_locations, goal_state, maze, width, height), solution]


def run_search(search_type: str, maze_type: str, verbose: int = 0, display: bool = False):
    # hardcode values here to do debugging tests
    # search_type = 'Uniform'
    # maze_type = 'nonuniform'
    # display = True

    if maze_type =='large_maze1' or maze_type == 'large_maze2':
        # turn on the time and space measurements for the larger maze
        verbose = True

    problem:MazeNavigation
    solution:tuple[str, ...]

    problem, solution = create_maze_problem(maze_type, search_type)

    if search_type == "random":
        search = random_search
    elif search_type == "BFS":
        search = breadth_first_search
    elif search_type == "DFS":
        search = depth_first_search
    elif search_type == "ID":
        search = iterative_deepening
    elif search_type == "Uniform":
        search = uniform_search
    elif search_type == "AStar":
        search = astar_search
    elif search_type == "Greedy":
        search = greedy_search
    else:
        print("Invalid search type")
        return

    # memory and time measurements
    start = time.time()
    tracemalloc.start()

    path = search(problem)

    memory_usage = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    run_time = time.time() - start
    path_length = len(path)

    print(f"\nSearch: {search_type}")
    print(f"Maze type: {maze_type}")
    print(f"\tResulting actions:\n{path}")
    if problem.verify(path):
        print(f"Returned path solves the problem.")
    else:
        print(f"{Fore.RED}Returned path does not solve the problem.{Fore.RESET}")
    if path == solution:
        print(f"{Fore.GREEN}Returned actions matches expected actions.{Fore.RESET}")
    else:
        print(f"{Fore.RED}Returned actions do not match expected actions.{Fore.RESET}")
        print(f"\tExpected actions:\n{solution}")
        # for i in range(len(path)):
        #     print(f"{path[i]} {expected[i]}")

    if display:
        vis = MazeVisualizer(WIN_WIDTH, WIN_HEIGHT, problem.width, problem.height)

        current = problem.initial_state
        for a in path:
            # make a copy of the maze
            display_maze = problem.maze.copy()
            # adds the agent to the maze
            x,y = current[0]
            display_maze[problem.to_index(x,y)] = Agent()
            # add dust to the maze
            for thing in current[1:]:
                x,y = thing
                display_maze[problem.to_index(x,y)] = Dust()

            vis.display_maze(display_maze, [problem.to_location(i) for i in range(len(display_maze))])
            vis.update_display(UPDATE_INTERVAL)
            current = problem.result(current, a)

    if verbose > 0:
        print(f"Memory usage: {memory_usage:.2e}")
        print(f"Elapsed time {run_time:.4f}")
        print(f"Path length: {path_length}")
        print(f"Path: {path}")

def main():
    parser = argparse.ArgumentParser(description="Main program to run the search.")
    parser.add_argument("-s", "--search_type", type=str, default="random",
                        help="Search algorithm type: random, BFS, DFS, ID, Uniform, AStar, Greedy.")
    parser.add_argument("-m", "--maze_type", type=str, default="basic",
                        help=f"Maze types {maze_types}")
    parser.add_argument("-d", "--display", action='store_true',
                        help="Turns on the display.")
    parser.add_argument('-v', '--verbose', action="count", default=0,
                        help="Increase verbosity level.")

    args = parser.parse_args()
    run_search(**vars(args))

if __name__ == '__main__':
    main()
