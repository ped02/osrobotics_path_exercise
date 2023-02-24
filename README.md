# Probabilistic Roadmap Path Planning Exercise

This is an attempt solution to the exercise problem for path planning and path post-processing from Open-Source Robotics [[Exercise 1](https://osrobotics.org/osr/planning/path_planning.html)] [[Exercise 2](https://osrobotics.org/osr/planning/path_planning.html)]. Python package version is given in `requirements.txt`.

Language: Python3 - Python 3.8.6

## Roadmap Result
### Paths only
![PRM 1](/imgs/PRM_1.png)

<b>Green Path</b>: Path generated from PRM

<b>Cyan Path</b>: Post-processed path generated from path shortcutting


### Paths and PRM Graph
![PRM 2](/imgs/PRM_2.png)
<b>Blue Vertices</b>: Vertices of roadmap graph

<b>Blue Edges</b>: Edges of roadmap graph

## Implementations

` environment_2d.py `

Modified from exercise template from osrobotics.org [[original](https://github.com/crigroup/osr_course_pkgs/blob/master/osr_examples/scripts/environment_2d.py)] to include triangle-segment intersection for obstacle and numpy calculations(for fast calculation), as well as various refactor.

` rpm_1.ipynb `

This jupyter notebook file contains majority of the Probabilistic Roadmap implementation. The classes and functions are implemented at the top and the exercise trials are to be run at the bottom. You can run the notebook cells multiple times to adjust and see different map configuration and parameters. If the plot does not show, you can try to remove or change the notebook magic ` %matplotlib notebook `.


Parameters | Description | Example Value
-----------|-------------|---------------
CONNECT_RADIUS | Radius to connect neighbouring nodes | 1.5 (distance units)
SEARCH_ITERATION | Max number of iterations to attempt add_node for path search | 300
PATH_QUERY_PERIOD | Number of iterations to periodically query if start and goal can be reached | 15
SHORTCUT_ITERATION | Number of iterations to attempt path shortcutting | 1000


` env_test_1.py `

File to test environment generation. Based from exercise template.

## Notes
For PRM `add_node` method, currently use numpy for distance calculation to determine neighbor for connecting the new node to existing graph. I'm not sure whether using 'bruteforce' distance calculation using numpy vs storing order of points ordered by axis (x and y) then binary search to query range of possible neighbour would be better. But due to time constraints, I am not able to implement both for benchmarking. I feel the advantage with the 'bruteforce' method is it is more general and it allows other 'cost' function to be used rather than just the euclidean distance. 