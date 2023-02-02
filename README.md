# ProComSci
Project Computational Science for the bachelor Computer Science at UvA.
 Imagine you're a paratrooper on a mission to destroy the football stadium of the Las Vegas Raiders.
 This is (hopefully) quite a leap of the imagination for you, but in this project, this happens thousands of times a day... on a computer.
 We investigate (some of) the work that goes into dropping a skydiver from an airplane, 
 with the objective of landing on a designated landing point, 
 subject to restrictions that a realistic situation of this nature would also encounter.
 
 In our research, answering the following question is key: 
 Given a desired landing point, a permitted flying path in Las Vegas and an approximate wind direction conforming to a typical day in Las Vegas, 
 what are the best drop position and opening height of the parachute, for a skydiver to land as close as possible to the landing point?
 
 The code simulates a skydiver that jumps from an airplane. Different calculations are made such as best drop position and opening height for the parachute.
 Also different numerical methods are tested against the Runge-Kutta order 4 method with step size 10^{-4}.
 There are also simple calculations such as speed, acceleration and the height. Also the wind is calculated, the wind is
 obtained from a database, then since the wind is scatterd a random walk through the data points has been made with a cubic spline, this will be the wind that will be used in the simulation. For the speed, acceleration, height and wind see the figure below.
 
 ![plot_for_git](https://user-images.githubusercontent.com/85616002/216413659-57a5bc1d-972d-4a9b-bb23-cd5fa20c09f3.png)
 
 To obtain this plot, do the following:
 - Download al the files from this repository.
 - Store them in a folder all together.
 - Run main.py, by the following command 'python3 main.py' or different if you run python files different.
 - Now you see different options, press now 2 and then Enter to get the figure above.

