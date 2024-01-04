# agriculture_robot
Fruit pick and place robot

# This package spawns a UR10 robot into a virtual agriculture field and picks up apples from the trees
## Instructions to run the package

### Step 1
>git clone this package into your src folder

>sudo apt install ros-galactic-ros2-control ros-galactic-ros2-controllers ros-galactic-gazebo-ros2-control

>sudo apt-get install ros-galactic-controller-manager

### Step 2
Extract the contents of models.zip and run the following command for each folder inside the zip folder
> cp -r your_folder_name/ ~/.gazebo/models

### Step 3
>cd \<path to workspace root directory\>
>colcon build

Next time you can do the following
>colcon build --packages-select agriculture_robot

### Step 4
>source install/setup.bash

### Step 5
>ros2 launch agriculture_robot gazebo.launch.py

### Step 6
open a new terminal window source ros2, source workspace
>ros2 run agriculture_robot pick_and_place.py

### Troubleshooting
You will have to modify the path to the "live_data.csv" file before building the package in the agriculture_robot/src/pick_and_place.py file 

### Live Joint Update
After running the launch file and the ros2 node, and the robot has finished moving along the pre-programmed trajectory, you can give it live joint angles by modifying the live_data.csv fine located in the /src folder. Another way to update this file is via the FK_test_script.py file located in the same directory. 



