Repo very much wip

install the modified gym-minigrid i.e
 git submodule init
 git submodule update
 cd gym_minigrid
 pip install -e .

run the wall colour task with the online SF net with 
python main.py

to viz the env:
python main.py --show_training 1

All tasks are in part-obs envs (7x7 cone of vision) for now.

vanilla Task:

3 classes of object, r_c sampled from {-1,0,1}, r_g ~= 1*0.999^t
object positions are static, goal location static
resample every 100000 transisions
run for 5M steps(?)

Structure task 1 (wall_colour):
same reward set up
colour of wall will change periodically (set period)
uncorrelated with reward

Structure task 2 (landmarks):
same reward set up
introduce land marks into room (eg pillar)
move landmarks periodically 
uncorrelated with reward
