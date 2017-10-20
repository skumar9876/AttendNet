import curses
stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(1)

import gym
import gym_minecraft
env = gym.make('MinecraftSimpleRoomMaze-v0')
env.load_mission_file('/home/linguini/Documents/DeepRL/game.xml')
env.init(start_minecraft=True, allowDiscreteMovement=True)

output_file = ('Minecraft_Player_Output.txt')

f = open(output_file, 'w')

for i_episode in range(1):
    observation = env.reset()
    print env.action_space
    for t in range(1000):
        env.render()
        #print(observation)
        
        key = ''

        # Keys for continuous action space
        # allowed_keys = [ord('r'), # 0 = crouch
        #                 ord('t'), # 1 = uncrouch?
        #                 ord(' '), # 2 = jump
        #                 ord('y'), # 3 = click?
        #                 curses.KEY_UP, # 4 = move forward
        #                 curses.KEY_DOWN, # 5 = move backwards
        #                 ord('w'), # 6 = turn up
        #                 ord('s'), # 7 = turn down
        #                 curses.KEY_RIGHT, # 8 = move right
        #                 curses.KEY_LEFT, # 9 = move left
        #                 ord('d'), # 10 = turn right
        #                 ord('a'), # 11 = turn left
        #                 ord('g'), # 12 = ?
        #                 ord('h'), # 13 = ?
        #                 ]

        # Keys for discrete action space
        allowed_keys = [curses.KEY_UP, # 0 = move forward,
                        curses.KEY_DOWN, # 1 = move forward
                        ord('#'),
                        ord('#'),
                        curses.KEY_RIGHT,
                        curses.KEY_LEFT,
                        ord('#'),
                        ord('#'),
                        ord('d'),
                        ord('a'),

                        ] + 14 * [0]
        
        while key not in allowed_keys:
            env.render()
            key = stdscr.getch()
            action = 0
            for i in xrange(len(allowed_keys)):
                if key == allowed_keys[i]:
                    action = i

        observation, reward, done, info = env.step(action)
        f.write(str(observation) + "\t" + str(reward) + "\t" + str(action) + "\n")
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()