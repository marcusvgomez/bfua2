import cv2
import numpy as np
import os, sys

IMG_DIM = 256
RADIUS = 10
CLIP_BOUND = 10.0
PATH = os.path.dirname(__file__) + '/../imgs/'

BLACK = (0, 0, 0)
GRAY = (120, 120, 120)

def convert_color(c):
    # converts int color c in our state to rgb
    cp = int(c.data[0])
    if cp == 0:
        return (255, 0, 0)
    elif cp == 1:
        return (0, 255, 0)
    elif cp == 2:
        return (0, 0, 255)
    else:
        return (30, 30, 30)
        


def draw(agent_state, name='visualization.png'): 
    # takes in UNIVERSAL agent state from env as a numpy array and outputs a file with given name
    img = np.zeros((IMG_DIM, IMG_DIM, 3)) # blank white img to begin with
    img.fill(255)
    STATE_DIM, N = agent_state.shape
    CENTER = IMG_DIM/2
    
    # draw axes
    cv2.line(img, (CENTER, 0), (CENTER, IMG_DIM - 1), (0, 0, 0))
    cv2.line(img, (0, CENTER), (IMG_DIM - 1, CENTER), (0, 0, 0))
    
    for i in range(N):
        pos_x, pos_y = agent_state[0:2, i]
        pos_y = -pos_y # fix differences in directions between matrix and spatial coordinates
        vel_x, vel_y = agent_state[2:4, i]
        vel_y = -vel_y
        gaze_x, gaze_y = agent_state[4:6, i]
        gaze_y = -gaze_y
        color = agent_state[6, i]
        
        ## draw position with color
        max_dist = int(0.9 * IMG_DIM/2) # maximum number of pixels to go in x/y axis direction with center of image being the origin
        x, y = int(pos_x / CLIP_BOUND * max_dist), int(pos_y / CLIP_BOUND * max_dist)
        cv2.circle(img, (CENTER + x, CENTER + y), RADIUS, convert_color(color), -1)
        
        ## draw velocity as an arrow
        v_x, v_y = int(vel_x / CLIP_BOUND * max_dist), int(vel_y / CLIP_BOUND * max_dist)
        cv2.arrowedLine(img, (CENTER + x, CENTER + y), (CENTER + x + v_x, CENTER + y + v_y), color=BLACK, thickness=2)
        
        ## draw gaze as a differently colored arrow
        g_x, g_y = int(gaze_x / CLIP_BOUND * max_dist), int(gaze_y / CLIP_BOUND * max_dist)
        cv2.arrowedLine(img, (CENTER + x, CENTER + y), (CENTER + x + g_x, CENTER + y + g_y), color=GRAY, thickness=2)
        
    cv2.imwrite(name, img)


if __name__ == '__main__':
    agent_state = np.zeros((7, 2))
    agent_state[0:2, 0] = (5, 6)
    agent_state[2:4, 0] = (1, 2)
    agent_state[4:6, 0] = (0, 5)
    agent_state[6, 0] = 1
    agent_state[0:2, 1] = (-10, -10)
    draw(agent_state, name='testviz.png')
    
