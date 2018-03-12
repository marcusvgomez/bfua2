import cv2
import numpy as np

IMG_DIM = 512 # want it divisible by 14
RADIUS = 10
GRID_LEN = IMG_DIM / 14 # length in pixels of a single square of the grid

GRAY = (220, 220, 220)
BLACK = (0, 0, 0)

def in_bounds(x, y):
    return x >= 0 and x < 14 and y >= 0 and y < 14

def get_color_for_idx(idx):
    # assigns each index a unique fixed color 
    num = hash(str(idx))
    r = (num % 255)
    g = (num/200) % 255
    b = (num/40000) % 255
    return (r*2/3, g*2/3, b*2/3) # multiplier to make sure colors are sufficiently dark

def draw_track(img):
    vert_squares = [(r*GRID_LEN, c*GRID_LEN) for c in [6, 7] for r in range(14)]
    horiz_squares = [(r*GRID_LEN, c*GRID_LEN) for r in [6, 7] for c in range(14)]
    track_squares = list(set(vert_squares + horiz_squares)) # eliminate duplicates in the center
    for r, c in track_squares:
        cv2.rectangle(img, (r, c), (r + GRID_LEN, c + GRID_LEN), color=GRAY, thickness=cv2.FILLED) # filled in rect
        cv2.rectangle(img, (r, c), (r + GRID_LEN, c + GRID_LEN), color=(150, 150, 150), thickness=1) # border

def draw_traffic(agents, name='visualization_traffic.png'):
    # creates a visualization image of the agents taken from a traffic environment
    img = np.zeros((IMG_DIM, IMG_DIM, 3)) # blank white img to begin with
    img.fill(255)
    
    draw_track(img)
    for agent in agents:
        idx, loc, route, t, remaining_steps = agent
        if not in_bounds(*loc):
            continue
        r, c = loc[0]*GRID_LEN, loc[1]*GRID_LEN
        
        color = get_color_for_idx(idx)
        cv2.circle(img, (c + GRID_LEN/2, r + GRID_LEN/2), radius=GRID_LEN/2 - 1, color=color, thickness=-1)
        
        num_steps_to_draw = 4 if route == 1 else 8 # route 1 is going straight, the other two involve a turn
        steps_to_draw = remaining_steps[0:num_steps_to_draw]
        for i in range(len(steps_to_draw)):
            dx, dy = remaining_steps[i]
            shift = (idx % 4 - 2)*GRID_LEN/8 # to make sure the paths don't all overlap when drawn
            r_start, c_start = loc[0]*GRID_LEN + GRID_LEN/2 + shift, loc[1]*GRID_LEN + GRID_LEN/2 + shift
            r_end, c_end = (loc[0]+dx)*GRID_LEN + GRID_LEN/2 + shift, (loc[1]+dy)*GRID_LEN + GRID_LEN/2 + shift
            # if i is the last index, then we draw an arrowed line, otherwise regular solid
            if i == len(steps_to_draw) - 1:
                cv2.arrowedLine(img, (c_start, r_start), (c_end, r_end), color, thickness=2, tipLength=0.2)
            else:    
                cv2.line(img, (c_start, r_start), (c_end, r_end), color, thickness=2)
            
            loc = (loc[0] + dx, loc[1] + dy)
    
    cv2.imwrite(name, img)
    
if __name__ == '__main__':
    test_agents = [[0, (7, 3), 0, 1, 4*[(0,1)] + 7*[(-1,0)]], [1, (6, 10), 0, 1, 3*[(0,-1)] + 6*[(-1,0)]], [2, (7, 0), 0, 1,13*[(0,1)]] ]
    draw_traffic(test_agents, name='testvis.png')
    
