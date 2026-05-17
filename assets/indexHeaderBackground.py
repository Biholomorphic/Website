import math
import numpy as np
import random
import colorsys
from pathlib import Path
from PIL import Image, ImageDraw

# Params
threshold = 150     # Threshold for projection overlap, higher -> more likely to accept lines, lower -> less likely to accept lines.
k = 30              # Num of arrempts per active point in Bridson's algo
length_start = 60   # min line length
length_end = 80     # max line length
vibrance_min = 60   # min vibrance for line colour
vibrance_max = 95   # max vibrance for line colour

canvas_width    = 1920
canvas_height   = 1080


# --------------------
# TODO:
# 1. Add distance testing to the end points, so that they don't overlap with each other.
#   - Needs to compare to other end points, so need to store in a list and loop through to check min distance.

im = Image.new('RGB', (canvas_width, canvas_height), "#0e1017")
draw = ImageDraw.Draw(im)

# Source - https://stackoverflow.com/a/24852375
# Posted by Cory Kramer, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-15, License - CC BY-SA 3.0

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

# End source

random_vibrance_value = [random.randint(vibrance_min, vibrance_max) for _ in range(40)] # Also very fire way of using loop
random_lengths = [random.randint(length_start, length_end) for _ in range(40)]

# arr - dummy array, radius is min dist between two points, k is num of attempts per active pts
def BridsonPoissonDiskSampler(arr, radius):
    # Cell size, comes from pythagoras r^2 = cellSize^2 + cellSize^2 => cellSize = r / sqrt(2)
    cellSize = radius / math.sqrt(2)
    # Grid dimensions comes from dividing the canvas into cells of size cellSize, and rounding up to make sure we cover the whole canvas
    gridWidth = int(math.ceil(arr.shape[1] / cellSize))
    gridHeight = int(math.ceil(arr.shape[0] / cellSize))
    grid = np.full((gridHeight, gridWidth), -1, dtype=int)
    # create empty list of active points and samples
    activeList = []
    samples = []

    # seed the algo - pick a random starting point, add it to samples and activeList, mark its cell
    seed_x = random.randint(0, arr.shape[1])
    seed_y = random.randint(0, arr.shape[0])
    activeList.append((seed_x, seed_y))
    samples.append((seed_x, seed_y))
    gx = int(seed_x / cellSize)
    gy = int(seed_y / cellSize)
    grid[gy][gx] = 0

    while len(activeList) > 0:
        # pick random point from activeList
        idx = random.randint(0, len(activeList) - 1)
        point = activeList[idx]

        accepted = False
        for _ in range(k):
            # Generate candidate point between r and 2*r from active pt. at random angle
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(radius, 2 * radius)
            candidate_x = int(point[0] + r * math.cos(angle))
            candidate_y = int(point[1] + r * math.sin(angle))
            
            # Do a bounds check - fuck off if outside of canvas dimensions
            if candidate_x < 0 or candidate_x >= arr.shape[1] or candidate_y < 0 or candidate_y >= arr.shape[0]:
                continue

            # Do a grid check with euclidean distance (pythagoras)
            gx = int(candidate_x / cellSize)
            gy = int(candidate_y / cellSize)

            tooClose = False
            for nx in range(max(0, gx - 2), min(gridWidth, gx + 3)):
                for ny in range(max(0, gy - 2), min(gridHeight, gy + 3)):
                    neighbour_idx = grid[ny][nx]
                    if neighbour_idx == -1:
                        continue
                    neighbour = samples[neighbour_idx]
                    dx = candidate_x - neighbour[0]
                    dy = candidate_y - neighbour[1]
                    if math.sqrt(dx**2 + dy**2) < radius:
                        tooClose = True
                        break
                if tooClose:
                    break
            
            if not tooClose:
                samples.append((candidate_x, candidate_y))
                activeList.append((candidate_x, candidate_y))
                grid[gy][gx] = len(samples) - 1
                accepted = True
                break
        
        if not accepted:
            activeList.pop(idx)

    return samples

def gen_random_line(start_x, start_y, end_x, end_y):
    # start_x = random.randint(0, canvas_width)
    # start_y = random.randint(0, canvas_height)

    # end_x = start_x + length
    # end_y = start_y + length

    return draw.line(
        (start_x, start_y, end_x, end_y), 
        fill=(hsv2rgb(230, 1/100, random.choice(random_vibrance_value)/100)), 
        width=2
    )

arr = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

sample_points = BridsonPoissonDiskSampler(arr, 70)

samples_proj = []
for x, y in sample_points:
    accepted = True
    offPageY = False
    offPageX = False

    length = random.choice(random_lengths)
    candidateEnd_x = x + length
    candidateEnd_y = y + length

    if candidateEnd_x >= canvas_width:
        offPageX = True
    if candidateEnd_y >= canvas_height:
        offPageY = True

    if offPageX:
        newStartX = 0
        newStartY = y + (canvas_width - x) / (candidateEnd_x - x) * (candidateEnd_y - y)

    if offPageY:
        newStartY = 0
        newStartX = x + (canvas_height - y) / (candidateEnd_y - y) * (candidateEnd_x - x)

    if offPageX and offPageY:
        newStartX = 0
        newStartY = 0

    # projections, uses projection formulae at 45deg which causes simplifications, note: perp is perpindicular.
    candidate_start_proj = x + y
    candidate_end_proj = candidateEnd_x + candidateEnd_y
    candidate_perp_proj = x - y

    # Check is the projections overlap with all (need to use AND) of the existing projections
    for start_proj, end_proj, perp_proj in samples_proj:
        if abs(candidate_perp_proj - perp_proj) < threshold and candidate_start_proj < end_proj and start_proj < candidate_end_proj:
            accepted = False
            break
    
    if accepted:
        samples_proj.append((candidate_start_proj, candidate_end_proj, candidate_perp_proj))
        gen_random_line(x, y, candidateEnd_x, candidateEnd_y)

        if offPageX:
            gen_random_line(newStartX, newStartY, candidateEnd_x % canvas_width, candidateEnd_y)
        if offPageY:
            gen_random_line(newStartX, newStartY, candidateEnd_x, candidateEnd_y % canvas_height)
        if offPageX and offPageY:
            gen_random_line(newStartX, newStartY, candidateEnd_x % canvas_width, candidateEnd_y % canvas_height)

output_path = Path('assets/indexHeaderBackground.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
im.save(output_path, quality=100)


















# ------------
# Merged Image
# ------------

merged = Image.new('RGB', (canvas_width * 2, canvas_height * 2), "#0e1017")
merged.paste(im, (0, 0))
merged.paste(im, (canvas_width, 0))
merged.paste(im, (0, canvas_height))
merged.paste(im, (canvas_width, canvas_height))

output_path = Path('assets/indexHeaderBackground_merged.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
merged.save(output_path, quality=100)