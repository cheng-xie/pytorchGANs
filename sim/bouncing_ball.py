
import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import math, sys, random


pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
running = True

### Physics stuff
space = pymunk.Space()
space.gravity = (0.0, -900.0)
draw_options = pymunk.pygame_util.DrawOptions(screen)

## Ball
mass = 10
radius = 10
inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
body = pymunk.Body(mass, inertia)
x = random.randint(115,350)
body.position = x, 400
shape = pymunk.Circle(body, radius, (0,0))
shape.elasticity = 0.95
shape.friction = 0.9
space.add(body, shape)
ball = shape

### walls
static_body = space.static_body
static_lines = [pymunk.Segment(static_body, (0.0, 380.0), (407.0, 246.0), 0.0)
                ,pymunk.Segment(static_body, (407.0, 246.0), (600.0, 443.0), 0.0)
                ]
for line in static_lines:
    line.elasticity = 0.95
    line.friction = 0.9
space.add(static_lines)

ticks_to_next_ball = 10

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            running = False
        elif event.type == KEYDOWN and event.key == K_p:
            pygame.image.save(screen, "bouncing_balls.png")

    ticks_to_next_ball -= 1
    if ticks_to_next_ball <= 0:
        ticks_to_next_ball = 100
        x = random.randint(115,350)
        ball.body.position = x, 400
        ball.body.velocity = random.randint(-250,250), random.randint(0,300)

    ### Clear screen
    screen.fill(THECOLORS["white"])

    space.debug_draw(draw_options)

    ### Update physics
    dt = 1.0/60.0
    for x in range(1):
        space.step(dt)

    ### Flip screen
    pygame.display.flip()
    clock.tick(0)
    pygame.display.set_caption("fps: " + str(clock.get_fps()))

