import math
import random
import sys
import os
import pickle
import neat
import pygame

# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) # color to crash on hit

MAX_GEN = 50 # number of simulations to run

current_generation = 0 # generation counter

map_img_path = "img/map1.png"
car_img_path = "img/car.png"

class Car:
    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load(car_img_path).convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 

        self.position = [830, 920] # starting position
        self.angle = 0
        self.speed = 0

        self.speed_set = False # flag for default speed later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] # calculate center

        self.radars = [] # list for sensors / radars
        self.drawing_radars = [] # radars to be drawn

        self.alive = True # for checking if car is crashed

        self.distance = 0 # distance driven
        self.time = 0 # time passed

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) # draw sprite
        self.draw_radar(screen) # OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # optionally draw all sensors / radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # if any corner touches border color -> crash
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # while we don't hit BORDER_COLOR and length < 300 (just a max) -> go farther
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # calculate distance to border and append to radars list
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        # set the speed to 20 for the first time
        # only when having 4 output nodes with speed up and down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # get rotated sprite and move into the right x-direction
        # don't let the car go closer than 20px to the edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # increase distance and time
        self.distance += self.speed
        self.time += 1
        
        # same for y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # calculate new center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # calculate 4 corners
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # check collisions and clear radars
        self.check_collision(game_map)
        self.radars.clear()

        # from -90 to 120 with step-size 45 check radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # get distance to border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        # calculate reward
        # return self.distance / 50.0
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def run_simulation(genomes, config):
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # for all genomes passed create a new NN
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load(map_img_path).convert()

    global current_generation
    current_generation += 1

    # counter to limit time
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # get actions each car can take
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10 # left
            elif choice == 1:
                car.angle -= 10 # right
            elif choice == 2:
                if(car.speed - 2 >= 12):
                    car.speed -= 2 # slow down
            else:
                car.speed += 2 # speed up
        
        # check if car is alive
        # increase fitness if true And break loop if not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40: # stop after 20 seconds
            break

        # draw map And all cars that are alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(500) 

def run_neat(config):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    ppl = neat.Population(config)
    ppl.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    ppl.add_reporter(stats)
    ppl.add_reporter(neat.Checkpointer(10))
    
    best = ppl.run(run_simulation, MAX_GEN)

    with open("best.pickle", "wb") as f:
        pickle.dump(best, f)

def test_best_model(config):
    # load best genome
    with open('best.pickle', 'rb') as f:
        best = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config)
    
    # create the neural network from best genome
    net = neat.nn.FeedForwardNetwork.create(best, config)
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    game_map = pygame.image.load(map_img_path).convert()
    car = Car()

    while car.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        output = net.activate(car.get_data())
        choice = output.index(max(output))
        if choice == 0:
            car.angle += 10  # left
        elif choice == 1:
            car.angle -= 10  # right
        elif choice == 2:
            if car.speed - 2 >= 12:
                car.speed -= 2  # slow down
        else:
            car.speed += 2  # speed up
        
        car.update(game_map)

        # draw everything
        screen.blit(game_map, (0, 0))
        car.draw(screen)

        text = generation_font.render("Best model generation: " + str(MAX_GEN), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (900, 500)
        screen.blit(text, text_rect)
        
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    config_path = "./config.txt"

    #run_neat(config_path)
    test_best_model(config_path)
