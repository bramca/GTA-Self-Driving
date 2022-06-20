import cv2
import os
import time
import numpy as np
import pyautogui
import pandas as pd
import random
from PIL import ImageGrab


def input_to_key_strokes(prediction):
    if prediction[0]:
        pyautogui.keyDown('z')
    else:
        pyautogui.keyUp('z')
    if prediction[1]:
        pyautogui.keyDown('d')
    else:
        pyautogui.keyUp('d')
    if prediction[2]:
        pyautogui.keyDown('s')
    else:
        pyautogui.keyUp('s')
    if prediction[3]:
        pyautogui.keyDown('q')
    else:
        pyautogui.keyUp('q')

def create_starting_population(individuals, chromosome_length):
    # Set up an initial array of all zeros
    population = np.zeros((individuals, chromosome_length))
    # Loop through each row (individual)
    for i in range(individuals):
        # Choose a random number of ones to create
        ones = random.randint(0, chromosome_length)
        # Change the required number of zeros to ones
        population[i, 0:ones] = 1
        # Sfuffle row
        np.random.shuffle(population[i])
    
    return population

def select_individual_by_tournament(population, scores):
    # Get population size
    population_size = len(scores)
    
    # Pick individuals for tournament
    fighter_1 = random.randint(0, population_size-1)
    fighter_2 = random.randint(0, population_size-1)
    
    # Get fitness score for each
    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]
    
    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2

    # Return the chromsome of the winner
    return population[winner, :]

def breed_by_crossover(parent_1, parent_2):
    # Get length of chromosome
    chromosome_length = len(parent_1)
    
    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1,chromosome_length-1)
    
    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                        parent_2[crossover_point:]))
    
    child_2 = np.hstack((parent_2[0:crossover_point],
                        parent_1[crossover_point:]))
    
    # Return children
    return child_1, child_2

def randomly_mutate_population(population, mutation_probability):
    # Apply random mutation
    random_mutation_array = np.random.random(size=(population.shape))
        
    random_mutation_boolean = random_mutation_array <= mutation_probability

    population[random_mutation_boolean] = np.logical_not(population[random_mutation_boolean])
        
    # Return mutation population
    return population

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def main():
    pyautogui.click(100, 100)
    winname = "screen"
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 800, 232)
    screen = np.ones(shape=(600, 799))
    oldscreen = np.zeros(shape=(600, 799))
    population_size = 20
    population = create_starting_population(population_size, 4)
    max_generation = 10
    pop_iterator = 0
    generation = 0
    frame_count = 0
    max_frame_count = 4
    score = 0
    scores = np.zeros(population_size)
    while True:
        if pop_iterator == population_size and generation < max_generation:
            # print(scores)
            new_population = []
            for i in range(int(population_size / 2)):
                parent_1 = select_individual_by_tournament(population, scores)
                parent_2 = select_individual_by_tournament(population, scores)
                child_1, child_2 = breed_by_crossover(parent_1, parent_2)
                new_population.append(child_1)
                new_population.append(child_2)
                
                # Replace the old population with the new one
            population = np.array(new_population)
                
                # Apply mutation
            mutation_rate = 0.1
            population = randomly_mutate_population(population, mutation_rate)
            print("generation %d best score: %.2f" % (generation, np.max(scores)))
            generation += 1
            pop_iterator = 0
        elif generation == max_generation:
            print("final_score: %.2f" % np.max(scores))
            cv2.destroyAllWindows()
            break

        keyboard_input = population[pop_iterator]
        input_to_key_strokes(keyboard_input)
        screen = np.array(ImageGrab.grab(bbox=(1, 30, 800, 630)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        frame_compare = np.absolute(np.array(screen) - np.array(oldscreen))
        
        if frame_count == max_frame_count:
            score /= 3
            score /= (600 * 799)
            score *= 100
            scores[pop_iterator] = score
            pop_iterator += 1
            frame_count = 0
      
            
        if frame_count > 0:
            score += np.count_nonzero(frame_compare != 0)
            
        cv2.imshow(winname, screen)
        oldscreen = screen
        frame_count += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
