import random
import asyncio
import logging
from datetime import datetime, timedelta
from collections import namedtuple
import matplotlib.pyplot as plt
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define structures for rockets, payloads, weather, and crew using namedtuples for clarity
Rocket = namedtuple('Rocket', ['name', 'available_from'])
Payload = namedtuple('Payload', ['name', 'ready_by'])
Weather = namedtuple('Weather', ['date', 'condition'])
Crew = namedtuple('Crew', ['name', 'available_on'])

# Classes for Launch Windows and Schedules
class LaunchWindow:
    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end

class Schedule:
    def __init__(self, launch_time: datetime, rocket: Rocket, payload: Payload, crew: Crew):
        self.launch_time = launch_time
        self.rocket = rocket
        self.payload = payload
        self.crew = crew

# Define the launch windows
launch_windows = [
    LaunchWindow(datetime(2024, 9, 1, 6, 0), datetime(2024, 9, 1, 12, 0)),
    LaunchWindow(datetime(2024, 9, 2, 6, 0), datetime(2024, 9, 2, 12, 0)),
    LaunchWindow(datetime(2024, 9, 3, 6, 0), datetime(2024, 9, 3, 12, 0)),
]

# Define the rockets, payloads, weather conditions, and crew
rockets = [
    Rocket('New Shepard', datetime(2024, 9, 1, 0, 0)),
    Rocket('New Glenn', datetime(2024, 9, 2, 0, 0)),
]

payloads = [
    Payload('Satellite A', datetime(2024, 9, 1, 6, 0)),
    Payload('Crew Capsule B', datetime(2024, 9, 2, 6, 0)),
]

weather_conditions = [
    Weather(datetime(2024, 9, 1), 'favorable'),
    Weather(datetime(2024, 9, 2), 'unfavorable'),
    Weather(datetime(2024, 9, 3), 'favorable'),
]

crew_availability = [
    Crew('Crew A', datetime(2024, 9, 1)),
    Crew('Crew B', datetime(2024, 9, 2)),
]

# Function to check if a launch window is valid
def is_valid_launch_window(rocket: Rocket, payload: Payload, weather: Weather, crew: Crew, launch_window: LaunchWindow) -> bool:
    logging.info(f"Checking if {rocket.name} can launch with {payload.name} at {launch_window.start}")
    if rocket.available_from > launch_window.start:
        return False
    if payload.ready_by > launch_window.start:
        return False
    if weather.condition != 'favorable':
        return False
    if crew.available_on.date() != launch_window.start.date():
        return False
    return True

# Genetic Algorithm for Optimization
async def optimize_launch_schedule(launch_windows: List[LaunchWindow], rockets: List[Rocket], payloads: List[Payload], 
                                   weather_conditions: List[Weather], crew_availability: List[Crew]) -> List[Schedule]:
    population_size = 50
    generations = 100
    mutation_rate = 0.1

    def fitness(schedule: List[Schedule]) -> int:
        return sum(is_valid_launch_window(launch.rocket, launch.payload, 
                                          weather_conditions[launch.launch_time.day - 1], 
                                          launch.crew, launch_windows[launch.launch_time.day - 1]) 
                   for launch in schedule)

    def generate_population() -> List[List[Schedule]]:
        return [[Schedule(random.choice(launch_windows).start, random.choice(rockets), random.choice(payloads), random.choice(crew_availability)) 
                 for _ in launch_windows] for _ in range(population_size)]

    def mutate(schedule: List[Schedule]) -> List[Schedule]:
        if random.random() < mutation_rate:
            i = random.randint(0, len(schedule) - 1)
            schedule[i] = Schedule(random.choice(launch_windows).start, random.choice(rockets), random.choice(payloads), random.choice(crew_availability))
        return schedule

    def crossover(schedule1: List[Schedule], schedule2: List[Schedule]) -> List[Schedule]:
        crossover_point = random.randint(0, len(schedule1) - 1)
        return schedule1[:crossover_point] + schedule2[crossover_point:]

    population = generate_population()

    for generation in range(generations):
        logging.info(f"Generation {generation+1}/{generations}")
        population = sorted(population, key=lambda s: fitness(s), reverse=True)
        next_population = population[:population_size//2]

        for i in range(population_size // 2, population_size):
            parent1 = random.choice(next_population)
            parent2 = random.choice(next_population)
            child = mutate(crossover(parent1, parent2))
            next_population.append(child)
        
        population = next_population

    best_schedule = sorted(population, key=lambda s: fitness(s), reverse=True)[0]
    return best_schedule

# Visualization of the schedule using Gantt Chart
def visualize_schedule(schedule: List[Schedule]):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_labels = [launch.rocket.name for launch in schedule]
    start_times = [launch.launch_time for launch in schedule]
    durations = [1 for _ in schedule]  # Each launch is 1 hour for simplicity
    ax.barh(y_labels, durations, left=start_times, height=0.5, color='skyblue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rocket')
    ax.set_title('Launch Schedule Optimization')
    plt.show()

# Main function to run the optimization
async def main():
    best_schedule = await optimize_launch_schedule(launch_windows, rockets, payloads, weather_conditions, crew_availability)
    logging.info("Best schedule found:")
    for launch in best_schedule:
        logging.info(f"Launch Time: {launch.launch_time}, Rocket: {launch.rocket.name}, Payload: {launch.payload.name}, Crew: {launch.crew.name}")
    
    visualize_schedule(best_schedule)

# Run the main function
asyncio.run(main())
