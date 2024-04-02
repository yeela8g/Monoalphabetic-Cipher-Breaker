import multiprocessing
import random
import string
import os , sys
application_path = os.path.dirname(sys.executable)
# Genetic Algorithm parameters
POPULATION_SIZE = 1000


# Fitness function
def calculate_fitness(population, encoded_text, dictionary_words, letter_distribution):
    fitness = []
    for solution in population:
        decoded_text = encoded_text.translate(str.maketrans(string.ascii_uppercase, solution))

        # Calculate the number of letters in the decoded text
        num_letters = len(decoded_text)

        # Calculate the expected letter counts based on the distribution
        expected_counts = {letter: round(num_letters * freq) for letter, freq in letter_distribution.items()}

        # Count the actual occurrences of each letter in the decoded text
        actual_counts = {letter: decoded_text.count(letter) for letter in string.ascii_uppercase}

        # Calculate the difference between the expected and actual counts
        diff_sum = sum(abs(expected_counts[letter] - actual_counts[letter]) for letter in string.ascii_uppercase)

        # Calculate fitness based on the number of words found in the dictionary and the difference sum
        fitness_tmp = sum(word.upper() in dictionary_words for word in decoded_text.split())

        if fitness_tmp - diff_sum > 0:
            fitness_tmp -= diff_sum
        fitness.append(fitness_tmp)

    return fitness


# Mutation function
def mutate(solution):
    solution_list = list(solution)
    index_a, index_b = random.sample(range(26), 2)
    solution_list[index_a], solution_list[index_b] = solution_list[index_b], solution_list[index_a]
    return ''.join(solution_list)


def validate_solution(solution):
    # Check for alphabet repetition
    if len(set(solution)) != 26:
        missing_letters = ''.join(sorted(set(string.ascii_uppercase) - set(solution)))
        for i in range(len(solution)):
            if solution.count(solution[i]) > 1:
                solution = solution[:i] + missing_letters[0] + solution[i + 1:]
                missing_letters = missing_letters[1:]
    return solution


# Genetic Algorithm
def genetic_algorithm(encoded_text, dictionary_words, letter_distribution):
    counter = 0
    population = [''.join(random.sample(string.ascii_uppercase, 26)) for _ in range(POPULATION_SIZE)]
    new_population = [''] * POPULATION_SIZE
    fitness_scores = [0] * POPULATION_SIZE  # Initialize fitness scores array
    unchanged_generations = 0  # Counter for unchanged maximum fitness generations

    # Arrays to store fitness statistics for each generation
    avg_fitness_scores = []
    max_fitness_scores = []
    min_fitness_scores = []

    while unchanged_generations <= 150:
        # Calculate fitness for each solution
        fitness_scores = calculate_fitness(population, encoded_text, dictionary_words, letter_distribution)
        print("generation number: ",counter)
        counter += 1

        # Store fitness statistics for the current generation
        avg_fitness_scores.append(sum(fitness_scores) / len(fitness_scores))
        max_fitness_scores.append(max(fitness_scores))
        min_fitness_scores.append(min(fitness_scores))

        # Select parents for crossover
        num_parents = int(POPULATION_SIZE * 0.6)
        parents = random.choices(population, weights=fitness_scores, k=num_parents)

        # Perform crossover
        for i in range(num_parents):
            parent1, parent2 = random.choices(parents, k=2)
            crossover_point = random.randint(1, 25)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            child = validate_solution(child)
            new_population[i] = child

        # Perform mutation on new population
        for i in range(0, num_parents):
            new_population[i] = mutate(new_population[i])
            new_population[i] = mutate(new_population[i])

        if unchanged_generations > 20:
            for i in range(0, num_parents):
                new_population[i] = mutate(new_population[i])

        # Add 20% of the best solutions to new population
        num_best_solutions = int(POPULATION_SIZE * 0.2)
        best_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[
                       :num_best_solutions]

        # Select 25% of the best solutions to save for the next generation
        num_save_solutions = int(num_best_solutions * 0.25)
        save_indices = best_indices[:num_save_solutions]
        save_solutions = [population[i] for i in save_indices]

        # Select 75% of the best solutions for mutation
        num_mutations = int(num_best_solutions * 0.75)
        mutation_indices = best_indices[num_save_solutions:num_save_solutions + num_mutations]
        mutation_solutions = [population[i] for i in mutation_indices]

        # Perform mutation on the selected solutions
        for i in range(num_mutations):
            mutation_solutions[i] = mutate(mutation_solutions[i])
            mutation_solutions[i] = mutate(mutation_solutions[i])

        if unchanged_generations > 20:
            for i in range(num_mutations):
                mutation_solutions[i] = mutate(mutation_solutions[i])

        # Add saved, mutated solutions to new population
        new_population[num_parents:num_parents + num_save_solutions] = save_solutions
        new_population[
        num_parents + num_save_solutions:num_parents + num_save_solutions + num_mutations] = mutation_solutions

        # Add 20% of new randomly generated permutations to new population
        num_random_permutations = int(POPULATION_SIZE * 0.2)
        random_permutations = [''.join(random.sample(string.ascii_uppercase, 26)) for _ in
                               range(num_random_permutations)]
        new_population[num_parents + num_best_solutions:] = random_permutations

        # Update the population with the new population
        population = new_population[:]

        if len(max_fitness_scores) > 2:
            if max_fitness_scores[-1] == max_fitness_scores[-2]:  # Check if maximum fitness is unchanged
                unchanged_generations += 1
            else:
                unchanged_generations = 0  # Reset the counter if maximum fitness changes

    fitness_scores = calculate_fitness(population, encoded_text, dictionary_words, letter_distribution)
    counter += 1
    best_solution_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    best_solution = population[best_solution_index]
    # Save best solution to perm.txt
    with open("perm.txt", "w") as perm_file:
        for i in range(26):
            perm_file.write(string.ascii_lowercase[i] + " " + best_solution[i].lower() + "\n")

    # Decode encoded text using the best solution and save to plain.txt
    decoded_text = encoded_text.translate(str.maketrans(string.ascii_uppercase, best_solution.lower()))
    with open("plain.txt", "w") as plain_file:
        plain_file.write(decoded_text)

    return avg_fitness_scores, max_fitness_scores, min_fitness_scores, counter


def compare_fitness_scores(population, optimize_population, fitness_scores, fitness_scores_optimize):
    fitness_score_uniun = []
    uniun_population = []

    for i in range(len(fitness_scores)):
        if fitness_scores[i] >= fitness_scores_optimize[i]:
            fitness_score_uniun.append(fitness_scores[i])
            uniun_population.append(population[i])
        else:
            fitness_score_uniun.append(fitness_scores_optimize[i])
            uniun_population.append(optimize_population[i])

    return uniun_population, fitness_score_uniun


def darwin_algorithm(encoded_text, dictionary_words, letter_distribution):
    counter = 0
    population = [''.join(random.sample(string.ascii_uppercase, 26)) for _ in range(POPULATION_SIZE)]
    new_population = [''] * POPULATION_SIZE
    fitness_scores = [0] * POPULATION_SIZE  # Initialize fitness scores array
    unchanged_generations = 0  # Counter for unchanged maximum fitness generations
    optimized_population = []

    # Arrays to store fitness statistics for each generation
    avg_fitness_scores = []
    max_fitness_scores = []
    min_fitness_scores = []
    while unchanged_generations <= 150:
        # Calculate fitness for each solution
        fitness_scores = calculate_fitness(population, encoded_text, dictionary_words, letter_distribution)
        print("generation number: ",counter)

        # Store fitness statistics for the current generation
        avg_fitness_scores.append(sum(fitness_scores) / len(fitness_scores))
        max_fitness_scores.append(max(fitness_scores))
        min_fitness_scores.append(min(fitness_scores))

        for i in range(POPULATION_SIZE):  # optimization , N=1
            optimized_population.append(mutate(population[i]))

        fitness_scores_optimized = calculate_fitness(optimized_population, encoded_text, dictionary_words,
                                                     letter_distribution)
        counter += 1
        optimized_population, fitness_scores_optimized = compare_fitness_scores(population, optimized_population,
                                                                                fitness_scores,
                                                                                fitness_scores_optimized)

        # Select parents for crossover
        num_parents = int(POPULATION_SIZE * 0.6)
        parents = random.choices(population, weights=fitness_scores_optimized, k=num_parents)

        # Perform crossover
        for i in range(num_parents):
            parent1, parent2 = random.choices(parents, k=2)
            crossover_point = random.randint(1, 25)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            child = validate_solution(child)
            new_population[i] = child

        # Perform mutation on new population
        for i in range(0, num_parents):
            new_population[i] = mutate(new_population[i])
            new_population[i] = mutate(new_population[i])

        if unchanged_generations > 20:
            for i in range(0, num_parents):
                new_population[i] = mutate(new_population[i])

        # Add 20% of the best solutions to new population
        num_best_solutions = int(POPULATION_SIZE * 0.2)
        best_indices = sorted(range(len(fitness_scores_optimized)), key=lambda k: fitness_scores_optimized[k], reverse=True)[
                       :num_best_solutions]

        # Select 25% of the best solutions to save for the next generation
        num_save_solutions = int(num_best_solutions * 0.25)
        save_indices = best_indices[:num_save_solutions]
        save_solutions = [population[i] for i in save_indices]

        # Select 75% of the best solutions for mutation
        num_mutations = int(num_best_solutions * 0.75)
        mutation_indices = best_indices[num_save_solutions:num_save_solutions + num_mutations]
        mutation_solutions = [population[i] for i in mutation_indices]

        # Perform mutation on the selected solutions
        for i in range(num_mutations):
            mutation_solutions[i] = mutate(mutation_solutions[i])
            mutation_solutions[i] = mutate(mutation_solutions[i])

        if unchanged_generations > 20:
            for i in range(num_mutations):
                mutation_solutions[i] = mutate(mutation_solutions[i])

        # Add saved, mutated solutions to new population
        new_population[num_parents:num_parents + num_save_solutions] = save_solutions
        new_population[
        num_parents + num_save_solutions:num_parents + num_save_solutions + num_mutations] = mutation_solutions

        # Add 20% of new randomly generated permutations to new population
        num_random_permutations = int(POPULATION_SIZE * 0.2)
        random_permutations = [''.join(random.sample(string.ascii_uppercase, 26)) for _ in
                               range(num_random_permutations)]
        new_population[num_parents + num_best_solutions:] = random_permutations

        best_solution_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        new_population[-1] = population[best_solution_index] # override with the best solution to ensure monotonic function

        # Update the population with the new population
        population = new_population[:]

        if len(max_fitness_scores) > 2:
            if max_fitness_scores[-1] == max_fitness_scores[-2]:  # Check if maximum fitness is unchanged
                unchanged_generations += 1
            else:
                unchanged_generations = 0  # Reset the counter if maximum fitness changes

    fitness_scores = calculate_fitness(population, encoded_text, dictionary_words, letter_distribution)
    counter += 1
    best_solution_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    best_solution = population[best_solution_index]
    # Save best solution to perm.txt
    with open("perm.txt", "w") as perm_file:
        for i in range(26):
            perm_file.write(string.ascii_lowercase[i] + " " + best_solution[i].lower() + "\n")

    # Decode encoded text using the best solution and save to plain.txt
    decoded_text = encoded_text.translate(str.maketrans(string.ascii_uppercase, best_solution.lower()))
    with open("plain.txt", "w") as plain_file:
        plain_file.write(decoded_text)

    return avg_fitness_scores, max_fitness_scores, min_fitness_scores, counter


def lamarck_algorithm(encoded_text, dictionary_words, letter_distribution):
    counter = 0
    population = [''.join(random.sample(string.ascii_uppercase, 26)) for _ in range(POPULATION_SIZE)]
    new_population = [''] * POPULATION_SIZE
    fitness_scores = [0] * POPULATION_SIZE  # Initialize fitness scores array
    unchanged_generations = 0  # Counter for unchanged maximum fitness generations
    optimized_population = []

    # Arrays to store fitness statistics for each generation
    avg_fitness_scores = []
    max_fitness_scores = []
    min_fitness_scores = []
    while unchanged_generations <= 150:
        # Calculate fitness for each solution
        fitness_scores = calculate_fitness(population, encoded_text, dictionary_words, letter_distribution)
        print("generation number: ",counter)
        # Store fitness statistics for the current generation
        avg_fitness_scores.append(sum(fitness_scores) / len(fitness_scores))
        max_fitness_scores.append(max(fitness_scores))
        min_fitness_scores.append(min(fitness_scores))

        for i in range(POPULATION_SIZE):  # optimization , N=2
            temp_s = mutate(population[i])
            temp_s = mutate(temp_s)
            optimized_population.append(temp_s)
        fitness_scores_optimized = calculate_fitness(optimized_population, encoded_text, dictionary_words,
                                                     letter_distribution)
        counter += 1
        optimized_population, fitness_scores_optimized = compare_fitness_scores(population, optimized_population,
                                                                                fitness_scores,
                                                                                fitness_scores_optimized)

        # Select parents for crossover
        num_parents = int(POPULATION_SIZE * 0.6)
        parents = random.choices(optimized_population, weights=fitness_scores_optimized, k=num_parents)

        # Perform crossover
        for i in range(num_parents):
            parent1, parent2 = random.choices(parents, k=2)
            crossover_point = random.randint(1, 25)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            child = validate_solution(child)
            new_population[i] = child

        # Perform mutation on new population
        for i in range(0, num_parents):
            new_population[i] = mutate(new_population[i])
            new_population[i] = mutate(new_population[i])

        if unchanged_generations > 20:
            for i in range(0, num_parents):
                new_population[i] = mutate(new_population[i])

        # Add 20% of the best solutions to new population
        num_best_solutions = int(POPULATION_SIZE * 0.2)
        best_indices = sorted(range(len(fitness_scores_optimized)), key=lambda k: fitness_scores_optimized[k], reverse=True)[
                       :num_best_solutions]

        # Select 25% of the best solutions to save for the next generation
        num_save_solutions = int(num_best_solutions * 0.25)
        save_indices = best_indices[:num_save_solutions]
        save_solutions = [optimized_population[i] for i in save_indices]

        # Select 75% of the best solutions for mutation
        num_mutations = int(num_best_solutions * 0.75)
        mutation_indices = best_indices[num_save_solutions:num_save_solutions + num_mutations]
        mutation_solutions = [optimized_population[i] for i in mutation_indices]

        # Perform mutation on the selected solutions
        for i in range(num_mutations):
            mutation_solutions[i] = mutate(mutation_solutions[i])
            mutation_solutions[i] = mutate(mutation_solutions[i])

        if unchanged_generations > 20:
            for i in range(num_mutations):
                mutation_solutions[i] = mutate(mutation_solutions[i])

        # Add saved, mutated solutions to new population
        new_population[num_parents:num_parents + num_save_solutions] = save_solutions
        new_population[
        num_parents + num_save_solutions:num_parents + num_save_solutions + num_mutations] = mutation_solutions

        # Add 20% of new randomly generated permutations to new population
        num_random_permutations = int(POPULATION_SIZE * 0.2)
        random_permutations = [''.join(random.sample(string.ascii_uppercase, 26)) for _ in
                               range(num_random_permutations)]
        new_population[num_parents + num_best_solutions:] = random_permutations

        # Update the population with the new population
        population = new_population[:]

        if len(max_fitness_scores) > 2:
            if max_fitness_scores[-1] == max_fitness_scores[-2]:  # Check if maximum fitness is unchanged
                unchanged_generations += 1
            else:
                unchanged_generations = 0  # Reset the counter if maximum fitness changes

    fitness_scores = calculate_fitness(population, encoded_text, dictionary_words, letter_distribution)
    counter += 1
    best_solution_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    best_solution = population[best_solution_index]
    # Save best solution to perm.txt
    with open("perm.txt", "w") as perm_file:
        for i in range(26):
            perm_file.write(string.ascii_lowercase[i] + " " + best_solution[i].lower() + "\n")

    # Decode encoded text using the best solution and save to plain.txt
    decoded_text = encoded_text.translate(str.maketrans(string.ascii_uppercase, best_solution.lower()))
    with open("plain.txt", "w") as plain_file:
        plain_file.write(decoded_text)

    return avg_fitness_scores, max_fitness_scores, min_fitness_scores, counter


def run_genetic_algorithm(encoded_text, dictionary_words, letter_distribution, program_choice):
    if program_choice == "1":
        return genetic_algorithm(encoded_text, dictionary_words, letter_distribution)
    elif program_choice == "2":
        return darwin_algorithm(encoded_text,dictionary_words, letter_distribution)
    elif program_choice == "3":
        return lamarck_algorithm(encoded_text,dictionary_words,letter_distribution)


# Main program
if __name__ == '__main__':
    with open('enc.txt', 'r') as file:
        encoded_text = file.read().upper()

    # Load the dictionary file
    with open('dict.txt', 'r') as file:
        dictionary_words = set(file.read().upper().split('\n'))

    letter_distribution = {}
    with open('Letter_Freq.txt', 'r') as file:
        for line in file:
            freq, letter = line.strip().split('\t')
            letter_distribution[letter.strip()] = float(freq)

    # Ask the user for the program choice
    program_choice = input(
        "Which program do you want to run?\nFor genetic_algorithm enter 1\nFor darwin_algorithm enter 2\nFor lamarck_algorithm enter 3:")
    if program_choice != "1" and  program_choice != "2" and program_choice != "3":
        sys.exit()
    else:
        # num_executions = 1
        # pool = multiprocessing.Pool(processes=num_executions)
        # results = [pool.apply_async(run_genetic_algorithm, (encoded_text, dictionary_words, letter_distribution, program_choice)) for _ in
        #            range(num_executions)]
        # pool.close()
        # pool.join()

        results = run_genetic_algorithm(encoded_text, dictionary_words, letter_distribution, program_choice)

        avg_fitness_scores = []
        max_fitness_scores = []
        min_fitness_scores = []

        for result in results:
            avg_scores, max_scores, min_scores, counter = result.get()
            avg_fitness_scores.append(avg_scores)
            max_fitness_scores.append(max_scores)
            min_fitness_scores.append(min_scores)
