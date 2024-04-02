# Genetic Algorithm - Monoalphabetic Cipher Breaker

Welcome to the Monoalphabetic Cipher Breaker project! This project aims to decrypt text encoded using a monoalphabetic cipher. We achieve this using a genetic algorithm that searches for the best solution to decode the text.

## Introduction

The genetic algorithm implemented in this project provides three options: regular genetic algorithm, Darwinian genetic algorithm, and Lamarckian genetic algorithm. Each option offers a unique approach to decoding the encrypted text.

## Input and Output Files

- **Input Files**:
  - `enc.txt`: Contains the encoded text that needs to be decrypted.
  - `dict.txt`: Contains a list of dictionary words for comparison during fitness calculation.
  - `Letter_Freq.txt`: Provides the frequency distribution of letters in the English language.

- **Output Files**:
  - `plain.txt`: Contains the decrypted text.
  - `perm.txt`: Contains the permutation table used for decoding.

## How to Run

To run the program, you have two options:

1. **Using the Executable File**: 
   - Download the attached directory named `main.zip`.
   - Extract the contents of the zip file.
   - Double-click on the `main.exe` program.
   - Choose the desired genetic algorithm type (1 for regular, 2 for Darwinian, 3 for Lamarckian).
   - Wait for the program to stop calculating.
   - The output files (`plain.txt` and `perm.txt`) will be shown in the same directory.

2. **Running the Python Script**: 
   - Place the input files (`enc.txt`, `dict.txt`, `Letter_Freq.txt`) in the same directory as the Python script.
   - Execute the Python script and follow the prompts.

## Example Demonstration of Use

To run the program using the executable file, simply double-click on `main.exe`:

![image](https://github.com/yeela8g/geneticAlgorithms/assets/118124478/9b2e289b-60fc-4bb3-9e05-1c0d13353cf6)


The program will prompt you to choose the desired genetic algorithm type:

![image](https://github.com/yeela8g/geneticAlgorithms/assets/118124478/54540932-e1e3-4b99-92a5-07f1c8eb0251)


After the program finishes calculating:

![image](https://github.com/yeela8g/geneticAlgorithms/assets/118124478/63f0174e-20a7-4303-bffe-331f79c638c4)


the output files will be displayed in the same directory:

![image](https://github.com/yeela8g/geneticAlgorithms/assets/118124478/dc5429d9-d5e0-48d3-abc7-39232e8406b3)


## The coding process

The implemented genetic algorithm follows the principles of modularity, reusability, and efficiency. The code is structured into modular functions and documentations, allowing for easy maintenance and extension. Additionally, reusable components such as fitness calculation and mutation functions enhance code readability and maintainability.

## Solution Calculation through Genetic Algorithm

+ In this genetic algorithm, each solution represents a permutation of the English alphabet, where each letter corresponds to a unique character. For example, a solution might look like "QWERTYUIOPASDFGHJKLZXCVBNM", meaning that the letter 'a' is encoded to Q, 'b' to W and so on by this order, where each letter can legally appear exactly once.

+ The population size is constantly maintained at 1000 individuals, and the initial population is randomly created.

+ The new population is created through several steps:

    + Selection of Parents for Crossover: A subset of the current population is selected as parents for crossover based on their fitness scores. The probability of selection is proportional to the fitness of each individual.
    + Crossover Operation: Pairs of parents are selected, and a random crossover point is chosen. Offspring are created by combining segments from both parents.
    + Mutation Operation: Mutation is applied to introduce diversity into the population. Random pairs of letters within each offspring are swapped to create mutations.
    + Elitism Mechanism: The best solutions from the current generation are preserved in the new population to ensure that the fittest individuals survive to the next generation.
    + Diversification Mechanism: To prevent premature convergence, additional mutations are applied to a subset of solutions if the maximum fitness remains unchanged for an extended period.
      
+ The fitness function evaluates each solution by comparing the decoded text with a dictionary of English words and considering the frequency distribution of letters. It calculates the fitness based on the number of recognized words in the decoded text and the similarity between the expected and actual letter frequencies.

## Lamarckian and Darwinian Algorithms

**Theoretical Principles:**
- **Lamarckian Algorithm**: In Lamarckian evolution, acquired traits are inherited by offspring. This means that the optimization process modifies the solution directly, and the modified solution is passed to the next generation. After generating a solution using the genetic algorithm's methods, the solution undergoes local optimizations. The fitness computation and passing to the next generation are done after optimization.
- **Darwinian Algorithm**: In Darwinian evolution, the fitness of each solution is evaluated before any optimization. The optimization process is then applied to improve the fitness of selected solutions, but the original solutions are passed to the next generation. Fitness computation is performed for each solution before optimization, but the optimized solution is passed to the next generation.

---
In conclusion, genetic algorithms offer a versatile and efficient approach to solving complex optimization problems. Their ability to iteratively refine solutions mimics natural selection, making them valuable tools across diverse fields. As computational capabilities advance, genetic algorithms stand poised to continue driving innovation and problem-solving in numerous domains.
