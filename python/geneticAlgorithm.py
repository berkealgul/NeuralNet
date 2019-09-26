import random
import math
from car import Car


mutationRate = 0.01


# ana fonksiyon
def create_new_generation(oldGeneration, generationlen=None):
    if generationlen is None:
        generationlen = len(oldGeneration)

    best_fitness = pick_best_fitness(oldGeneration)

    clear_bad_ones(oldGeneration, best_fitness/10)

    calculate_fitness(oldGeneration)
    newGeneration = []

    # mutasyon ihtimali optimize edilmeye çalışılır
    optimize_mutationRate(best_fitness)

    for i in range(generationlen):
        mom = choose_parent(oldGeneration)
        dad = choose_parent(oldGeneration)
        child = crossover(mom, dad)
        newGeneration.append(child)

    return newGeneration


# yardımcı fonksiyonlar
def calculate_fitness(generation):
    sum = 0
    for member in generation:
        member.fitness = math.pow(member.score, 2)
        sum += member.fitness
    for member in generation:
        member.fitness /= sum


def choose_parent(generation):
    r = random.random()
    index = 0
    while r > 0:
        r -= generation[index].fitness
        index += 1
    index -= 1
    return generation[index]


def crossover(parent1, parent2):
    # beyin nöron ağı objesidir
    brain1 = parent1.brain.copy()
    brain2 = parent2.brain.copy()
    # W: ağırlık F: fitness
    # Wçocuk = (Wbaba * Fbaba + Wana * Fana) / (Fana + Fbaba)
    # aynı kural sapmalar içinde uygulanabilir

    # iki ebebeyn de aynı sayıda nöron ve katmana sahip olduğu için
    # indeks hatası ile uğraşmamız gerekmiyecek
    for i in range(len(brain1.weights)):
        Wp1 = brain1.weights[i]
        Wp2 = brain2.weights[i]
        Bp1 = brain1.biases[i]
        Bp2 = brain2.biases[i]

        # ağırlıkları ve sapmaları ebebeynlerin fitnesslariyla çarparız
        Wp1.multiply(parent1.fitness)
        Wp2.multiply(parent2.fitness)
        Bp1.multiply(parent1.fitness)
        Bp2.multiply(parent2.fitness)

        # işlemlerimizi brain1 üzerinden yapacağız
        # çocuğada brain1 objesini vereceğiz
        Wp1.add(Wp2)
        Wp1.multiply(1 / (parent1.fitness + parent2.fitness))
        Bp1.add(Bp2)
        Bp1.multiply(1 / (parent1.fitness + parent2.fitness))

    brain1.mutate(mutationRate)

    child = Car(brain=brain1)
    return child


last_gen_best = None
counter = 0
limit = 3
similarityTreshold = 0.05
initialMutationRate = mutationRate


# kodu temizlemelisin !!!
def optimize_mutationRate(best_score):
    global counter, last_gen_best, mutationRate

    if last_gen_best is None:
        last_gen_best = best_score
        return

    df = math.fabs(best_score - last_gen_best)
    print(str(df))

    if df < similarityTreshold:
        print('benzerlik')
        # eğer benzerlik 'limit' kere tekrarlanırsa aksiyona geçilir
        if counter < limit:
            counter += 1
        else:
            # eğer hala nesiller arası performans değişmiyorsa mutasyon ihtimali arttırılır
            if mutationRate < 0.04:
                 mutationRate += 0.01
            print('mutasyon değeri arttı ' + str(mutationRate))
            last_gen_best = best_score
    else:
        # eğer nesil arası performans farkı gözetilirse parametreler sıfırlanır
        if counter == limit:
            print('sıfırlandı')
            counter = 0
            mutationRate = initialMutationRate
        last_gen_best = best_score

    last_gen_best = best_score


def pick_best_fitness(generation):
    bestInd = 0

    for i in range(len(generation)):
        if generation[bestInd].fitness < generation[i].fitness:
            bestInd = i

    return generation[i].fitness


def clear_bad_ones(generation, fitnessTrehsold):
    for ind in generation:
        if ind.fitness < fitnessTrehsold:
            generation.remove(ind)
