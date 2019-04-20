from random import randint
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

# generate a sequence of random integers


def generate_sequence(length, n_unique):
    return [randint(1, n_unique-1) for _ in range(length)]


def get_dataset():

    # generate source
    source = generate_sequence(10, 52)

    print("source : ==> ", source)
    # define padded target sequence
    target = source[:3]

    print("target : ==> ", target)

    target.reverse()

    print("target reverse : ==> ", target)

    target_in = [0] + target[:-1]

    print("target timestep : ==> ", target_in)

    src_encoded = to_categorical([0, 6], num_classes=52)
    tar_encoded = to_categorical([target], num_classes=52)
    tar2_encoded = to_categorical([target_in], num_classes=52)

    # print(src_encoded)
    # print(tar_encoded)
    print(tar2_encoded)


print(get_dataset())
