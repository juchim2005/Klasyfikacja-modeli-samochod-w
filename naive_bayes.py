import csv
import random
import math

def load_csv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [str(x) for x in dataset[i]]
    return dataset

def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return [train_set, test_set]

def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries

def calculate_probability(x, mean, stdev):
    if stdev == 0:
        if x == mean:
            return 1
        else:
            return 0
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions

def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0

def get_attribute_value_counts(dataset):
    counts = {}
    for vector in dataset:
        for i, value in enumerate(vector[:-1]):
            if i not in counts:
                counts[i] = {}
            if value not in counts[i]:
                counts[i][value] = 0
            counts[i][value] += 1
    return counts

def calculate_conditional_probabilities(dataset, class_summaries, attribute_value_counts):
    separated = separate_by_class(dataset)
    probabilities = {}
    for class_value, instances in separated.items():
        probabilities[class_value] = {}
        for i in range(len(instances[0]) - 1):
            probabilities[class_value][i] = {}
            for value in attribute_value_counts[i]:
                count = 0
                for instance in instances:
                    if instance[i] == value:
                        count += 1
                probabilities[class_value][i][value] = (count + 1) / (len(instances) + len(attribute_value_counts[i]))
    return probabilities

def calculate_class_probabilities_categorical(class_summaries, conditional_probabilities, input_vector):
    probabilities = {}
    total_count = sum([class_summaries[label] for label in class_summaries])
    for class_value, class_count in class_summaries.items():
        probabilities[class_value] = class_count / total_count
        for i in range(len(input_vector) - 1):
            value = input_vector[i]
            if value in conditional_probabilities[class_value][i]:
                probabilities[class_value] *= conditional_probabilities[class_value][i][value]
            else:
                probabilities[class_value] *= 1 / (class_count + len(conditional_probabilities[class_value][i]))
    return probabilities


def predict_categorical(class_summaries, conditional_probabilities, input_vector):
    probabilities = calculate_class_probabilities_categorical(class_summaries, conditional_probabilities, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def main():
    filename = 'car_evaluation.data'
    split_ratio = 0.8
    dataset = load_csv(filename)

    training_set, test_set = split_dataset(dataset, split_ratio)

    separated_by_class_training = separate_by_class(training_set)
    class_summaries = {class_value: len(instances) for class_value, instances in separated_by_class_training.items()}
    attribute_value_counts = get_attribute_value_counts(training_set)

    conditional_probabilities = calculate_conditional_probabilities(training_set, class_summaries, attribute_value_counts)

    predictions = []
    for row in test_set:
        prediction = predict_categorical(class_summaries, conditional_probabilities, row)
        predictions.append(prediction)

    accuracy = get_accuracy(test_set, predictions)
    print(f'Accuracy: {accuracy}%')

if __name__ == '__main__':
    main()