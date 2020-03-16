import sys
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier


def main(args):
    """
    Trains a J48 classifier on a training set and outputs the predicted class and class distribution alongside the
    actual class from a test set. Class attribute is assumed to be the last attribute.
    :param args: the commandline arguments (train and test datasets)
    :type args: list
    """

    # load a dataset
    helper.print_info("Loading train: " + args[1])
    loader = Loader(classname="weka.core.converters.ArffLoader")
    train = loader.load_file(args[1])
    train.class_index = train.num_attributes - 1
    helper.print_info("Loading test: " + args[2])
    test = loader.load_file(args[2])
    test.class_is_last()

    # classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)

    # output predictions
    print("# - actual - predicted - error - distribution")
    for index, inst in enumerate(test):
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        print(
            "%d - %s - %s - %s  - %s" %
            (index+1,
             inst.get_string_value(inst.class_index),
             inst.class_attribute.value(int(pred)),
             "yes" if pred != inst.get_value(inst.class_index) else "no",
             str(dist.tolist())))


if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
