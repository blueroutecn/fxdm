import os
import sys
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier


def main(args):
    """
    Trains a NaiveBayesUpdateable classifier incrementally on a dataset. The dataset can be supplied as parameter.
    :param args: the commandline arguments
    :type args: list
    """

    # load a dataset
    if len(args) <= 1:
        data_file = helper.get_data_dir() + os.sep + "vote.arff"
    else:
        data_file = args[1]
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(data_file, incremental=True)
    data.class_is_last()

    # classifier
    nb = Classifier(classname="weka.classifiers.bayes.NaiveBayesUpdateable")
    nb.build_classifier(data)

    # train incrementally
    for inst in loader:
        nb.update_classifier(inst)

    print(nb)


if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
