import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    bodyfat_file = helper.get_data_dir() + os.sep + "bodyfat.arff"
    helper.print_info("Loading dataset: " + bodyfat_file)
    loader = Loader("weka.core.converters.ArffLoader")
    bodyfat_data = loader.load_file(bodyfat_file)
    bodyfat_data.class_is_last()

    # classifier help
    helper.print_title("Creating help string")
    classifier = Classifier(classname="weka.classifiers.trees.M5P")
    classifier.build_classifier(bodyfat_data)
    print(classifier)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
