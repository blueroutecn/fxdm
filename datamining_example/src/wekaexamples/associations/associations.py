import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.associations import Associator


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    vote_file = helper.get_data_dir() + os.sep + "vote.arff"
    helper.print_info("Loading dataset: " + vote_file)
    loader = Loader("weka.core.converters.ArffLoader")
    vote_data = loader.load_file(vote_file)
    vote_data.class_is_last()

    # train and output associator
    associator = Associator(classname="weka.associations.Apriori", options=["-N", "9", "-I"])
    associator.build_associations(vote_data)
    print(associator)

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
