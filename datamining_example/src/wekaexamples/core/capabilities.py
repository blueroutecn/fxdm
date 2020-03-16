import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.capabilities import Capability
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.core.capabilities import Capabilities


def main():
    """
    Just runs some example code.
    """

    classifier = Classifier("weka.classifiers.trees.J48")

    helper.print_title("Capabilities")
    capabilities = classifier.capabilities
    print(capabilities)

    # load a dataset
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    loader = Loader("weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()
    data_capabilities = Capabilities.for_instances(iris_data)
    print(data_capabilities)
    print("classifier handles dataset: " + str(capabilities.supports(data_capabilities)))

    # disable/enable
    helper.print_title("Disable/Enable")
    capability = Capability(member="UNARY_ATTRIBUTES")
    capabilities.disable(capability)
    capabilities.min_instances = 10
    print("Removing: " + str(capability))
    print(capabilities)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
