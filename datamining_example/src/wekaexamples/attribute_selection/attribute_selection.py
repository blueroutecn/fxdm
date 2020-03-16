import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    anneal_file = helper.get_data_dir() + os.sep + "anneal.arff"
    helper.print_info("Loading dataset: " + anneal_file)
    loader = Loader("weka.core.converters.ArffLoader")
    anneal_data = loader.load_file(anneal_file)
    anneal_data.class_is_last()

    # perform attribute selection
    helper.print_title("Attribute selection")
    search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
    evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(anneal_data)
    print("# attributes: " + str(attsel.number_attributes_selected))
    print("attributes (as numpy array): " + str(attsel.selected_attributes))
    print("attributes (as list): " + str(list(attsel.selected_attributes)))
    print("result string:\n" + attsel.results_string)

    # perform ranking
    helper.print_title("Attribute ranking (2-fold CV)")
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation("weka.attributeSelection.InfoGainAttributeEval")
    attsel = AttributeSelection()
    attsel.ranking(True)
    attsel.folds(2)
    attsel.crossvalidation(True)
    attsel.seed(42)
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(anneal_data)
    print("ranked attributes:\n" + str(attsel.ranked_attributes))
    print("result string:\n" + attsel.results_string)

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
