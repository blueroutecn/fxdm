import traceback
import os
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader, TextDirectoryLoader


def main():
    """
    Just runs some example code.
    """

    # load ARFF file
    helper.print_title("Loading ARFF file")
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(helper.get_data_dir() + os.sep + "iris.arff")
    print(str(data))

    # load CSV file
    helper.print_title("Loading CSV file")
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(helper.get_data_dir() + os.sep + "iris.csv")
    print(str(data))

    # load directory
    # changes this to something sensible
    text_dir = "/some/where"
    if os.path.exists(text_dir) and os.path.isdir(text_dir):
        helper.print_title("Loading directory: " + text_dir)
        loader = TextDirectoryLoader(options=["-dir", text_dir, "-F", "-charset", "UTF-8"])
        data = loader.load()
        print(unicode(data))

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
