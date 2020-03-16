import traceback
import weka.core.jvm as jvm
from weka.core.dataset import Attribute, Instance, Instances


def main():
    """
    Creates a dataset from scratch using random data and outputs it.
    """

    atts = []
    for i in range(5):
        atts.append(Attribute.create_numeric("x" + str(i)))

    data = Instances.create_instances("data", atts, 10)

    for n in range(10):
        values = []
        for i in range(5):
            values.append(n*100 + i)
        inst = Instance.create_instance(values)
        data.add_instance(inst)

    print(data)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
