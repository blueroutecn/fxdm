import traceback
import weka.core.jvm as jvm
import weka.core.version as version


def main():
    """
    Just runs some example code.
    """

    print(version.weka_version())


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
