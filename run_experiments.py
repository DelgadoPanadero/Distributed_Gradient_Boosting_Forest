from src.tools import ConfigParser
from src.tools import ExperimentRunner


if __name__ == '__main__':

    config_parser = ConfigParser('config.json')
    config = config_parser.config

    runner = ExperimentRunner(config)
    runner.run()

