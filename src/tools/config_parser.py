import os
import json


class ConfigParser():


    CONFIG_SCHEMA={
      "Description": str,
      "Models": {
        "model_1": {
          "module": str,
          "object": str,
          "parameters": dict
        },
        "model_2": {
          "module": str,
          "object": str,
          "parameters": dict
        },
        "model_3": {
          "module": str,
          "object": str,
          "parameters": dict
        }
      },
      "Experiments": [
        {
          "module": str,
          "object": str,
          "parameters": dict
        }
      ],
      "Datasets": [
        {
          "name": str,
          "url": str,
          "sep": str,
          "function": str,
          "file": str
        }
      ]
    }

    def __init__(self, config_file = 'config.json'):

        try:
            with open(config_file) as file:
                config = json.load(file)

        except Exception as error:
            raise Exception(f'{config_file} has not a proper json structure')

        try:
            self._check_json_schema(config)

        except Exception as error:
            raise Exception(f'Error {config_file} does not follow the'+ \
                             'expected json schema')

        try:
            self._check_json_types(config)

        except Exception as error:
            raise Exception(f'Error {config_file} fields does not has the'+\
                             'expected value types')

        self._config = config


    @property
    def config(self):
        return self._config


    def _check_json_schema(self,config):

        schema = self.CONFIG_SCHEMA

        assert config.keys() == \
               schema.keys()

        assert config['Models'].keys() == \
               schema['Models'].keys()

        assert config['Models']['model_1'].keys() == \
               schema['Models']['model_1'].keys()

        assert config['Models']['model_2'].keys() == \
               schema['Models']['model_2'].keys()

        assert all(experiment.keys()==schema['Experiments'][0].keys() for
                        experiment in config['Experiments'])

        assert all(dataset.keys()==schema['Datasets'][0].keys() for
                        dataset in config['Datasets'])


    def _check_json_types(self,config):

        schema = self.CONFIG_SCHEMA

        assert all(type(val) == schema['Models']['model_1'][key] for
                   key, val  in config['Models']['model_1'].items())

        assert all(type(val) == schema['Models']['model_2'][key] for
                   key, val  in config['Models']['model_2'].items())

        for experiment in config['Experiments']:
            assert(type(val) == schema['Experiments'][0][key] for
                   key, val  in experiment)

        for dataset in config['Datasets']:
            assert(type(val) == schema['Datasets'][0][key] for
                   key, val  in dataset)
