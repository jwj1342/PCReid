import yaml
import os


class ExperimentConfig:
    def __init__(self, config_file=None):
        # 设置默认配置文件路径
        self.default_config_file = 'config/default_config.yaml'
        # 如果用户没有提供配置文件，则使用默认配置文件
        if config_file is None:
            config_file = self.default_config_file
        self.config = self.load_yaml_config(config_file)

    def load_yaml_config(self, path):
        """
        从给定路径加载YAML配置文件。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件{path}不存在。")
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def get_config(self):
        """
        返回配置字典。
        """
        return self.config


if __name__ == '__main__':
    """
    测试配置文件是否正确读写
    """
    config = ExperimentConfig(config_file='default_config.yaml').get_config()
    print(config['Training_Settings'])
