import configparser


class ConfigReader():

    def readConfig(self, configName, section='DEFAULT'):
        config = configparser.ConfigParser()
        config.read("config.ini")
        value = config.get(section, configName)
        return value
