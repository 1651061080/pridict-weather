import sys


class StdoutRedirection:

    def __init__(self, paths_MODEL_FOLDER):

        self.stdout = sys.stdout
        self.log_path = paths_MODEL_FOLDER + "model_params.log"
        self.log_file = open(self.log_path, "w")

    def redirect_to_file(self):
        self.log_file = open(self.log_path, "w")
        sys.stdout = self.log_file

    def redirect_to_stdout(self):
        sys.stdout = self.stdout
        self.log_file.close()
